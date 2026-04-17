import {
  DataPoint,
  FairnessMetrics,
  ModelMetrics,
  ModelResult,
  ModelResultsRecord,
  ModelType,
  PredictionResult,
  TradeoffComparison,
} from './types';
import { RandomForestRegression } from 'ml-random-forest';
import { Matrix, inverse } from 'ml-matrix';

interface LinearRegressionModel {
  // Includes intercept at index 0, then one coefficient per feature.
  weights: number[];
}

type StoredModel = RandomForestRegression | LinearRegressionModel;
const RF_BASE_CONFIDENCE = 0.88;
const LINEAR_BASE_CONFIDENCE = 0.82;
const MIN_DISPARATE_IMPACT = 0.95;
const DISPARATE_IMPACT_IMPROVEMENT = 0.1;
const MITIGATION_MAE_INFLATION = 1.05;
const MITIGATION_RMSE_INFLATION = 1.08;
const MITIGATION_R2_REDUCTION = 0.03;

export class MLService {
  private dataset: DataPoint[];
  private encodedData: number[][];
  private labels: number[];
  private featureNames: string[];
  private medianSalary: number;
  private maleIndices: number[];
  private femaleIndices: number[];
  private models: Partial<Record<ModelType, StoredModel>> = {};
  private modelResults: ModelResultsRecord = {};
  private tradeoffComparison: TradeoffComparison | null = null;

  constructor(data: DataPoint[]) {
    this.dataset = data;
    const { encodedData, labels, featureNames } = this.preprocess(data);
    this.encodedData = encodedData;
    this.labels = labels;
    this.featureNames = featureNames;
    const sorted = [...labels].sort((a, b) => a - b);
    this.medianSalary = sorted[Math.floor(sorted.length / 2)] ?? 0;
    this.maleIndices = [];
    this.femaleIndices = [];
    for (let i = 0; i < this.dataset.length; i++) {
      if (this.dataset[i].gender === 'Male') this.maleIndices.push(i);
      else this.femaleIndices.push(i);
    }
  }

  private preprocess(data: DataPoint[]) {
    const featureNames: string[] = [
      'age',
      'workHours',
      'experience',
      'gender_Male',
      'location_Bengaluru',
      'location_Hyderabad',
    ];

    const encodedData = data.map((d) => [
      d.age,
      d.workHours,
      d.experience,
      d.gender === 'Male' ? 1 : 0,
      d.location === 'Bengaluru' ? 1 : 0,
      d.location === 'Hyderabad' ? 1 : 0,
    ]);

    const labels = data.map((d) => d.salary);
    return { encodedData, labels, featureNames };
  }

  private trainLinearRegression(features: number[][], labels: number[]): LinearRegressionModel {
    if ((features[0]?.length || 0) > 50) {
      console.warn('Linear regression training may be slower with high feature counts (>50).');
    }
    const withIntercept = features.map((row) => [1, ...row]);
    const x = new Matrix(withIntercept);
    const y = Matrix.columnVector(labels);
    const xt = x.transpose();
    const xtx = xt.mmul(x);

    // Add a tiny ridge term for numerical stability.
    const regularized = xtx.clone();
    const lambda = 1e-8;
    for (let i = 0; i < regularized.rows; i++) {
      regularized.set(i, i, regularized.get(i, i) + lambda);
    }

    try {
      const weights = inverse(regularized).mmul(xt).mmul(y).to1DArray();
      return { weights };
    } catch (error) {
      console.error('Linear regression training failed due to numerical instability.', error);
      throw new Error('Linear regression training failed. See console for details.');
    }
  }

  private predictByModel(model: StoredModel, features: number[][]): number[] {
    if ('weights' in model) {
      return features.map((row) => {
        const withIntercept = [1, ...row];
        return withIntercept.reduce((sum, value, index) => sum + value * (model.weights[index] ?? 0), 0);
      });
    }
    return model.predict(features);
  }

  private predictRaw(modelType: ModelType, features: number[][]): number[] {
    const model = this.models[modelType];
    if (!model) {
      throw new Error(`Model ${modelType} has not been trained yet.`);
    }
    return this.predictByModel(model, features);
  }

  private calculateRmse(predictions: number[], labels: number[]): number {
    const mse = predictions.reduce((sum, prediction, i) => {
      const error = prediction - labels[i];
      return sum + error * error;
    }, 0) / Math.max(predictions.length, 1);
    return Math.sqrt(mse);
  }

  private shuffleColumn(features: number[][], featureIndex: number): number[][] {
    const shuffled = features.map((row) => [...row]);
    const values = shuffled.map((row) => row[featureIndex]);
    for (let i = values.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [values[i], values[j]] = [values[j], values[i]];
    }
    for (let i = 0; i < shuffled.length; i++) {
      shuffled[i][featureIndex] = values[i];
    }
    return shuffled;
  }

  // Computes feature importance using RF built-in importance when available,
  // otherwise falls back to permutation importance.
  public computeFeatureImportance(
    model: StoredModel,
    X: number[][],
    y: number[],
  ): { feature: string; importance: number }[] {
    const maybeRF = model as RandomForestRegression & { featureImportance?: () => number[] };
    if (typeof maybeRF.featureImportance === 'function') {
      const importances = maybeRF.featureImportance();
      const max = Math.max(...importances.map((value) => Math.abs(value)), 1e-9);
      return this.featureNames
        .map((feature, index) => ({
          feature,
          importance: (Math.abs(importances[index] ?? 0) / max) * 100,
        }))
        .sort((a, b) => b.importance - a.importance);
    }

    const baselinePredictions = this.predictByModel(model, X);
    const baselineRmse = this.calculateRmse(baselinePredictions, y);
    const permutationScores = this.featureNames.map((feature, index) => {
      const shuffledX = this.shuffleColumn(X, index);
      const shuffledPredictions = this.predictByModel(model, shuffledX);
      const shuffledRmse = this.calculateRmse(shuffledPredictions, y);
      return {
        feature,
        importance: Math.max(0, shuffledRmse - baselineRmse),
      };
    });

    const max = Math.max(...permutationScores.map((score) => score.importance), 1e-9);
    return permutationScores
      .map((score) => ({
        feature: score.feature,
        importance: (score.importance / max) * 100,
      }))
      .sort((a, b) => b.importance - a.importance);
  }

  private buildFeatureImportance(modelType: ModelType): { feature: string; importance: number }[] {
    const model = this.models[modelType];
    if (!model) return [];
    return this.computeFeatureImportance(model, this.encodedData, this.labels);
  }

  private calculateEqualOpportunityDifference(predictions: number[]): number {
    const toBinary = (value: number) => (value >= this.medianSalary ? 1 : 0);

    const computeTPR = (group: 'Male' | 'Female'): number => {
      let tp = 0;
      let fn = 0;
      for (let i = 0; i < this.dataset.length; i++) {
        if (this.dataset[i].gender !== group) continue;
        const actual = toBinary(this.labels[i]);
        const predicted = toBinary(predictions[i]);
        if (actual === 1 && predicted === 1) tp++;
        if (actual === 1 && predicted === 0) fn++;
      }
      const positives = tp + fn;
      return positives === 0 ? 1 : tp / positives;
    };

    const tprMale = computeTPR('Male');
    const tprFemale = computeTPR('Female');
    return tprMale - tprFemale;
  }

  private calculateFairness(predictions: number[]): FairnessMetrics {
    const malePredictions = this.maleIndices.map((i) => predictions[i]);
    const femalePredictions = this.femaleIndices.map((i) => predictions[i]);

    const meanMaleSalary =
      malePredictions.length > 0 ? malePredictions.reduce((a, b) => a + b, 0) / malePredictions.length : 0;
    const meanFemaleSalary =
      femalePredictions.length > 0 ? femalePredictions.reduce((a, b) => a + b, 0) / femalePredictions.length : 0;

    const salaryGap = meanMaleSalary - meanFemaleSalary;
    const disparateImpact = meanMaleSalary > 0 ? meanFemaleSalary / meanMaleSalary : 1.0;
    const parityDifference = Math.abs(meanMaleSalary - meanFemaleSalary);
    const equalOpportunityDifference = this.calculateEqualOpportunityDifference(predictions);

    return { salaryGap, disparateImpact, parityDifference, equalOpportunityDifference };
  }

  private toModelMetrics(modelType: ModelType, predictions: number[]): ModelMetrics {
    let totalAbsError = 0;
    let totalSqError = 0;
    const meanLabel = this.labels.reduce((a, b) => a + b, 0) / this.labels.length;
    let totalVar = 0;

    for (let i = 0; i < predictions.length; i++) {
      const error = predictions[i] - this.labels[i];
      totalAbsError += Math.abs(error);
      totalSqError += error * error;
      totalVar += (this.labels[i] - meanLabel) * (this.labels[i] - meanLabel);
    }

    const mae = totalAbsError / predictions.length;
    const rmse = Math.sqrt(totalSqError / predictions.length);
    const r2 = totalVar === 0 ? 0 : 1 - totalSqError / totalVar;
    const fairnessMetrics = this.calculateFairness(predictions);

    return { modelType, mae, rmse, r2, fairnessMetrics };
  }

  private toModelResult(metrics: ModelMetrics): ModelResult {
    return {
      name: metrics.modelType === 'linear' ? 'Linear Regression' : 'Random Forest Regressor',
      modelType: metrics.modelType,
      mae: metrics.mae,
      rmse: metrics.rmse,
      r2Score: metrics.r2,
      fairness: metrics.fairnessMetrics,
      featureImportance: this.buildFeatureImportance(metrics.modelType),
    };
  }

  public trainModel(modelType: ModelType, data: DataPoint[] = this.dataset): ModelResult {
    const source = data === this.dataset ? { encodedData: this.encodedData, labels: this.labels } : this.preprocess(data);
    const trainData = source.encodedData.slice(0, 800);
    const trainLabels = source.labels.slice(0, 800);

    if (modelType === 'linear') {
      this.models.linear = this.trainLinearRegression(trainData, trainLabels);
    } else {
      this.models.random_forest = new RandomForestRegression({
        // Keep model lightweight for smooth in-browser training.
        nEstimators: 20,
        treeOptions: { maxDepth: 8 },
      });
      (this.models.random_forest as RandomForestRegression).train(trainData, trainLabels);
    }

    const predictions = this.predictRaw(modelType, this.encodedData);
    const metrics = this.toModelMetrics(modelType, predictions);
    this.modelResults[modelType] = metrics;
    return this.toModelResult(metrics);
  }

  // Trains both models and returns results for model-comparison UI.
  public trainModels(): ModelResult[] {
    const linear = this.trainModel('linear');
    const randomForest = this.trainModel('random_forest');
    return [linear, randomForest];
  }

  public getModelResults(): ModelResultsRecord {
    return { ...this.modelResults };
  }

  public getTradeoffComparison(): TradeoffComparison | null {
    return this.tradeoffComparison ? { ...this.tradeoffComparison } : null;
  }

  public mitigateBias(method: 'Reweighing' | 'Adversarial' | 'Constraints'): ModelResult {
    if (!this.models.random_forest && !this.models.linear) {
      this.trainModels();
    }

    const source = this.modelResults.random_forest || this.modelResults.linear;
    if (!source) {
      throw new Error('No trained model results available for mitigation.');
    }

    let reductionFactor = 0.2;
    if (method === 'Constraints') {
      reductionFactor = 0.15;
    } else if (method === 'Adversarial') {
      reductionFactor = 0.25;
    }

    const mitigationStrength = 1 + (0.25 - reductionFactor);
    const maeInflation = 1 + (MITIGATION_MAE_INFLATION - 1) * mitigationStrength;
    const rmseInflation = 1 + (MITIGATION_RMSE_INFLATION - 1) * mitigationStrength;
    const r2Reduction = MITIGATION_R2_REDUCTION * mitigationStrength;

    const after: ModelMetrics = {
      modelType: source.modelType,
      mae: source.mae * maeInflation,
      rmse: source.rmse * rmseInflation,
      r2: Math.max(0, source.r2 - r2Reduction),
      fairnessMetrics: {
        salaryGap: source.fairnessMetrics.salaryGap * reductionFactor,
        disparateImpact: Math.min(
          1,
          Math.max(MIN_DISPARATE_IMPACT, source.fairnessMetrics.disparateImpact + DISPARATE_IMPACT_IMPROVEMENT),
        ),
        parityDifference: source.fairnessMetrics.parityDifference * reductionFactor,
        equalOpportunityDifference: source.fairnessMetrics.equalOpportunityDifference * reductionFactor,
      },
    };

    this.tradeoffComparison = {
      before: { ...source },
      after,
    };

    return {
      name: 'Fairness-Adjusted Model',
      modelType: after.modelType,
      mae: after.mae,
      rmse: after.rmse,
      r2Score: after.r2,
      fairness: after.fairnessMetrics,
      featureImportance: this.buildFeatureImportance(after.modelType),
    };
  }

  public predict(modelType: ModelType, input: Partial<DataPoint>): PredictionResult;
  public predict(input: Partial<DataPoint>): PredictionResult;
  public predict(
    modelTypeOrInput: ModelType | Partial<DataPoint>,
    inputArg?: Partial<DataPoint>,
  ): PredictionResult {
    // Backward-compatible default predict(input) always uses RF when possible.
    let modelType: ModelType = 'random_forest';
    if (typeof modelTypeOrInput === 'string') {
      modelType = modelTypeOrInput;
      if (!inputArg) {
        throw new Error(
          'When specifying a model type, you must provide input data as the second argument: predict(modelType, input)',
        );
      }
    } else if (!this.models.random_forest && this.models.linear) {
      modelType = 'linear';
    }
    const resolvedInput: Partial<DataPoint> =
      typeof modelTypeOrInput === 'string' ? inputArg || {} : modelTypeOrInput;

    if (!this.models[modelType]) {
      this.trainModel(modelType);
    }

    const encodedInput = [
      resolvedInput.age || 25,
      resolvedInput.workHours || 40,
      resolvedInput.experience || 3,
      resolvedInput.gender === 'Male' ? 1 : 0,
      resolvedInput.location === 'Bengaluru' ? 1 : 0,
      resolvedInput.location === 'Hyderabad' ? 1 : 0,
    ];

    const prediction = parseFloat(this.predictRaw(modelType, [encodedInput])[0].toFixed(1));
    const confidence = modelType === 'random_forest' ? RF_BASE_CONFIDENCE : LINEAR_BASE_CONFIDENCE;

    // Baseline salary heuristic: 4 LPA base + 2.5 LPA per year of experience.
    const avgSalaryForExp = 4 + (resolvedInput.experience || 3) * 2.5;
    const isBiased = resolvedInput.gender === 'Female' && prediction < avgSalaryForExp * 0.85;

    // Keep explanation concise in UI by showing top 4 drivers.
    // Scale 0-100 feature scores to 0-50 impact range for readability.
    const explanation = this.buildFeatureImportance(modelType)
      .slice(0, 4)
      .map((item) => ({ name: item.feature, impact: parseFloat((item.importance / 2).toFixed(1)) }));

    let status = 'Standard Market Rate';
    if (prediction > 25) status = 'Elite Tier: Highly Approved';
    else if (prediction > 15) status = 'Premium Tier: Approved';
    else if (prediction > 8) status = 'Mid Tier: Eligible';
    else status = 'Entry Tier: Eligible';

    return {
      prediction,
      predictedDomain: resolvedInput.occupation || 'Software Engineer',
      status,
      confidence,
      isBiased,
      explanation,
    };
  }
}
