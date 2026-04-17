export interface DataPoint {
  age: number;
  gender: 'Male' | 'Female';
  education: string;
  occupation: string; // Software Engineer, DevOps, etc.
  location: string; // Bengaluru, Hyderabad, etc.
  workHours: number;
  experience: number;
  salary: number; // Approximate salary in Lakhs per Annum (LPA)
  [key: string]: any;
}

export type ModelType = 'linear' | 'random_forest';

export interface FairnessMetrics {
  salaryGap: number; // Difference in mean predicted salary between groups
  disparateImpact: number; // Ratio of mean predicted salary
  parityDifference: number; // Absolute difference in mean predicted salary
}

export interface ModelResult {
  name: string;
  modelType: ModelType;
  mae: number; // Mean Absolute Error
  rmse: number; // Root Mean Square Error
  r2Score: number; // R-squared score
  fairness: FairnessMetrics;
  featureImportance: { name: string; value: number }[];
}

export interface ModelMetrics {
  modelType: ModelType;
  mae: number;
  rmse: number;
  r2: number;
  fairnessMetrics: FairnessMetrics;
}

export type ModelResultsRecord = Partial<Record<ModelType, ModelMetrics>>;

export interface PredictionResult {
  prediction: number; // Predicted salary in LPA
  predictedDomain: string; // Predicted interested domain
  status: string; // Human-readable status (e.g., "Highly Competitive")
  confidence: number;
  isBiased: boolean;
  explanation: { name: string; impact: number }[];
}
