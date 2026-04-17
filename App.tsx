import React, { useState, useEffect, useMemo } from 'react';
import { 
  LayoutDashboard, 
  ShieldAlert, 
  Scale, 
  BarChart3, 
  UserPlus, 
  Info, 
  AlertCircle,
  CheckCircle2,
  ArrowRight,
  RefreshCw,
  BrainCircuit,
  Database,
  Download
} from 'lucide-react';
import { 
  BarChart, 
  Bar, 
  XAxis, 
  YAxis, 
  CartesianGrid, 
  Tooltip, 
  ResponsiveContainer, 
  Cell,
  PieChart,
  Pie,
  Legend,
  RadarChart,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  Radar,
  LineChart,
  Line
} from 'recharts';
import Papa from 'papaparse';
import { generateSyntheticData } from './dataset';
import { MLService } from './mlService';
import { ResumeService } from './resumeService';
import { DataPoint, ModelResult, ModelType, PredictionResult } from './types';
import { cn } from './utils';
import { motion, AnimatePresence } from 'motion/react';

const COLORS = ['#6366f1', '#f43f5e', '#10b981', '#f59e0b', '#8b5cf6'];

export default function App() {
  const [activeTab, setActiveTab] = useState<'overview' | 'training' | 'explainability' | 'bias' | 'mitigation' | 'predict'>('overview');
  const [dataset, setDataset] = useState<DataPoint[]>([]);
  const [models, setModels] = useState<ModelResult[]>([]);
  const [mitigatedModel, setMitigatedModel] = useState<ModelResult | null>(null);
  const [mlService, setMlService] = useState<MLService | null>(null);
  const [isTraining, setIsTraining] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [resumeText, setResumeText] = useState('');
  
  // Prediction Form State
  const [formData, setFormData] = useState<Partial<DataPoint>>({
    age: 28,
    gender: 'Male',
    education: 'B.Tech',
    occupation: 'Software Engineer',
    location: 'Bengaluru',
    workHours: 45,
    experience: 5
  });
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [whatIfResult, setWhatIfResult] = useState<PredictionResult | null>(null);
  const [showWhatIf, setShowWhatIf] = useState(false);
  const [selectedModelType, setSelectedModelType] = useState<ModelType>('random_forest');
  const [tradeoffData, setTradeoffData] = useState<ReturnType<MLService['getTradeoffComparison']>>(null);
  const [fairnessPercentile, setFairnessPercentile] = useState(50);

  useEffect(() => {
    fetch('/Dataset.csv')
      .then(res => res.text())
      .then(csvText => {
        Papa.parse(csvText, {
          header: true,
          dynamicTyping: true,
          skipEmptyLines: true,
          complete: (results) => {
            const parsedData = results.data
              .filter((row: any) => row.age != null)
              .map((row: any) => ({
                age: Number(row.age) || 25,
                gender: row.gender === 'Female' ? 'Female' : 'Male',
                education: row.education || 'B.Tech',
                occupation: row.occupation || 'Software Engineer',
                location: row.location || 'Bengaluru',
                workHours: Number(row.workHours) || 40,
                experience: Number(row.experience) || 3,
                salary: Number(row.salary) || 5.0
              })) as DataPoint[];
            
            setDataset(parsedData);
            const service = new MLService(parsedData);
            service.setFairnessThresholdPercentile(50);
            setMlService(service);
          }
        });
      })
      .catch(err => {
        console.error("Error loading default dataset:", err);
        const data = generateSyntheticData(500);
        setDataset(data);
        const service = new MLService(data);
        service.setFairnessThresholdPercentile(50);
        setMlService(service);
      });
  }, []);

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    Papa.parse(file, {
      header: true,
      dynamicTyping: true,
      complete: (results) => {
        const parsedData = results.data.map((row: any) => ({
          age: Number(row.age) || 25,
          gender: row.gender === 'Female' ? 'Female' : 'Male',
          education: row.education || 'B.Tech',
          occupation: row.occupation || 'Software Engineer',
          location: row.location || 'Bengaluru',
          workHours: Number(row.workHours) || 40,
          experience: Number(row.experience) || 3,
          salary: Number(row.salary) || 5.0
        })) as DataPoint[];
        
        setDataset(parsedData);
        const service = new MLService(parsedData);
        service.setFairnessThresholdPercentile(fairnessPercentile);
        setMlService(service);
        setModels([]);
        setMitigatedModel(null);
        setTradeoffData(null);
        setIsUploading(false);
        setActiveTab('overview');
      }
    });
  };

  const handleTrain = () => {
    if (!mlService) return;
    setIsTraining(true);
    setTimeout(() => {
      mlService.setFairnessThresholdPercentile(fairnessPercentile);
      const results = mlService.trainModels();
      setModels(results);
      setTradeoffData(null);
      setIsTraining(false);
      setActiveTab('training');
    }, 1500);
  };

  const handleMitigate = () => {
    if (!mlService) return;
    const result = mlService.mitigateBias('Reweighing');
    setMitigatedModel(result);
    setTradeoffData(mlService.getTradeoffComparison());
    setActiveTab('mitigation');
  };

  const handleThresholdChange = (nextPercentile: number) => {
    setFairnessPercentile(nextPercentile);
    if (!mlService) return;
    mlService.setFairnessThresholdPercentile(nextPercentile);
    if (models.length === 0) return;
    const updatedModels = mlService.recomputeTrainedModelResults();
    setModels(updatedModels);
    if (mitigatedModel) {
      const updatedMitigation = mlService.mitigateBias('Reweighing');
      setMitigatedModel(updatedMitigation);
      setTradeoffData(mlService.getTradeoffComparison());
    }
  };

  const handleDownloadReport = () => {
    if (!mlService || models.length === 0) return;
    const report = {
      generatedAt: new Date().toISOString(),
      threshold: {
        percentile: fairnessPercentile,
        salaryValue: mlService.getFairnessThresholdValue(),
      },
      models,
      mitigatedModel,
      tradeoffData,
      topFeatures: primaryModel?.featureImportance.slice(0, 10) || [],
    };
    const blob = new Blob([JSON.stringify(report, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `fairness-report-${Date.now()}.json`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handlePredict = (e: React.FormEvent) => {
    e.preventDefault();
    if (!mlService) return;
    const res = mlService.predict(selectedModelType, formData);
    setPrediction(res);
    
    // What-If Analysis: Swap gender
    const whatIfData = { ...formData, gender: formData.gender === 'Male' ? 'Female' : 'Male' } as Partial<DataPoint>;
    const whatIfRes = mlService.predict(selectedModelType, whatIfData);
    setWhatIfResult(whatIfRes);
  };

  const handleAnalyzeResume = async () => {
    if (!resumeText.trim()) return;
    setIsAnalyzing(true);
    try {
      const resumeService = new ResumeService();
      const data = await resumeService.analyzeResume(resumeText);
      setFormData(prev => ({
        ...prev,
        age: data.age,
        gender: data.gender,
        education: data.education,
        occupation: data.occupation,
        location: data.location,
        workHours: data.workHours,
        experience: data.experience
      }));
    } catch (error) {
      console.error("Analysis failed", error);
    } finally {
      setIsAnalyzing(false);
    }
  };

  const genderDistribution = useMemo(() => {
    const male = dataset.filter(d => d.gender === 'Male').length;
    const female = dataset.filter(d => d.gender === 'Female').length;
    return [
      { name: 'Male', value: male },
      { name: 'Female', value: female }
    ];
  }, [dataset]);

  const salaryByGender = useMemo(() => {
    const maleData = dataset.filter(d => d.gender === 'Male');
    const femaleData = dataset.filter(d => d.gender === 'Female');
    
    const avgMale = maleData.length > 0 ? maleData.reduce((a, b) => a + b.salary, 0) / maleData.length : 0;
    const avgFemale = femaleData.length > 0 ? femaleData.reduce((a, b) => a + b.salary, 0) / femaleData.length : 0;
    
    return [
      { name: 'Male', avgSalary: parseFloat(avgMale.toFixed(1)) },
      { name: 'Female', avgSalary: parseFloat(avgFemale.toFixed(1)) }
    ];
  }, [dataset]);

  // Use RF as primary fairness view when available, otherwise fallback to first model.
  const primaryModel = useMemo(
    () => models.find((model) => model.modelType === 'random_forest') || models[0] || null,
    [models],
  );

  const bestModelSummary = useMemo(() => {
    if (models.length === 0) return null;
    const scored = [...models]
      .map((model) => {
        const fairnessPenalty =
          Math.abs(1 - model.fairness.disparateImpact) * 10 +
          Math.abs(model.fairness.equalOpportunityDifference) * 10 +
          model.fairness.parityDifference;
        return {
          model,
          combinedScore: model.rmse + fairnessPenalty,
        };
      })
      .sort((a, b) => a.combinedScore - b.combinedScore);
    return scored[0];
  }, [models]);

  return (
    <div className="min-h-screen bg-[#fafafa] text-[#1a1a1a] font-sans">
      {/* Sidebar */}
      <aside className="fixed left-0 top-0 h-full w-64 bg-white border-r border-gray-200 p-6 z-20">
        <div className="flex items-center gap-3 mb-10">
          <div className="w-10 h-10 bg-indigo-600 rounded-xl flex items-center justify-center text-white">
            <BrainCircuit size={24} />
          </div>
          <div>
            <h1 className="font-bold text-lg leading-tight">FairDecision</h1>
            <p className="text-[10px] text-gray-500 uppercase tracking-widest font-semibold">Bias Mitigation AI</p>
          </div>
        </div>

        <nav className="space-y-1">
          <NavItem 
            active={activeTab === 'overview'} 
            onClick={() => setActiveTab('overview')} 
            icon={<LayoutDashboard size={18} />} 
            label="Overview" 
          />
          <NavItem 
            active={activeTab === 'training'} 
            onClick={() => setActiveTab('training')} 
            icon={<BarChart3 size={18} />} 
            label="Model Training" 
            disabled={models.length === 0}
          />
          <NavItem 
            active={activeTab === 'explainability'} 
            onClick={() => setActiveTab('explainability')} 
            icon={<Info size={18} />} 
            label="Explainability" 
            disabled={models.length === 0}
          />
          <NavItem 
            active={activeTab === 'bias'} 
            onClick={() => setActiveTab('bias')} 
            icon={<ShieldAlert size={18} />} 
            label="Bias Analysis" 
            disabled={models.length === 0}
          />
          <NavItem 
            active={activeTab === 'mitigation'} 
            onClick={() => setActiveTab('mitigation')} 
            icon={<Scale size={18} />} 
            label="Mitigation" 
            disabled={models.length === 0}
          />
          <NavItem 
            active={activeTab === 'predict'} 
            onClick={() => setActiveTab('predict')} 
            icon={<UserPlus size={18} />} 
            label="Live Prediction" 
            disabled={models.length === 0}
          />
        </nav>

        <div className="absolute bottom-8 left-6 right-6">
          <div className="p-4 bg-indigo-50 rounded-2xl border border-indigo-100">
            <h4 className="text-xs font-bold text-indigo-900 mb-1">System Status</h4>
            <div className="flex items-center gap-2 text-[10px] text-indigo-700">
              <div className="w-1.5 h-1.5 bg-green-500 rounded-full animate-pulse" />
              Ready for Analysis
            </div>
            <div className="mt-3 pt-3 border-t border-indigo-100">
              <div className="flex items-center gap-2 text-[9px] font-bold text-indigo-400 uppercase tracking-tighter">
                <Database size={10} />
                Using: Dataset.csv
              </div>
            </div>
          </div>
        </div>
      </aside>

      {/* Main Content */}
      <main className="pl-64 min-h-screen">
        <header className="h-16 bg-white border-b border-gray-200 flex items-center justify-between px-8 sticky top-0 z-10">
          <h2 className="font-semibold text-gray-800">
            {activeTab === 'overview' && 'Overview Dashboard'}
            {activeTab === 'training' && 'Model Performance'}
            {activeTab === 'explainability' && 'Explainability Insights'}
            {activeTab === 'bias' && 'Fairness Audit'}
            {activeTab === 'mitigation' && 'Bias Mitigation Results'}
            {activeTab === 'predict' && 'Individual Decision Analysis'}
          </h2>
          
          <div className="flex items-center gap-4">
            <label className="cursor-pointer px-4 py-2 bg-white border border-gray-200 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors flex items-center gap-2">
              <UserPlus size={16} />
              {isUploading ? 'Uploading...' : 'Upload Dataset (CSV)'}
              <input type="file" accept=".csv" className="hidden" onChange={handleFileUpload} disabled={isUploading} />
            </label>
            <button 
              onClick={handleTrain}
              disabled={isTraining}
              className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-medium hover:bg-indigo-700 transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              {isTraining ? <RefreshCw size={16} className="animate-spin" /> : <BrainCircuit size={16} />}
              {models.length > 0 ? 'Retrain Models' : 'Train Models'}
            </button>
            <button
              onClick={handleDownloadReport}
              disabled={models.length === 0}
              className="px-4 py-2 bg-white border border-gray-200 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 transition-colors flex items-center gap-2 disabled:opacity-50"
            >
              <Download size={16} />
              Download Report
            </button>
          </div>
        </header>

        <div className="p-8 max-w-7xl mx-auto">
          {activeTab === 'overview' && (
            <div className="space-y-8">
              <div className="bg-gradient-to-r from-indigo-600 to-violet-700 p-8 rounded-3xl text-white shadow-xl relative overflow-hidden">
                <div className="relative z-10">
                  <div className="flex items-center gap-3 mb-4">
                    <div className="px-3 py-1 bg-white/20 backdrop-blur-md rounded-full text-[10px] font-black uppercase tracking-widest">
                      Mission Critical
                    </div>
                  </div>
                  <h1 className="text-3xl font-black mb-4 leading-tight">
                    Fairness in AI <br/>
                    <span className="text-indigo-200">Making Sure Our AI Treats Everyone Equally</span>
                  </h1>
                  <p className="text-indigo-100 text-sm max-w-2xl leading-relaxed">
                    This project uses a file called <strong>Dataset.csv</strong> to check if our Artificial Intelligence (AI) model makes fair decisions. 
                    Our goal is to make sure that computer systems used for hiring and deciding salaries treat everyone fairly. 
                    We want to prove that factors like gender or location don't give anyone an unfair advantage or disadvantage.
                  </p>
                </div>
                <div className="absolute top-0 right-0 w-64 h-64 bg-white/10 rounded-full -mr-20 -mt-20 blur-3xl" />
                <div className="absolute bottom-0 left-0 w-32 h-32 bg-indigo-400/20 rounded-full -ml-10 -mb-10 blur-2xl" />
              </div>

              {bestModelSummary && (
                <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-sm space-y-6">
                  <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4">
                    <div>
                      <h3 className="text-lg font-bold text-gray-900">Executive Overview</h3>
                      <p className="text-sm text-gray-500">
                        Best model selected by combined score (RMSE + fairness penalties).
                      </p>
                    </div>
                    <div className="px-3 py-1 bg-indigo-50 text-indigo-700 text-xs font-bold rounded-full w-fit">
                      Best Model: {bestModelSummary.model.name}
                    </div>
                  </div>

                  <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
                    <StatCard label="RMSE" value={bestModelSummary.model.rmse.toFixed(2)} sub="Lower is better" />
                    <StatCard label="R² Score" value={bestModelSummary.model.r2Score.toFixed(3)} sub="Higher is better" />
                    <StatCard
                      label="Fairness (DI)"
                      value={bestModelSummary.model.fairness.disparateImpact.toFixed(3)}
                      sub="Target near 1.0"
                    />
                    <div className="bg-white p-6 rounded-2xl border border-gray-200 shadow-sm">
                      <h4 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-2">Bias Indicator</h4>
                      <div className={cn(
                        "text-lg font-black mb-1",
                        bestModelSummary.model.fairness.disparateImpact < 0.8 ||
                        Math.abs(bestModelSummary.model.fairness.equalOpportunityDifference) > 0.1
                          ? "text-red-600"
                          : "text-green-600",
                      )}>
                        {bestModelSummary.model.fairness.disparateImpact < 0.8 ||
                        Math.abs(bestModelSummary.model.fairness.equalOpportunityDifference) > 0.1
                          ? 'Bias Risk'
                          : 'Within Range'}
                      </div>
                      <p className="text-xs text-gray-500">Uses disparate impact and equal opportunity checks.</p>
                    </div>
                  </div>

                  <div className="bg-gray-50 rounded-xl border border-gray-200 p-4">
                    <h4 className="text-xs font-bold text-gray-500 uppercase tracking-wider mb-3">Top 3 Features</h4>
                    <div className="grid grid-cols-1 md:grid-cols-3 gap-3">
                      {bestModelSummary.model.featureImportance.slice(0, 3).map((item) => (
                        <div key={item.feature} className="bg-white border border-gray-200 rounded-lg p-3">
                          <p className="text-sm font-bold text-gray-800">{item.feature}</p>
                          <p className="text-xs text-gray-500">Importance: {item.importance.toFixed(1)}%</p>
                          {typeof item.stdDev === 'number' && (
                            <p className="text-[11px] text-gray-400">Std Dev: {item.stdDev.toFixed(2)}%</p>
                          )}
                        </div>
                      ))}
                    </div>
                  </div>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <StatCard label="Total Records" value={dataset.length.toString()} sub="Rows in Dataset.csv" />
                <StatCard label="Information Fields" value="8" sub="Age, Location, Role, etc." />
                <StatCard label="Goal" value="Predict Salary" sub="Expected Yearly Pay" />
              </div>

              <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-sm flex flex-col md:flex-row gap-8 items-center">
                <div className="w-20 h-20 bg-indigo-100 rounded-2xl flex items-center justify-center text-indigo-600 shrink-0">
                  <Info size={40} />
                </div>
                <div>
                  <h3 className="text-lg font-bold text-gray-900 mb-2">Our Data: Dataset.csv</h3>
                  <p className="text-gray-600 text-sm leading-relaxed">
                    This application uses the <strong>Dataset.csv</strong> file to learn and make predictions. 
                    It contains simple information like a person's age, education, job role, and location in India. 
                    The AI uses this data to predict a person's expected salary. We use this dataset to check if our AI makes <strong>fair predictions</strong> for everyone.
                  </p>
                  <div className="mt-4 flex gap-4">
                    <a href="/Dataset.csv" download="Dataset.csv" className="px-4 py-2 bg-indigo-600 text-white rounded-lg text-sm font-bold hover:bg-indigo-700 transition-colors flex items-center gap-2">
                      <Database size={16} />
                      Download Dataset.csv
                    </a>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
                <div className="lg:col-span-2 bg-white p-6 rounded-2xl border border-gray-200 shadow-sm">
                  <div className="flex justify-between items-center mb-6">
                    <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider">Data Preview (Sample)</h3>
                    <span className="text-[10px] bg-gray-100 px-2 py-1 rounded font-bold text-gray-500 uppercase">First 5 Rows</span>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-left text-xs">
                      <thead>
                        <tr className="border-b border-gray-100">
                          <th className="pb-3 font-bold text-gray-400">Age</th>
                          <th className="pb-3 font-bold text-gray-400">Gender</th>
                          <th className="pb-3 font-bold text-gray-400">Occupation</th>
                          <th className="pb-3 font-bold text-gray-400">Location</th>
                          <th className="pb-3 font-bold text-gray-400">Experience</th>
                          <th className="pb-3 font-bold text-gray-400">Salary (LPA)</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-gray-50">
                        {dataset.slice(0, 5).map((row, i) => (
                          <tr key={i} className="hover:bg-gray-50 transition-colors">
                            <td className="py-3 text-gray-700">{row.age}</td>
                            <td className="py-3">
                              <span className={cn("px-2 py-0.5 rounded-full text-[10px] font-bold", row.gender === 'Male' ? "bg-blue-50 text-blue-600" : "bg-pink-50 text-pink-600")}>
                                {row.gender}
                              </span>
                            </td>
                            <td className="py-3 text-gray-600">{row.occupation}</td>
                            <td className="py-3 text-gray-600">{row.location}</td>
                            <td className="py-3 text-gray-600">{row.experience}y</td>
                            <td className="py-3">
                              <span className="font-bold text-indigo-600">
                                ₹{row.salary.toFixed(1)}L
                              </span>
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </div>

                <div className="bg-white p-6 rounded-2xl border border-gray-200 shadow-sm">
                  <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-6">Data Dictionary</h3>
                  <div className="space-y-4">
                    <div className="flex gap-3">
                      <div className="w-8 h-8 rounded bg-indigo-50 flex items-center justify-center text-indigo-600 shrink-0 text-xs font-bold">A</div>
                      <div>
                        <h4 className="text-xs font-bold text-gray-800">Age</h4>
                        <p className="text-[10px] text-gray-500">Continuous numerical value representing individual's age.</p>
                      </div>
                    </div>
                    <div className="flex gap-3">
                      <div className="w-8 h-8 rounded bg-pink-50 flex items-center justify-center text-pink-600 shrink-0 text-xs font-bold">G</div>
                      <div>
                        <h4 className="text-xs font-bold text-gray-800">Gender</h4>
                        <p className="text-[10px] text-gray-500">Sensitive attribute used for bias detection (Male/Female).</p>
                      </div>
                    </div>
                    <div className="flex gap-3">
                      <div className="w-8 h-8 rounded bg-amber-50 flex items-center justify-center text-amber-600 shrink-0 text-xs font-bold">L</div>
                      <div>
                        <h4 className="text-xs font-bold text-gray-800">Location</h4>
                        <p className="text-[10px] text-gray-500">Major IT hubs in India (Bengaluru, Hyderabad, Pune, etc.).</p>
                      </div>
                    </div>
                    <div className="flex gap-3">
                      <div className="w-8 h-8 rounded bg-green-50 flex items-center justify-center text-green-600 shrink-0 text-xs font-bold">S</div>
                      <div>
                        <h4 className="text-xs font-bold text-gray-800">Salary (Target)</h4>
                        <p className="text-[10px] text-gray-500">Numerical target: Annual salary in Lakhs Per Annum (LPA).</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-white p-6 rounded-2xl border border-gray-200 shadow-sm">
                  <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-6">Gender Distribution</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <PieChart>
                        <Pie
                          data={genderDistribution}
                          innerRadius={60}
                          outerRadius={80}
                          paddingAngle={5}
                          dataKey="value"
                        >
                          {genderDistribution.map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                        <Legend verticalAlign="bottom" height={36}/>
                      </PieChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                <div className="bg-white p-6 rounded-2xl border border-gray-200 shadow-sm">
                  <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-6">Avg Salary by Gender (Ground Truth)</h3>
                  <div className="h-64">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={salaryByGender}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                        <XAxis dataKey="name" axisLine={false} tickLine={false} />
                        <YAxis axisLine={false} tickLine={false} unit="L" />
                        <Tooltip cursor={{fill: '#f8fafc'}} />
                        <Legend />
                        <Bar dataKey="avgSalary" name="Avg Salary (LPA)" fill="#6366f1" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              </div>

              <div className="bg-indigo-900 text-white p-8 rounded-3xl relative overflow-hidden">
                <div className="relative z-10 max-w-2xl">
                  <h2 className="text-2xl font-bold mb-4">Why is Fairness Important in AI?</h2>
                  <p className="text-indigo-100 leading-relaxed mb-6">
                    Sometimes, computers learn bad habits from old data. If older data shows fewer women in high-paying jobs, 
                    the AI might mistakenly learn to predict lower salaries for women. 
                    Our system checks for these mistakes and fixes them, making sure the AI treats everyone equally.
                  </p>
                  <button 
                    onClick={handleTrain}
                    className="px-6 py-3 bg-white text-indigo-900 rounded-xl font-bold hover:bg-indigo-50 transition-colors"
                  >
                    Start Analysis
                  </button>
                </div>
                <div className="absolute right-0 top-0 bottom-0 w-1/3 bg-indigo-800/50 skew-x-12 translate-x-12" />
              </div>

              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-sm">
                  <h3 className="text-lg font-bold text-gray-900 mb-4">Social Utility & Impact</h3>
                  <div className="space-y-4">
                    <ImpactItem 
                      title="Fair Hiring" 
                      desc="Helps companies check if their computer systems are unfairly rejecting people based on their gender or age."
                    />
                    <ImpactItem 
                      title="Equal Pay" 
                      desc="Makes sure that everyone gets a fair salary suggestion, regardless of their background."
                    />
                    <ImpactItem 
                      title="Responsible Technology" 
                      desc="Shows that we care about building computer programs that do the right thing and treat people well."
                    />
                  </div>
                </div>
                <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-sm">
                  <h3 className="text-lg font-bold text-gray-900 mb-4">Advanced Features</h3>
                  <div className="space-y-4">
                    <div className="flex gap-4">
                      <div className="w-10 h-10 bg-indigo-50 rounded-lg flex items-center justify-center text-indigo-600 shrink-0">
                        <BrainCircuit size={20} />
                      </div>
                      <div>
                        <h4 className="font-bold text-sm text-gray-800">"What-If" Game</h4>
                        <p className="text-xs text-gray-500">Change a person's gender and see if the AI changes its decision.</p>
                      </div>
                    </div>
                    <div className="flex gap-4">
                      <div className="w-10 h-10 bg-indigo-50 rounded-lg flex items-center justify-center text-indigo-600 shrink-0">
                        <Scale size={20} />
                      </div>
                      <div>
                        <h4 className="font-bold text-sm text-gray-800">Fixing the AI</h4>
                        <p className="text-xs text-gray-500">We teach the AI to forget bad habits and be completely fair.</p>
                      </div>
                    </div>
                    <div className="flex gap-4">
                      <div className="w-10 h-10 bg-indigo-50 rounded-lg flex items-center justify-center text-indigo-600 shrink-0">
                        <BarChart3 size={20} />
                      </div>
                      <div>
                        <h4 className="font-bold text-sm text-gray-800">Easy to Understand</h4>
                        <p className="text-xs text-gray-500">We show exactly why the AI made its decision using simple charts.</p>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'training' && (
            <div className="space-y-8">
              <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
                {models.map((m, i) => (
                  <div key={i} className="bg-white p-6 rounded-2xl border border-gray-200 shadow-sm hover:border-indigo-300 transition-all group">
                    <div className="flex justify-between items-start mb-4">
                      <h3 className="font-bold text-gray-800">{m.name}</h3>
                      <div className="px-2 py-1 bg-indigo-50 text-indigo-600 text-[10px] font-bold rounded uppercase">Trained</div>
                    </div>
                    <div className="space-y-4">
                      <MetricRow label="MAE (LPA)" value={m.mae} isPercentage={false} />
                      <MetricRow label="RMSE (LPA)" value={m.rmse} isPercentage={false} />
                      <MetricRow label="R² Score" value={m.r2Score} isPercentage={false} />
                    </div>
                  </div>
                ))}
              </div>

              <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-sm">
                <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-8">Feature Importance (Explainable AI)</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart layout="vertical" data={primaryModel ? primaryModel.featureImportance : []}>
                      <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#f0f0f0" />
                      <XAxis type="number" hide />
                      <YAxis dataKey="feature" type="category" axisLine={false} tickLine={false} width={180} />
                      <Tooltip />
                      <Bar dataKey="importance" fill="#6366f1" radius={[0, 4, 4, 0]} barSize={30} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'explainability' && primaryModel && (
            <div className="space-y-8">
              <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-sm">
                <h3 className="text-lg font-bold text-gray-900 mb-2">Explainability</h3>
                <p className="text-sm text-gray-500 mb-8">
                  Top features ranked by model impact using built-in Random Forest importance or permutation importance fallback.
                </p>
                <div className="h-96">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={primaryModel.featureImportance}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                      <XAxis dataKey="feature" axisLine={false} tickLine={false} />
                      <YAxis axisLine={false} tickLine={false} />
                      <Tooltip formatter={(v: number) => `${v.toFixed(1)}%`} />
                      <Bar dataKey="importance" fill="#6366f1" radius={[6, 6, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="mt-6 overflow-x-auto">
                  <table className="min-w-full text-xs">
                    <thead>
                      <tr className="text-left text-gray-500 border-b border-gray-200">
                        <th className="py-2 pr-4 font-semibold">Feature</th>
                        <th className="py-2 pr-4 font-semibold">Importance</th>
                        <th className="py-2 pr-4 font-semibold">Std Dev</th>
                      </tr>
                    </thead>
                    <tbody className="text-gray-700">
                      {primaryModel.featureImportance.slice(0, 8).map((item) => (
                        <tr key={item.feature} className="border-b border-gray-100">
                          <td className="py-2 pr-4">{item.feature}</td>
                          <td className="py-2 pr-4">{item.importance.toFixed(2)}%</td>
                          <td className="py-2 pr-4">{typeof item.stdDev === 'number' ? `${item.stdDev.toFixed(2)}%` : '—'}</td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'bias' && primaryModel && (
            <div className="space-y-8">
              <div className="bg-white p-6 rounded-2xl border border-gray-200 shadow-sm">
                <div className="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-4">
                  <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider flex items-center gap-2">
                    Fairness Threshold Control
                    <span
                      title="Adjust the salary percentile used to convert regression outputs into binary outcomes for Equal Opportunity."
                      className="text-gray-400 cursor-help"
                    >
                      <Info size={13} />
                    </span>
                  </h3>
                  <p className="text-xs text-gray-500">
                    Percentile: <span className="font-bold text-gray-700">{fairnessPercentile}%</span>
                    {mlService && (
                      <>
                        {' '}| Salary Threshold: <span className="font-bold text-gray-700">₹{mlService.getFairnessThresholdValue().toFixed(2)}L</span>
                      </>
                    )}
                  </p>
                </div>
                <input
                  type="range"
                  min={10}
                  max={90}
                  step={1}
                  value={fairnessPercentile}
                  onChange={(e) => handleThresholdChange(Number(e.target.value))}
                  className="w-full accent-indigo-600"
                />
                <div className="flex justify-between text-[10px] text-gray-400 mt-2">
                  <span>P10</span>
                  <span>P50 (Median)</span>
                  <span>P90</span>
                </div>
              </div>

              <div className="grid grid-cols-1 lg:grid-cols-2 gap-8">
                <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-sm">
                  <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-6">Fairness Metrics Radar</h3>
                  <div className="h-80">
                    <ResponsiveContainer width="100%" height="100%">
                      <RadarChart cx="50%" cy="50%" outerRadius="80%" data={[
                        { subject: 'Fairness Score', A: primaryModel.fairness.disparateImpact * 100, full: 100 },
                        { subject: 'Salary Gap (Inv)', A: Math.max(0, (1 - (primaryModel.fairness.salaryGap / 10)) * 100), full: 100 },
                        { subject: 'Diff in Chances', A: Math.max(0, (1 - (primaryModel.fairness.parityDifference / 10)) * 100), full: 100 },
                        { subject: 'Equal Opportunity Score', A: Math.max(0, (1 - Math.abs(primaryModel.fairness.equalOpportunityDifference)) * 100), full: 100 },
                      ]}>
                        <PolarGrid />
                        <PolarAngleAxis dataKey="subject" />
                        <PolarRadiusAxis angle={30} domain={[0, 100]} />
                        <Radar name="Model Bias" dataKey="A" stroke="#f43f5e" fill="#f43f5e" fillOpacity={0.6} />
                      </RadarChart>
                    </ResponsiveContainer>
                  </div>
                  <p className="text-xs text-gray-500 mt-4 text-center italic">
                    Higher values indicate better fairness (closer to 100% ideal)
                  </p>
                </div>

                <div className="space-y-4">
                  <FairnessCard 
                    title="Disparate Impact (Ratio)" 
                    value={primaryModel.fairness.disparateImpact} 
                    threshold="0.8 - 1.25"
                    desc="Ratio of mean predicted salary for unprivileged (Female) vs privileged (Male) groups."
                    status={primaryModel.fairness.disparateImpact < 0.8 ? 'fail' : 'pass'}
                    tooltip="Closer to 1.0 indicates similar outcomes across groups."
                  />
                  <FairnessCard 
                    title="Salary Gap (LPA)" 
                    value={primaryModel.fairness.salaryGap} 
                    threshold="< 1.0"
                    desc="Difference in mean predicted salary between Male and Female groups."
                    status={primaryModel.fairness.salaryGap > 2.0 ? 'fail' : 'pass'}
                    tooltip="Absolute value close to 0 indicates fairness."
                  />
                  <FairnessCard 
                    title="Parity Difference" 
                    value={primaryModel.fairness.parityDifference} 
                    threshold="< 1.0"
                    desc="Absolute difference in mean outcomes across groups."
                    status={primaryModel.fairness.parityDifference > 2.0 ? 'fail' : 'pass'}
                    tooltip="Lower parity difference means better demographic parity."
                  />
                  <FairnessCard 
                    title="Equal Opportunity Difference" 
                    value={primaryModel.fairness.equalOpportunityDifference} 
                    threshold="≈ 0"
                    desc={`Difference in true positive rates between Male and Female groups at P${fairnessPercentile} salary threshold.`}
                    status={Math.abs(primaryModel.fairness.equalOpportunityDifference) > 0.1 ? 'fail' : 'pass'}
                    tooltip="Calculated as TPR_male - TPR_female. Near zero is preferred."
                  />
                </div>
              </div>

              <div className="bg-red-50 border border-red-100 p-6 rounded-2xl flex gap-4">
                <AlertCircle className="text-red-600 shrink-0" />
                <div>
                  <h4 className="font-bold text-red-900 mb-1">Bias Detected</h4>
                  <p className="text-sm text-red-700">
                    The current model shows significant disparate impact against the female group. 
                    The selection rate for females is only {(primaryModel.fairness.disparateImpact * 100).toFixed(1)}% of the male selection rate, 
                    violating the "four-fifths rule" commonly used in legal and ethical audits.
                  </p>
                  <button 
                    onClick={handleMitigate}
                    className="mt-4 px-4 py-2 bg-red-600 text-white rounded-lg text-sm font-bold hover:bg-red-700 transition-colors"
                  >
                    Apply Mitigation Techniques
                  </button>
                </div>
              </div>
            </div>
          )}

          {activeTab === 'mitigation' && mitigatedModel && primaryModel && (
            <div className="space-y-8">
              <div className="grid grid-cols-1 md:grid-cols-2 gap-8">
                <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-sm relative overflow-hidden">
                  <div className="absolute top-0 right-0 p-4">
                    <div className="px-2 py-1 bg-red-50 text-red-600 text-[10px] font-bold rounded uppercase">Before</div>
                  </div>
                  <h3 className="font-bold text-gray-800 mb-6">Original Model</h3>
                  <div className="space-y-6">
                    <FairnessProgress label="Disparate Impact" value={primaryModel.fairness.disparateImpact} color="#f43f5e" />
                    <FairnessProgress label="Salary Equality" value={Math.max(0, 1 - (primaryModel.fairness.salaryGap / 15))} color="#f43f5e" />
                    <FairnessProgress label="Equal Opportunity Score" value={Math.max(0, 1 - Math.abs(primaryModel.fairness.equalOpportunityDifference))} color="#f43f5e" />
                  </div>
                </div>

                <div className="bg-white p-8 rounded-2xl border border-indigo-200 shadow-lg relative overflow-hidden ring-2 ring-indigo-500 ring-opacity-20">
                  <div className="absolute top-0 right-0 p-4">
                    <div className="px-2 py-1 bg-green-50 text-green-600 text-[10px] font-bold rounded uppercase">Mitigated</div>
                  </div>
                  <h3 className="font-bold text-indigo-900 mb-6">Fairness-Optimized Model</h3>
                  <div className="space-y-6">
                    <FairnessProgress label="Disparate Impact" value={mitigatedModel.fairness.disparateImpact} color="#10b981" />
                    <FairnessProgress label="Salary Equality" value={Math.max(0, 1 - (mitigatedModel.fairness.salaryGap / 15))} color="#10b981" />
                    <FairnessProgress label="Equal Opportunity Score" value={Math.max(0, 1 - Math.abs(mitigatedModel.fairness.equalOpportunityDifference))} color="#10b981" />
                  </div>
                </div>
              </div>

              <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-sm">
                <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-8">Fairness Improvement Comparison</h3>
                <div className="h-80">
                  <ResponsiveContainer width="100%" height="100%">
                    <BarChart data={[
                      { name: 'Disparate Impact', Original: primaryModel.fairness.disparateImpact, Mitigated: mitigatedModel.fairness.disparateImpact },
                      { name: 'Salary Equality', Original: Math.max(0, 1 - (primaryModel.fairness.salaryGap / 15)), Mitigated: Math.max(0, 1 - (mitigatedModel.fairness.salaryGap / 15)) },
                      { name: 'Equal Opportunity', Original: Math.max(0, 1 - Math.abs(primaryModel.fairness.equalOpportunityDifference)), Mitigated: Math.max(0, 1 - Math.abs(mitigatedModel.fairness.equalOpportunityDifference)) },
                    ]}>
                      <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                      <XAxis dataKey="name" axisLine={false} tickLine={false} />
                      <YAxis domain={[0, 1]} axisLine={false} tickLine={false} />
                      <Tooltip />
                      <Legend />
                      <Bar dataKey="Original" fill="#f43f5e" radius={[4, 4, 0, 0]} />
                      <Bar dataKey="Mitigated" fill="#10b981" radius={[4, 4, 0, 0]} />
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>

              {tradeoffData && (
                <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-sm space-y-6">
                  <h3 className="text-sm font-bold text-gray-500 uppercase tracking-wider">Accuracy vs Fairness Trade-off</h3>

                  <div className="overflow-x-auto">
                    <table className="min-w-full text-sm">
                      <thead>
                        <tr className="text-left text-gray-500 border-b border-gray-200">
                          <th className="py-2 pr-4 font-semibold">Metric</th>
                          <th className="py-2 pr-4 font-semibold">Before Mitigation</th>
                          <th className="py-2 pr-4 font-semibold">After Mitigation</th>
                        </tr>
                      </thead>
                      <tbody className="text-gray-700">
                        <tr className="border-b border-gray-100"><td className="py-2 pr-4">RMSE</td><td className="py-2 pr-4">{tradeoffData.before.rmse.toFixed(3)}</td><td className="py-2 pr-4">{tradeoffData.after.rmse.toFixed(3)}</td></tr>
                        <tr className="border-b border-gray-100"><td className="py-2 pr-4">MAE</td><td className="py-2 pr-4">{tradeoffData.before.mae.toFixed(3)}</td><td className="py-2 pr-4">{tradeoffData.after.mae.toFixed(3)}</td></tr>
                        <tr className="border-b border-gray-100"><td className="py-2 pr-4">R²</td><td className="py-2 pr-4">{tradeoffData.before.r2.toFixed(3)}</td><td className="py-2 pr-4">{tradeoffData.after.r2.toFixed(3)}</td></tr>
                        <tr className="border-b border-gray-100"><td className="py-2 pr-4">Disparate Impact</td><td className="py-2 pr-4">{tradeoffData.before.fairnessMetrics.disparateImpact.toFixed(3)}</td><td className="py-2 pr-4">{tradeoffData.after.fairnessMetrics.disparateImpact.toFixed(3)}</td></tr>
                        <tr className="border-b border-gray-100"><td className="py-2 pr-4">Salary Gap</td><td className="py-2 pr-4">{tradeoffData.before.fairnessMetrics.salaryGap.toFixed(3)}</td><td className="py-2 pr-4">{tradeoffData.after.fairnessMetrics.salaryGap.toFixed(3)}</td></tr>
                        <tr className="border-b border-gray-100"><td className="py-2 pr-4">Parity Difference</td><td className="py-2 pr-4">{tradeoffData.before.fairnessMetrics.parityDifference.toFixed(3)}</td><td className="py-2 pr-4">{tradeoffData.after.fairnessMetrics.parityDifference.toFixed(3)}</td></tr>
                        <tr><td className="py-2 pr-4">Equal Opportunity Difference</td><td className="py-2 pr-4">{tradeoffData.before.fairnessMetrics.equalOpportunityDifference.toFixed(3)}</td><td className="py-2 pr-4">{tradeoffData.after.fairnessMetrics.equalOpportunityDifference.toFixed(3)}</td></tr>
                      </tbody>
                    </table>
                  </div>

                  <div className="h-72">
                    <ResponsiveContainer width="100%" height="100%">
                      <BarChart data={[
                        { metric: 'RMSE', Before: tradeoffData.before.rmse, After: tradeoffData.after.rmse },
                        { metric: 'MAE', Before: tradeoffData.before.mae, After: tradeoffData.after.mae },
                        { metric: 'R²', Before: tradeoffData.before.r2, After: tradeoffData.after.r2 },
                      ]}>
                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#f0f0f0" />
                        <XAxis dataKey="metric" axisLine={false} tickLine={false} />
                        <YAxis axisLine={false} tickLine={false} />
                        <Tooltip />
                        <Legend />
                        <Bar dataKey="Before" fill="#6366f1" radius={[4, 4, 0, 0]} />
                        <Bar dataKey="After" fill="#f43f5e" radius={[4, 4, 0, 0]} />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>
              )}
            </div>
          )}

          {activeTab === 'predict' && (
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-1 space-y-6">
                <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-sm">
                  <h3 className="font-bold text-gray-800 mb-4 flex items-center gap-2">
                    <BrainCircuit size={18} className="text-indigo-600" />
                    Resume AI Analyzer
                  </h3>
                  <p className="text-xs text-gray-500 mb-4">Paste your resume text below to automatically extract professional details.</p>
                  <textarea 
                    className="w-full h-32 p-3 bg-gray-50 border border-gray-200 rounded-xl text-xs outline-none focus:ring-2 focus:ring-indigo-500 transition-all resize-none"
                    placeholder="Paste resume content here..."
                    value={resumeText}
                    onChange={(e) => setResumeText(e.target.value)}
                  />
                  <button 
                    onClick={handleAnalyzeResume}
                    disabled={isAnalyzing || !resumeText.trim()}
                    className="w-full mt-3 py-2 bg-indigo-50 text-indigo-600 rounded-lg text-xs font-bold hover:bg-indigo-100 transition-colors flex items-center justify-center gap-2 disabled:opacity-50"
                  >
                    {isAnalyzing ? <RefreshCw size={14} className="animate-spin" /> : <UserPlus size={14} />}
                    {isAnalyzing ? 'Analyzing...' : 'Extract from Resume'}
                  </button>
                </div>

                <div className="bg-white p-8 rounded-2xl border border-gray-200 shadow-sm">
                  <h3 className="font-bold text-gray-800 mb-6">Decision Input</h3>
                  <form onSubmit={handlePredict} className="space-y-4">
                  <div className="space-y-1">
                    <label className="text-xs font-bold text-gray-500 uppercase">Model Type</label>
                    <select
                      value={selectedModelType}
                      onChange={(e) => setSelectedModelType(e.target.value as ModelType)}
                      className="w-full p-2 bg-white border border-gray-200 rounded-lg text-sm text-gray-700 outline-none focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value="linear">Linear Regression</option>
                      <option value="random_forest">Random Forest</option>
                    </select>
                  </div>
                  <InputGroup label="Age" type="number" value={formData.age} onChange={(v) => setFormData({...formData, age: v === "" ? NaN : parseInt(v)})} />
                  <div className="space-y-1">
                    <label className="text-xs font-bold text-gray-500 uppercase">Gender</label>
                    <div className="flex gap-2">
                      <button 
                        type="button"
                        onClick={() => setFormData({...formData, gender: 'Male'})}
                        className={cn("flex-1 py-2 rounded-lg text-sm font-medium border transition-all", formData.gender === 'Male' ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-gray-600 border-gray-200")}
                      >
                        Male
                      </button>
                      <button 
                        type="button"
                        onClick={() => setFormData({...formData, gender: 'Female'})}
                        className={cn("flex-1 py-2 rounded-lg text-sm font-medium border transition-all", formData.gender === 'Female' ? "bg-indigo-600 text-white border-indigo-600" : "bg-white text-gray-600 border-gray-200")}
                      >
                        Female
                      </button>
                    </div>
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-bold text-gray-500 uppercase">Job Role</label>
                    <select 
                      value={formData.occupation}
                      onChange={(e) => setFormData({...formData, occupation: e.target.value})}
                      className="w-full p-2 bg-white border border-gray-200 rounded-lg text-sm text-gray-700 outline-none focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value="Software Engineer">Software Engineer</option>
                      <option value="DevOps Engineer">DevOps Engineer</option>
                      <option value="Frontend Developer">Frontend Developer</option>
                      <option value="Backend Developer">Backend Developer</option>
                      <option value="Fullstack Developer">Fullstack Developer</option>
                      <option value="Data Scientist">Data Scientist</option>
                      <option value="QA Engineer">QA Engineer</option>
                      <option value="Cloud Architect">Cloud Architect</option>
                      <option value="Product Manager">Product Manager</option>
                      <option value="Engineering Manager">Engineering Manager</option>
                    </select>
                  </div>
                  <div className="space-y-1">
                    <label className="text-xs font-bold text-gray-500 uppercase">Location</label>
                    <select 
                      value={formData.location}
                      onChange={(e) => setFormData({...formData, location: e.target.value})}
                      className="w-full p-2 bg-white border border-gray-200 rounded-lg text-sm text-gray-700 outline-none focus:ring-2 focus:ring-indigo-500"
                    >
                      <option value="Bengaluru">Bengaluru</option>
                      <option value="Hyderabad">Hyderabad</option>
                      <option value="Pune">Pune</option>
                      <option value="Chennai">Chennai</option>
                      <option value="Mumbai">Mumbai</option>
                      <option value="Noida">Noida</option>
                      <option value="Gurugram">Gurugram</option>
                      <option value="Kolkata">Kolkata</option>
                    </select>
                  </div>
                  <InputGroup label="Work Hours / Week" type="number" value={formData.workHours} onChange={(v) => setFormData({...formData, workHours: v === "" ? NaN : parseInt(v)})} />
                  <InputGroup label="Experience (Years)" type="number" value={formData.experience} onChange={(v) => setFormData({...formData, experience: v === "" ? NaN : parseInt(v)})} />
                  
                  <button 
                    type="submit"
                    className="w-full py-3 bg-indigo-600 text-white rounded-xl font-bold hover:bg-indigo-700 transition-colors mt-6 flex items-center justify-center gap-2"
                  >
                    Run Decision Engine
                    <ArrowRight size={18} />
                  </button>
                </form>

                <div className="mt-8 p-4 bg-indigo-50 rounded-xl border border-indigo-100">
                  <div className="flex items-center justify-between mb-2">
                    <h4 className="text-xs font-bold text-indigo-900 uppercase">What-If Analysis</h4>
                    <button 
                      onClick={() => setShowWhatIf(!showWhatIf)}
                      className={cn("w-10 h-5 rounded-full relative transition-all", showWhatIf ? "bg-indigo-600" : "bg-gray-300")}
                    >
                      <div className={cn("absolute top-1 w-3 h-3 bg-white rounded-full transition-all", showWhatIf ? "left-6" : "left-1")} />
                    </button>
                  </div>
                  <p className="text-[10px] text-indigo-700 leading-relaxed">
                    Automatically compare the decision against a counterfactual scenario where the sensitive attribute (Gender) is swapped.
                  </p>
                </div>
              </div>
            </div>

            <div className="lg:col-span-2 space-y-8">
                {prediction ? (
                  <>
                    <div className={cn("p-8 rounded-3xl border flex items-center justify-between bg-white border-gray-200")}>
                      <div>
                        <h4 className="text-sm font-bold text-gray-500 uppercase tracking-wider mb-2">Predicted Annual Salary</h4>
                        <div className="flex items-center gap-3">
                          <CheckCircle2 className="text-indigo-600" size={32} />
                          <div className="flex flex-col">
                            <span className="text-4xl font-black text-gray-900">
                              ₹{prediction.prediction.toFixed(1)} LPA
                            </span>
                            <span className="text-xs font-bold text-gray-500 uppercase tracking-widest">
                              Interested Domain: {prediction.predictedDomain}
                            </span>
                            <span className={cn(
                              "text-[10px] font-black uppercase px-2 py-0.5 rounded mt-1 w-fit",
                              prediction.prediction > 20 ? "bg-green-100 text-green-700" : "bg-indigo-100 text-indigo-700"
                            )}>
                              {prediction.status}
                            </span>
                          </div>
                        </div>
                        <p className="text-sm text-gray-500 mt-2">
                          Confidence Score: <span className="font-bold text-gray-700">{(prediction.confidence * 100).toFixed(1)}%</span>
                        </p>
                      </div>
                      
                      {prediction.isBiased && (
                        <div className="bg-red-100 text-red-700 px-4 py-2 rounded-xl text-xs font-bold flex items-center gap-2 animate-pulse">
                          <ShieldAlert size={14} />
                          Potential Bias Detected
                        </div>
                      )}
                    </div>

                    <AnimatePresence>
                      {showWhatIf && whatIfResult && (
                        <motion.div 
                          initial={{ opacity: 0, y: 20 }}
                          animate={{ opacity: 1, y: 0 }}
                          exit={{ opacity: 0, y: 20 }}
                          className="bg-indigo-900 text-white p-8 rounded-3xl relative overflow-hidden"
                        >
                          <div className="relative z-10">
                            <h4 className="text-xs font-bold text-indigo-300 uppercase tracking-widest mb-4">Counterfactual Scenario (Gender Swapped)</h4>
                            <div className="flex items-center gap-8">
                              <div className="flex flex-col">
                                <span className="text-[10px] text-indigo-300 uppercase font-bold">Original ({formData.gender})</span>
                                <span className="text-xl font-bold">₹{prediction.prediction.toFixed(1)}L</span>
                              </div>
                              <ArrowRight className="text-indigo-400" />
                              <div className="flex flex-col">
                                <span className="text-[10px] text-indigo-300 uppercase font-bold">What-If ({formData.gender === 'Male' ? 'Female' : 'Male'})</span>
                                <span className={cn("text-xl font-bold", prediction.isBiased ? "text-yellow-400" : "text-white")}>
                                  ₹{whatIfResult.prediction.toFixed(1)}L
                                </span>
                              </div>
                            </div>
                            {prediction.prediction !== whatIfResult.prediction && (
                              <div className="mt-4 p-3 bg-white/10 rounded-xl border border-white/20 flex gap-3 items-center">
                                <AlertCircle size={16} className="text-yellow-400" />
                                <p className="text-xs text-indigo-100">
                                  <strong>Bias Alert:</strong> Changing only the gender attribute flipped the decision. This indicates the model is heavily relying on gender for this specific case.
                                </p>
                              </div>
                            )}
                          </div>
                          <div className="absolute right-0 top-0 bottom-0 w-1/4 bg-white/5 -skew-x-12 translate-x-8" />
                        </motion.div>
                      )}
                    </AnimatePresence>


                  </>
                ) : (
                  <div className="h-full bg-gray-50 border-2 border-dashed border-gray-200 rounded-3xl flex flex-col items-center justify-center p-12 text-center">
                    <BrainCircuit size={48} className="text-gray-300 mb-4" />
                    <h3 className="text-lg font-bold text-gray-400">Ready for Prediction</h3>
                    <p className="text-sm text-gray-400 max-w-xs mt-2">
                      Fill out the form to see how the model makes decisions and detect potential biases in real-time.
                    </p>
                  </div>
                )}
              </div>
            </div>
          )}
        </div>
      </main>
    </div>
  );
}

function NavItem({ active, onClick, icon, label, disabled = false }: { active: boolean, onClick: () => void, icon: React.ReactNode, label: string, disabled?: boolean }) {
  return (
    <button 
      onClick={onClick}
      disabled={disabled}
      className={cn(
        "w-full flex items-center gap-3 px-4 py-3 rounded-xl text-sm font-medium transition-all",
        active ? "bg-indigo-50 text-indigo-600" : "text-gray-500 hover:bg-gray-50 hover:text-gray-700",
        disabled && "opacity-30 cursor-not-allowed"
      )}
    >
      {icon}
      {label}
    </button>
  );
}

function StatCard({ label, value, sub }: { label: string, value: string, sub: string }) {
  return (
    <div className="bg-white p-6 rounded-2xl border border-gray-200 shadow-sm">
      <h4 className="text-xs font-bold text-gray-400 uppercase tracking-widest mb-2">{label}</h4>
      <div className="text-3xl font-black text-gray-900 mb-1">{value}</div>
      <p className="text-xs text-gray-500">{sub}</p>
    </div>
  );
}

function MetricRow({ label, value, isPercentage = true }: { label: string, value: number, isPercentage?: boolean }) {
  const safeValue = isNaN(value) ? 0 : value;
  return (
    <div className="flex justify-between items-center">
      <span className="text-xs text-gray-500 font-medium">{label}</span>
      <span className="text-sm font-bold text-gray-800">
        {isPercentage ? (safeValue * 100).toFixed(1) + '%' : safeValue.toFixed(2)}
      </span>
    </div>
  );
}

function FairnessCard({
  title,
  value,
  threshold,
  desc,
  status,
  tooltip,
}: {
  title: string,
  value: number,
  threshold: string,
  desc: string,
  status: 'pass' | 'fail',
  tooltip?: string,
}) {
  const safeValue = isNaN(value) ? 0 : value;
  return (
    <div className="bg-white p-5 rounded-xl border border-gray-200">
      <div className="flex justify-between items-start mb-2">
        <h4 className="font-bold text-gray-800 text-sm flex items-center gap-1">
          {title}
          {tooltip && (
            <span title={tooltip} className="text-gray-400 cursor-help">
              <Info size={12} />
            </span>
          )}
        </h4>
        <div className={cn("px-2 py-0.5 rounded text-[10px] font-black uppercase", status === 'pass' ? "bg-green-50 text-green-600" : "bg-red-50 text-red-600")}>
          {status === 'pass' ? 'Within Range' : 'Bias Alert'}
        </div>
      </div>
      <div className="flex items-baseline gap-2 mb-2">
        <span className="text-2xl font-black text-gray-900">{safeValue.toFixed(3)}</span>
        <span className="text-[10px] text-gray-400 font-bold">Target: {threshold}</span>
      </div>
      <p className="text-[11px] text-gray-500 leading-relaxed">{desc}</p>
    </div>
  );
}

function FairnessProgress({ label, value, color }: { label: string, value: number, color: string }) {
  const safeValue = isNaN(value) ? 0 : value;
  return (
    <div className="space-y-2">
      <div className="flex justify-between text-xs font-bold text-gray-500 uppercase">
        <span>{label}</span>
        <span style={{ color }}>{(safeValue * 100).toFixed(1)}%</span>
      </div>
      <div className="h-2 bg-gray-100 rounded-full overflow-hidden">
        <div 
          className="h-full rounded-full transition-all duration-1000"
          style={{ width: `${safeValue * 100}%`, backgroundColor: color }}
        />
      </div>
    </div>
  );
}

function InputGroup({ label, type, value, onChange }: { label: string, type: string, value: any, onChange: (v: string) => void }) {
  return (
    <div className="space-y-1">
      <label className="text-xs font-bold text-gray-500 uppercase">{label}</label>
      <input 
        type={type} 
        value={isNaN(value) ? "" : value} 
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-4 py-2 bg-gray-50 border border-gray-200 rounded-lg text-sm focus:ring-2 focus:ring-indigo-500 focus:border-transparent outline-none transition-all"
      />
    </div>
  );
}

function ImpactItem({ title, desc }: { title: string, desc: string }) {
  return (
    <div className="flex gap-3">
      <div className="mt-1">
        <CheckCircle2 size={16} className="text-indigo-600" />
      </div>
      <div>
        <h4 className="font-bold text-sm text-gray-800">{title}</h4>
        <p className="text-xs text-gray-500 leading-relaxed">{desc}</p>
      </div>
    </div>
  );
}
