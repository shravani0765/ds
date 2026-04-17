# 🚀 AI Fairness in Salary Prediction (Indian IT Market)

A **fairness-aware machine learning web application** that predicts salaries in the Indian IT job market while analyzing and mitigating algorithmic bias.

This project demonstrates **model comparison, fairness evaluation, explainable AI (XAI), and interactive bias analysis** — all running **client-side in the browser**.

---

## 📌 Overview

This application allows users to:

* Train ML models on salary datasets
* Compare multiple models (Linear Regression vs Random Forest)
* Evaluate fairness using multiple bias metrics
* Analyze feature importance (Explainable AI)
* Perform counterfactual analysis (e.g., gender swap)
* Explore accuracy vs fairness trade-offs
* Adjust fairness thresholds interactively
* Export results as structured reports

---

## ✨ Key Features

### 🤖 Machine Learning

* Linear Regression (baseline)
* Random Forest (ensemble model)
* Multi-model training & comparison

### ⚖️ Fairness Analysis

* Disparate Impact (DI)
* Parity Difference
* **Equal Opportunity (TPR-based)**
* Bias detection indicators
* Threshold-based fairness control (percentile slider)

### 📊 Explainable AI (XAI)

* Feature importance using:

  * Random Forest importance
  * Permutation importance (with averaging)
* Stability estimation using standard deviation

### 🔁 Counterfactual Analysis

* “What-if” prediction (e.g., gender swap)
* Detects bias at individual level

### 📉 Trade-off Analysis

* Compare:

  * Accuracy (RMSE, MAE, R²)
  * Fairness (before vs after mitigation)

### 🧠 Executive Dashboard

* Best model selection (accuracy + fairness)
* KPI cards
* Top contributing features
* Bias alert system

### 📤 Export Reports

* Download JSON containing:

  * Model metrics
  * Fairness metrics
  * Trade-off comparison
  * Feature importance
  * Threshold configuration

---

## 🛠️ Tech Stack

### Frontend

* React (Vite)
* TypeScript
* Tailwind CSS

### Visualization

* Recharts

### Machine Learning (Client-side)

* ml-random-forest
* ml-matrix

### Data Handling

* PapaParse

### AI Integration

* Google Gemini API (resume parsing)

---

## 🧩 Project Structure

```
/src
  ├── App.tsx              # Main UI and workflows
  ├── main.tsx             # React entry point
  ├── mlService.ts         # ML logic (training, fairness, explainability)
  ├── resumeService.ts     # Gemini API integration
  ├── dataset.ts           # Dataset loader / generator
  ├── types.ts             # TypeScript interfaces

/public
  ├── Dataset.csv
  ├── indian_it_job_market.csv
```

---

## ⚙️ How It Works

1. Load dataset (CSV or synthetic)
2. Preprocess data
3. Train models (Linear + Random Forest)
4. Evaluate:

   * Accuracy metrics
   * Fairness metrics
5. Apply bias mitigation
6. Visualize:

   * Model comparison
   * Feature importance
   * Trade-offs
7. Perform predictions + counterfactual analysis

---

## 🧪 How to Run Locally

### 🔹 Prerequisites

* Node.js (v18+ recommended)
* npm or yarn

---

### 🔹 1. Clone the repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

---

### 🔹 2. Install dependencies

```bash
npm install
```

---

### 🔹 3. Set up environment variables

Create a `.env` file in root:

```env
VITE_GEMINI_API_KEY=your_api_key_here
```

> ⚠️ Required only if using resume parsing feature

---

### 🔹 4. Run the development server

```bash
npm run dev
```

Open in browser:

```
http://localhost:5173
```

---

### 🔹 5. Build for production

```bash
npm run build
```

Preview build:

```bash
npm run preview
```

---

## 📊 Fairness Metrics Explained

### 📌 Disparate Impact

Measures ratio of favorable outcomes between groups

### 📌 Equal Opportunity

Ensures equal **True Positive Rate (TPR)** across groups

### 📌 Parity Difference

Difference in prediction rates between groups

---

## ⚖️ Accuracy vs Fairness Trade-off

Improving fairness often slightly reduces model accuracy due to constraints on sensitive attributes.
This project visualizes that trade-off explicitly.

---

## 📸 Screenshots

> Add these after deployment:

* Overview Dashboard
* Feature Importance Chart
* Trade-off Comparison
* Prediction UI

---

## 📈 Key Insights

* Random Forest achieves higher accuracy but may introduce bias
* Bias mitigation improves fairness but can reduce accuracy
* Feature importance reveals key drivers like experience and education
* Threshold selection significantly affects fairness outcomes

---

## 🚀 Future Improvements

* Backend integration (FastAPI)
* SHAP-based explainability
* Model persistence
* User authentication
* Real dataset integration

---

## 📜 License

MIT License

---

## 🙌 Acknowledgements

* Open-source ML libraries
* Google Gemini API
* Fairness in AI research community

---

## 📬 Contact


