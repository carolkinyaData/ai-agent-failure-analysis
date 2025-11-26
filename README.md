# 🧠 AI Agent Failure Analysis
### By **Caroline Kinya**

This project analyzes failure patterns in AI agent performance across finance-sector tasks. It applies statistical methods, anomaly detection, and feature importance analysis to uncover root causes of agent failures.

---

## 📌 Project Objectives
- Identify failure clusters in task types and file types  
- Detect anomalies in agent performance using ML techniques  
- Uncover patterns across agent behavior  
- Determine root causes of recurrent failures  
- Generate actionable recommendations for AI evaluation teams  

---

## 📊 Methods Used

### 🔹 1. **Exploratory Data Analysis (EDA)**
- Distribution analysis  
- Correlation checks  
- Failure frequencies by task/subcategory  
- Missing value analysis  

### 🔹 2. **Anomaly Detection**
- **Isolation Forest**  
- **One-Class SVM**  
- **Z-score outlier detection**

Outputs include:
- Anomaly heatmaps  
- Outlier tasks  
- Agent-performance anomaly indexes  

### 🔹 3. **Feature Importance Analysis**
Using machine learning models:
- Random Forest  
- Permutation Importance  
- SHAP values  

---

## 🧪 Project Structure

```
ai-agent-failure-analysis/
│
├── data/
│   ├── sample_agent_performance.csv
│   └── README.md
│
├── notebooks/
│   ├── 01_data_exploration.ipynb
│   ├── 02_anomaly_detection.ipynb
│   └── 03_feature_importance.ipynb
│
├── scripts/
│   ├── preprocess.py
│   ├── anomaly_detection.py
│   └── feature_importance.py
│
├── visuals/
│   ├── anomaly_plot.png
│   └── feature_importance_chart.png
│
└── README.md
```

---

## 🛠️ Tech Stack
- **Python** (pandas, numpy, scikit-learn)
- **Jupyter Notebook**
- **Matplotlib / Seaborn**
- **Tableau** (for visualization planning)
- **Git & GitHub**

---

## 📁 Folders Overview

### `/data`  
Contains sample dataset + explanation.

### `/notebooks`  
Jupyter notebooks for exploration, anomaly detection, and feature importance.

### `/scripts`  
Python modules to run outside notebooks.

### `/visuals`  
Charts showing anomalies & feature importance results.

---

## 📬 Contact
**LinkedIn:** www.linkedin.com/in/kinya-carol-741b8a397  
Open for collaboration and feedback!

