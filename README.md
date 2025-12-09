# 🎓 Student Skill Gap Analyzer & Career Recommendation System

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-In%20Development-yellow.svg)]()

## 📋 Project Overview

The **Student Skill Gap Analyzer & Career Recommendation System** is an AI/ML-powered application designed to help students and job seekers identify skill gaps and receive personalized career recommendations based on their current skill set. This system analyzes user skills, compares them with industry requirements, and provides actionable insights for career development.

## 🎯 Problem Statement

In today's rapidly evolving job market, students and professionals often struggle to:
- Identify which skills they need to acquire for their desired career path
- Understand the gap between their current skills and industry requirements
- Get personalized recommendations for career transitions
- Make informed decisions about skill development and career planning

This project aims to solve these challenges by leveraging machine learning to:
1. **Analyze** student/user skill profiles
2. **Identify** skill gaps for specific job roles
3. **Recommend** suitable career paths based on existing skills
4. **Suggest** skills to learn for career advancement

## 📊 Dataset Description

### Dataset: `skills_dataset.csv`

The dataset contains skill-to-job role mappings with the following structure:

| Column | Description |
|--------|-------------|
| `skills` | Comma-separated list of technical skills |
| `job_role` | Target job role/position |

**Sample Data:**
```
skills,job_role
python numpy pandas,data analyst
python deep learning tensorflow,ai engineer
java spring sql,backend developer
html css javascript,frontend developer
```

**Dataset Statistics:**
- **Total Records:** 40+ skill-job mappings
- **Job Roles Covered:** 
  - Data Analyst
  - ML Engineer
  - AI Engineer
  - Backend Developer
  - Frontend Developer
  - DevOps Engineer
  - Data Scientist
  - QA Engineer
  - Mobile Developer
  - Data Engineer
  - Database Administrator

**Key Features:**
- Diverse skill combinations across multiple tech domains
- Real-world job role mappings
- Covers both traditional and emerging tech roles
- Suitable for multi-class classification

## 🛠️ Tools & Technologies

### Programming Languages
- **Python 3.8+** - Core programming language

### Libraries & Frameworks
- **Data Manipulation:** pandas, numpy
- **Machine Learning:** scikit-learn
- **Deep Learning:** TensorFlow/PyTorch (for advanced models)
- **Natural Language Processing:** NLTK, spaCy
- **Data Visualization:** matplotlib, seaborn, plotly
- **Model Deployment:** Flask/FastAPI (planned)

### Development Tools
- **Version Control:** Git & GitHub
- **IDE:** VS Code / Jupyter Notebook
- **Environment:** Virtual Environment (venv)

## 📁 Project Structure

```
project-root/
│
├── dataset/                    # Dataset files
│   └── skills_dataset.csv     # Main skills-job mapping dataset
│
├── notebooks/                  # Jupyter notebooks
│   └── 01_eda.ipynb           # Exploratory Data Analysis
│
├── src/                        # Source code
│   └── preprocess.py          # Data preprocessing scripts (upcoming)
│
├── models/                     # Saved ML models
│
├── results/                    # Model results and visualizations
│
└── README.md                   # Project documentation
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.8 or higher
pip package manager
Git
```

### Installation

1. **Clone the repository**
```bash
git clone https://github.com/Sudharsanv06/Student-Skill-Gap-Analyzer-Career-Recommendation-System-AI-ML-.git
cd Student-Skill-Gap-Analyzer-Career-Recommendation-System-AI-ML-
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
```

3. **Install dependencies** (upcoming)
```bash
pip install -r requirements.txt
```

## 📈 Project Phases

### ✅ Day 1: Project Setup + Dataset + EDA
- [x] Project structure created
- [x] Dataset prepared (40+ skill-job mappings)
- [x] Initial EDA notebook created
- [x] Git repository initialized

### 🔄 Day 2: Data Preprocessing + Feature Engineering (Planned)
- [ ] Text preprocessing (tokenization, cleaning)
- [ ] Feature extraction (TF-IDF, word embeddings)
- [ ] Label encoding for job roles
- [ ] Train-test split

### 🔄 Day 3: Model Building (Planned)
- [ ] Baseline model (Naive Bayes, Logistic Regression)
- [ ] Advanced models (Random Forest, XGBoost)
- [ ] Deep learning models (LSTM, transformers)
- [ ] Model evaluation and comparison

### 🔄 Day 4: Deployment + Documentation (Planned)
- [ ] Build recommendation engine
- [ ] Skill gap analysis module
- [ ] API development (Flask/FastAPI)
- [ ] Final documentation and deployment

## 🎯 Key Features (Planned)

1. **Career Path Recommendation**
   - Input your current skills
   - Get top job role predictions with confidence scores

2. **Skill Gap Analysis**
   - Identify missing skills for target job role
   - Prioritize skills based on importance

3. **Career Transition Suggestions**
   - Analyze feasibility of career transitions
   - Suggest learning paths for career change

4. **Skill Demand Insights**
   - Visualize trending skills in different domains
   - Industry-specific skill requirements

## 📊 Expected Outputs

- Trained ML models for job role classification
- Skill gap analysis reports
- Interactive visualizations
- RESTful API for predictions
- Web interface (optional)

## 🤝 Contributing

This is an academic project. Suggestions and feedback are welcome!

## 📧 Contact

**Developer:** Sudharsan V  
**GitHub:** [@Sudharsanv06](https://github.com/Sudharsanv06)  
**Project Link:** [Student-Skill-Gap-Analyzer-Career-Recommendation-System-AI-ML-](https://github.com/Sudharsanv06/Student-Skill-Gap-Analyzer-Career-Recommendation-System-AI-ML-)

## 📝 License

This project is open source and available for educational purposes.

---

**Note:** This project is currently in active development. Features and structure may evolve over time.

*Last Updated: December 2025*
