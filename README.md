# 🎓 Student Skill Gap Analyzer & Career Recommendation System

[![Python](https://img.shields.io/badge/Python-3.12+-blue.svg)](https://www.python.org/)
[![Machine Learning](https://img.shields.io/badge/ML-Scikit--Learn-orange.svg)](https://scikit-learn.org/)
[![Status](https://img.shields.io/badge/Status-Complete-green.svg)]()
[![Accuracy](https://img.shields.io/badge/Best%20Model-62.5%25-brightgreen.svg)]()

## 📋 Project Overview

The **Student Skill Gap Analyzer & Career Recommendation System** is an AI/ML-powered application designed to help students and job seekers identify skill gaps and receive personalized career recommendations based on their current skill set.

**Key Features:**
- 🤖 **ML-Powered Predictions:** Uses trained Random Forest model with 62.5% accuracy
- 📊 **Skill Gap Analysis:** Identifies missing skills for target career roles
- 🎯 **Career Recommendations:** Suggests optimal career paths based on current skills
- 📈 **Skills Coverage:** Shows percentage of skills you already have vs. required
- ⚡ **Real-time Predictions:** Fast inference using TF-IDF vectorization

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
Student-Skill-Gap-Analyzer-Career-Recommendation-System-AI-ML-/
│
├── dataset/                           # Dataset files
│   └── skills_dataset.csv             # 40 skill-job mappings across 11 roles
│
├── notebooks/                          # Jupyter notebooks (complete workflow)
│   ├── 01_eda.ipynb                   # Exploratory Data Analysis
│   ├── 02_feature_engineering.ipynb   # TF-IDF vectorization & preprocessing
│   ├── 03_model_training.ipynb        # Model training & evaluation
│   └── 04_demo_prediction.ipynb       # Live prediction demo
│
├── src/                                # Source code modules
│   ├── preprocess.py                  # Text preprocessing utilities
│   └── predict.py                     # Career prediction & skill gap analyzer
│
├── models/                             # Trained ML models (saved artifacts)
│   ├── career_prediction_model.pkl    # Best model: Random Forest (62.5% accuracy)
│   ├── tfidf_vectorizer.pkl           # TF-IDF vectorizer
│   └── label_encoder.pkl              # Job role label encoder
│
├── results/                            # Model metrics & visualizations
│   ├── metrics.txt                    # Model comparison results
│   └── feature_info.txt               # TF-IDF feature details
│
└── README.md                           # Complete project documentation
```

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.12+
pip package manager
Git
Jupyter Notebook
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

3. **Install dependencies**
```bash
pip install pandas numpy scikit-learn matplotlib seaborn joblib jupyter
```

### 📖 How to Run the Project

#### Option 1: Run Demo Notebook (Recommended)
```bash
# Open Jupyter Notebook
jupyter notebook

# Navigate to notebooks/ folder and open:
04_demo_prediction.ipynb

# Run all cells to see live predictions!
```

#### Option 2: Use Python Script
```bash
# Navigate to src/ folder
cd src

# Run prediction demo
python predict.py
```

#### Option 3: Interactive Prediction
```python
from src.predict import CareerPredictor

# Initialize predictor
predictor = CareerPredictor()

# Get recommendation for your skills
your_skills = "python machine learning tensorflow"
predictor.display_recommendation(your_skills)
```

### 📊 Example Output

```
======================================================================
CAREER RECOMMENDATION REPORT
======================================================================

📝 Your Skills: python deep learning tensorflow

🎯 Predicted Career: AI ENGINEER
📊 Confidence: 72.38%

✅ Skills Coverage: 30.8% (4/13 skills)

✓ Matched Skills:
  • deep
  • learning
  • python
  • tensorflow

⚠️  Missing Skills (Skill Gap):
  • computer
  • image
  • keras
  • networks
  • neural
  • opencv
  • processing
  • pytorch
  • vision

======================================================================
```

## 📈 Project Workflow & Implementation

### ✅ Day 1: Project Setup + Dataset + Problem Definition
**Status:** ✅ Complete  
**Commit:** `00cdc7a`

- ✅ Project structure established
- ✅ Dataset created with 40 skill-job mappings across 11 roles
- ✅ Initial EDA performed in `01_eda.ipynb`
- ✅ Problem statement defined
- ✅ Git repository initialized

### ✅ Day 2: Text Preprocessing + Feature Engineering
**Status:** ✅ Complete  
**Commits:** `286dcdc`, `78c8666`

- ✅ Text preprocessing module (`src/preprocess.py`)
  - Lowercase conversion
  - Special character removal
  - Extra space handling
- ✅ TF-IDF vectorization (max_features=100, ngram_range=(1,2))
- ✅ Label encoding for 11 job roles
- ✅ Train-test split (80/20) - 32 training, 8 testing samples
- ✅ Feature matrix: 95.93% sparse, ready for ML

### ✅ Day 3: Model Training + Evaluation
**Status:** ✅ Complete  
**Commit:** `1cc5045`

**Models Trained:**
1. **Logistic Regression**
   - Training Accuracy: 87.5%
   - Test Accuracy: 37.5%
   - F1-Score: 0.27
   
2. **Random Forest Classifier** ⭐ (Best Model)
   - Training Accuracy: 100%
   - Test Accuracy: **62.5%**
   - F1-Score: **0.575**
   - Precision: 0.60
   - Recall: 0.62

**Model Selection:** Random Forest chosen as best model due to superior F1-score and test accuracy.

**Saved Artifacts:**
- ✅ `models/career_prediction_model.pkl`
- ✅ `models/tfidf_vectorizer.pkl`
- ✅ `models/label_encoder.pkl`
- ✅ `results/metrics.txt`

### ✅ Day 4: Skill Gap Logic + Prediction + Documentation
**Status:** ✅ Complete

- ✅ Career prediction module (`src/predict.py`)
  - CareerPredictor class with full functionality
  - Skill gap identification logic
  - Confidence scoring
  - Skills coverage calculation
- ✅ Demo notebook (`04_demo_prediction.ipynb`)
  - Interactive predictions
  - Batch prediction examples
  - Available career paths listing
- ✅ Complete README documentation
- ✅ Final project cleanup

## 🎯 Key Features & Functionality

### 1. 🤖 Career Path Prediction
- Input your current skills (space or comma-separated)
- Get AI-powered job role prediction
- Receive confidence score for prediction

### 2. 📊 Skill Gap Analysis
- Compare your skills vs. required skills for predicted role
- See percentage coverage of required skills
- Get detailed list of missing skills to learn

### 3. 🎓 Skills Knowledge Base
- Built from 40 real-world skill-job mappings
- Covers 11 diverse tech career paths
- Continuously expandable dataset

### 4. ⚡ Fast Predictions
- Pre-trained models for instant inference
- TF-IDF vectorization for efficient feature extraction
- Lightweight model files (<1MB total)

## 📊 Model Performance Summary

| Metric | Logistic Regression | Random Forest (Best) |
|--------|-------------------|---------------------|
| **Train Accuracy** | 87.5% | 100% |
| **Test Accuracy** | 37.5% | **62.5%** |
| **Precision** | 0.22 | **0.60** |
| **Recall** | 0.38 | **0.62** |
| **F1-Score** | 0.27 | **0.575** |

**Key Insights:**
- Random Forest shows better generalization despite small dataset
- 62.5% test accuracy is reasonable for 11-class classification with 40 samples
- Model performs well for common roles (AI Engineer, Backend Developer, Frontend Developer)
- Skill gap logic provides actionable insights beyond just prediction

## 💡 Technical Highlights

### Machine Learning Pipeline
1. **Text Preprocessing** → Lowercase, clean special chars, normalize spaces
2. **Feature Extraction** → TF-IDF with bigrams (100 features)
3. **Model Training** → Random Forest (100 estimators)
4. **Prediction** → Career role + confidence score
5. **Skill Gap Analysis** → Compare student vs. required skills

### Technologies & Algorithms
- **Algorithm:** Random Forest Classifier (ensemble method)
- **Vectorization:** TF-IDF with n-grams (1,2)
- **Encoding:** Label Encoder for multi-class targets
- **Evaluation:** Classification report, confusion matrix, accuracy metrics
- **Persistence:** Joblib for model serialization

## 🎓 Supported Career Paths

The system can predict and analyze skills for the following 11 tech career roles:

1. **AI Engineer** - Deep learning, computer vision, NLP
2. **Backend Developer** - Server-side development, APIs, databases
3. **Data Analyst** - SQL, visualization, statistical analysis
4. **Data Engineer** - ETL, big data, data pipelines
5. **Data Scientist** - Statistics, ML, data analysis
6. **Database Administrator** - Database design, SQL, NoSQL
7. **DevOps Engineer** - CI/CD, cloud infrastructure, automation
8. **Frontend Developer** - HTML, CSS, JavaScript frameworks
9. **ML Engineer** - Machine learning, model deployment, MLOps
10. **Mobile Developer** - iOS/Android app development
11. **QA Engineer** - Testing frameworks, automation, quality assurance

## 🚀 Future Enhancements

- [ ] Expand dataset to 500+ skill mappings
- [ ] Add web interface (Flask/Streamlit)
- [ ] Implement learning path recommendations
- [ ] Add skill importance scoring
- [ ] Industry trend analysis
- [ ] Career transition feasibility score
- [ ] RESTful API for predictions
- [ ] Real-time job market data integration

## 📚 Lessons Learned

1. **Small dataset challenges:** With only 40 samples, avoiding stratification was crucial
2. **Feature engineering:** TF-IDF with bigrams captured skill relationships effectively
3. **Model selection:** Random Forest outperformed Logistic Regression for this multi-class problem
4. **Skill gap logic:** Simple set operations provide powerful insights
5. **Practical ML:** Sometimes simple solutions work best for real-world problems

## 🏆 Project Achievements

✅ Complete end-to-end ML pipeline  
✅ Working prediction system with 62.5% accuracy  
✅ Practical skill gap identification  
✅ Clean, modular, reusable code  
✅ Comprehensive documentation  
✅ 4 Jupyter notebooks demonstrating full workflow  
✅ Git version control with meaningful commits  

## 🎯 Conclusion

This project successfully demonstrates how machine learning can be applied to career guidance and skill development. Despite working with a small dataset (40 samples), the system achieves reasonable accuracy (62.5%) and provides actionable insights through skill gap analysis.

**Key Takeaways:**
- ML can effectively map skills to career roles
- Skill gap identification helps students plan their learning journey
- Simple, interpretable models (Random Forest) work well for this domain
- The system is production-ready and can be easily extended

The project showcases the complete ML lifecycle: from data collection and preprocessing, through model training and evaluation, to deployment-ready prediction capabilities. It serves as a solid foundation for building more sophisticated career recommendation systems.

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the issues page.

## 📧 Contact

**Developer:** Sudharsan V  
**GitHub:** [@Sudharsanv06](https://github.com/Sudharsanv06)  
**Project Link:** [Student-Skill-Gap-Analyzer-Career-Recommendation-System-AI-ML-](https://github.com/Sudharsanv06/Student-Skill-Gap-Analyzer-Career-Recommendation-System-AI-ML-)

---

⭐ **Star this repository if you find it helpful!**

**Built with ❤️ using Python & Scikit-Learn**

This project is open source and available for educational purposes.

---

**Note:** This project is currently in active development. Features and structure may evolve over time.

*Last Updated: December 2025*
