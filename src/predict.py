"""
Career Prediction and Skill Gap Analysis Module

This module provides functionality to:
1. Load trained ML models
2. Predict career role based on student skills
3. Identify skill gaps by comparing student skills with required skills
"""

import joblib
import pandas as pd
import numpy as np
from preprocess import preprocess_text


class CareerPredictor:
    """
    A class to predict career roles and identify skill gaps
    """
    
    def __init__(self, model_path='../models/career_prediction_model.pkl',
                 vectorizer_path='../models/tfidf_vectorizer.pkl',
                 encoder_path='../models/label_encoder.pkl',
                 dataset_path='../dataset/skills_dataset.csv'):
        """
        Initialize the predictor with trained models and dataset
        
        Args:
            model_path: Path to the trained model
            vectorizer_path: Path to the TF-IDF vectorizer
            encoder_path: Path to the label encoder
            dataset_path: Path to the skills dataset
        """
        # Load the trained model and preprocessing objects
        self.model = joblib.load(model_path)
        self.vectorizer = joblib.load(vectorizer_path)
        self.label_encoder = joblib.load(encoder_path)
        
        # Load the dataset to build required skills knowledge base
        self.df = pd.read_csv(dataset_path)
        self._build_skill_knowledge_base()
    
    def _build_skill_knowledge_base(self):
        """
        Build a knowledge base of required skills for each job role
        """
        self.skill_knowledge = {}
        
        for job_role in self.df['job_role'].unique():
            # Get all skills entries for this job role
            role_skills = self.df[self.df['job_role'] == job_role]['skills'].tolist()
            
            # Extract unique skills (splitting by space and cleaning)
            all_skills = set()
            for skill_entry in role_skills:
                skills = skill_entry.lower().split()
                all_skills.update(skills)
            
            self.skill_knowledge[job_role] = sorted(list(all_skills))
    
    def predict_career(self, student_skills):
        """
        Predict the most suitable career role for given skills
        
        Args:
            student_skills: String of student's skills (space or comma separated)
        
        Returns:
            predicted_role: The predicted job role
            confidence: Prediction probability/confidence
        """
        # Preprocess the input skills
        cleaned_skills = preprocess_text(student_skills)
        
        # Transform using TF-IDF vectorizer
        X = self.vectorizer.transform([cleaned_skills])
        
        # Predict
        prediction = self.model.predict(X)[0]
        predicted_role = self.label_encoder.inverse_transform([prediction])[0]
        
        # Get prediction probability (if available)
        if hasattr(self.model, 'predict_proba'):
            proba = self.model.predict_proba(X)[0]
            confidence = np.max(proba)
        else:
            confidence = None
        
        return predicted_role, confidence
    
    def identify_skill_gaps(self, student_skills, predicted_role):
        """
        Identify missing skills for the predicted role
        
        Args:
            student_skills: String of student's skills
            predicted_role: The predicted job role
        
        Returns:
            missing_skills: List of skills the student needs to acquire
            matched_skills: List of skills the student already has
        """
        # Get student's skills as a set
        student_skill_list = set(student_skills.lower().replace(',', ' ').split())
        
        # Get required skills for the predicted role
        required_skills = set(self.skill_knowledge.get(predicted_role, []))
        
        # Find matches and gaps
        matched_skills = sorted(list(student_skill_list.intersection(required_skills)))
        missing_skills = sorted(list(required_skills - student_skill_list))
        
        return missing_skills, matched_skills
    
    def get_career_recommendation(self, student_skills):
        """
        Get complete career recommendation including prediction and skill gaps
        
        Args:
            student_skills: String of student's skills
        
        Returns:
            Dictionary containing prediction, confidence, matched skills, and gaps
        """
        # Predict career
        predicted_role, confidence = self.predict_career(student_skills)
        
        # Identify skill gaps
        missing_skills, matched_skills = self.identify_skill_gaps(student_skills, predicted_role)
        
        return {
            'predicted_role': predicted_role,
            'confidence': confidence,
            'matched_skills': matched_skills,
            'missing_skills': missing_skills,
            'total_required_skills': len(self.skill_knowledge.get(predicted_role, [])),
            'skills_coverage': len(matched_skills) / len(self.skill_knowledge.get(predicted_role, [])) * 100 
                               if self.skill_knowledge.get(predicted_role, []) else 0
        }
    
    def display_recommendation(self, student_skills):
        """
        Display formatted career recommendation
        
        Args:
            student_skills: String of student's skills
        """
        result = self.get_career_recommendation(student_skills)
        
        print("="*70)
        print("CAREER RECOMMENDATION REPORT")
        print("="*70)
        print(f"\n📝 Your Skills: {student_skills}")
        print(f"\n🎯 Predicted Career: {result['predicted_role'].upper()}")
        
        if result['confidence']:
            print(f"📊 Confidence: {result['confidence']*100:.2f}%")
        
        print(f"\n✅ Skills Coverage: {result['skills_coverage']:.1f}% "
              f"({len(result['matched_skills'])}/{result['total_required_skills']} skills)")
        
        if result['matched_skills']:
            print(f"\n✓ Matched Skills:")
            for skill in result['matched_skills']:
                print(f"  • {skill}")
        
        if result['missing_skills']:
            print(f"\n⚠️  Missing Skills (Skill Gap):")
            for skill in result['missing_skills']:
                print(f"  • {skill}")
        else:
            print("\n🎉 Congratulations! You have all required skills!")
        
        print("\n" + "="*70)
        
        return result


def main():
    """
    Demo function to test the prediction system
    """
    # Initialize predictor
    print("Loading trained models...")
    predictor = CareerPredictor()
    print("Models loaded successfully!\n")
    
    # Test cases
    test_cases = [
        "python sklearn pandas",
        "python deep learning tensorflow",
        "java spring sql",
        "html css javascript react",
        "python nlp transformers"
    ]
    
    print("Running demo predictions...\n")
    
    for i, skills in enumerate(test_cases, 1):
        print(f"\n{'='*70}")
        print(f"TEST CASE {i}")
        result = predictor.display_recommendation(skills)
        print()


if __name__ == "__main__":
    main()
