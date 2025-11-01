# An Ensemble Learning Approach for Early Detection of Hepatitis C

# Overview

This project aims to develop a machine learning model for the early detection of liver disease. It leverages advanced preprocessing techniques and multiple machine learning algorithms to improve diagnostic accuracy and interpretability.
The model integrates KNN Imputer, SMOTE, Optuna hyperparameter tuning, and SHAP explainability to provide robust and interpretable predictions.

 # Objectives

To build a reliable predictive model for early liver disease detection.

To analyze the impact of various blood parameters on disease progression.

To optimize performance through ensemble learning and Optuna tuning.

To provide interpretable model explanations using SHAP values.

# Dataset

Source: Hepatitis C Virus (HCV) dataset from the UCI Machine Learning Repository
Size: 615 records, 13 attributes

# Attributes
Feature	Description
Age	Patient’s age
Sex	Gender (M/F)
ALB	Albumin level
ALP	Alkaline Phosphatase
ALT	Alanine Transaminase
AST	Aspartate Aminotransferase
BIL	Bilirubin
CHE	Cholinesterase
CHOL	Cholesterol
CREA	Creatinine
GGT	Gamma-glutamyl Transferase
PROT	Total Protein
Category	Target variable (Blood Donor, Hepatitis, Fibrosis, Cirrhosis)
# Methodology
1. Data Preprocessing

Handled missing values using KNN Imputer

Standardized features with StandardScaler

Balanced dataset using SMOTE

2. Model Training

Algorithms used:

Logistic Regression

K-Nearest Neighbors (KNN)

Support Vector Machine (SVM)

Random Forest

Gaussian Naive Bayes

Ensemble Voting Classifier

3. Optimization

Optuna used for hyperparameter tuning

Metrics evaluated: Accuracy, Precision, Recall, F1-score

4. Model Interpretability

SHAP used to interpret feature importance and visualize the impact of predictors.

# Results
Model	Accuracy	Remarks
Logistic Regression	87%	Baseline performance
KNN	89%	Good for small datasets
SVM	90%	High precision on nonlinear boundaries
Random Forest	93%	Best performing model
Voting Classifier	92%	Stable and consistent results

Feature importance (via SHAP):

Bilirubin, ALP, ALT, and Age were among the most significant indicators.

# Tools and Libraries

Python

NumPy, Pandas — Data manipulation

Matplotlib, Seaborn — Visualization

Scikit-learn — ML model implementation

Imbalanced-learn (SMOTE) — Class balancing

Optuna — Hyperparameter optimization

SHAP — Model interpretability

# Evaluation Metrics

Accuracy

Precision

Recall

F1 Score

These metrics were used to evaluate the model’s predictive power and generalization capability.

# Future Enhancements

Integrate deep learning models (e.g, Neural Networks).

Deploy the model as a web or mobile diagnostic tool for medical professionals.

Include real-time data from hospitals or health APIs.

Implement a user-friendly dashboard for visualization.
