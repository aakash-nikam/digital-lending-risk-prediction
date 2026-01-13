Dublin Loan Risk Early-Warning System

AI-Driven Behavioural Credit Risk Prediction

Author: Akash Nikam (DBS – MSc Data Analytics)
Tools: Python (XGBoost, SHAP, LIME, scikit-learn, Streamlit)

Project Overview

This project develops an AI-driven early-warning system to predict loan default risk in Dublin’s digital lending environment.

Unlike traditional credit risk models that rely mainly on static financial variables, this system combines:

Financial information

Behavioural indicators

Sentiment and market signals

The objective is to identify risky borrowers at an early stage while keeping the model transparent and explainable for real-world lending decisions.

The system outputs a probability of default, a risk classification, and an explanation of why a borrower is flagged as risky.

Tech and Libraries

Python 3.x

pandas, numpy

scikit-learn

XGBoost

SHAP

LIME

matplotlib, seaborn

Streamlit

Install requirements with:

pip install -r requirements.txt

Dataset

The project integrates multiple Dublin-specific datasets representing different dimensions of borrower behaviour:

Loan and repayment data (loan amount, missed payments, loan term)

Behavioural indicators (digital lending app usage, BNPL search behaviour)

Sentiment indicators derived from consumer reviews

Macroeconomic and survey-based financial indicators

All datasets are cleaned, aligned by time and region, and merged into a unified modelling dataset.

Feature Engineering

Key engineered features include:

Missed payment ratio

Loan per month

Debt-to-income ratio

Consumer sentiment index

Financial pressure indicators

Time-based features (month, year-month index)

Feature engineering plays a critical role in improving early risk detection.

Model
XGBoost Classifier

Selected for handling non-linear relationships and mixed feature types

Trained on combined financial, behavioural, and sentiment features

Optimised using probability-based risk prediction

Evaluation focuses on Precision-Recall curves due to class imbalance in loan default data.

Explainability

To ensure transparency and regulatory relevance, two explainability techniques are applied:

SHAP

Provides global feature importance

Explains how features increase or decrease default risk

Used for both population-level and individual borrower analysis

LIME

Provides local, rule-based explanations

Shows which conditions push a borrower into the risky class

Useful for manual review and underwriting decisions

Results

Models using behavioural and sentiment features outperform baseline financial-only models

Missed payments and loan amount are the strongest risk drivers

Behavioural indicators improve early detection of risky borrowers

Explainability results are consistent with financial intuition

Precision-Recall analysis shows improved recall for high-risk borrowers when behavioural features are included.

Streamlit Application

A Streamlit web application demonstrates the system in practice.

The app allows users to:

Enter minimal borrower information

Automatically derive behavioural indicators

View probability of default

Adjust the decision threshold

Receive a clear risk classification and recommendation

This simulates a real-world early-warning decision support tool for lenders.

How to Run

Clone the repository

Install dependencies

pip install -r requirements.txt


Run the notebook

jupyter notebook main_final.ipynb


Launch the Streamlit app

streamlit run app.py

Project Structure
├── main_final.ipynb        # Data processing, modelling, explainability
├── app.py                 # Streamlit application
├── requirements.txt       # Python dependencies
├── figures/               # Plots and visualisations
├── README.md              # Project documentation

Future Work

Incorporate real-time transaction data

Extend to multi-city or national-level lending datasets

Explore fairness and bias analysis

Integrate automated monitoring for concept drift

Academic Artefacts

Full dissertation report (DBS MSc Data Analytics)

Presentation slides and supporting code

Explainability visualisations (SHAP & LIME)

Contact

Akash Nikam
MSc Data Analytics
Dublin Business School
Email: aakashn3118@gmail.com
