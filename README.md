# Home-Credit-Group-analysis

## Introduction
This is a data analysis project for Home credit group dataset. The purpose of this analysis is to develop a risk evaluation tool. The main objectives are:

1. Data exploration and understanding
2. Model training and evaluation
3. Model deployment

My goal with this project is to practice conducting exploratory and statistical data analysis, data handling with python, and predicting with machine learning models. The project is part of Turing collage Data science course curriculum. 

## Project Structure
The analysis is divided into four parts:
- First part concentrates on preprocessing and aggregating data
	- The objectives in this part are 
        1. conduct data preprocessing and checking data quality. 
        2. conduct data aggregation for historical data
- Second part concentrates on exploratory data analysis
    - The objectives in this part are 
        1. conduct exploratory data analysis and gain understanding of the data. 
        2. identify interesting target features
- Third part concentrates on model building
    - The objectives in this part are 
        1. select best model for production for the identified problems
- Fourth part concentrates on model deployment
    - The objective in this part is to deploy the model into production. 
	
For parts 1 to 3 there are separate notebooks showing the process, but for part four there are only source codes provided for deployed model and streamlit app. The models are deployed using custom docker containers. The source codes for these containers are provided in CustomContainer folder. The source code for the application is provided in DefaultClfApp folder without the GCP service-key json file.


The Home Credit group dataset is preprocessed and aggregated using CustomTransformer class and its child classes for each table. The source code for these classes is in files:
- applicationpreprocess.py
- bureaupreprocess.py
- creditpreprocess.py
- customtransformer.py
- installmentpreprocess.py
- pospreprocess.py
- previouspreprocess.py

## Technologies Used
- Seaborn
- Matplotlib
- NumPy
- Pandas
- Scikit-learn
- Optuna
- XGBoost
- LightGBM
- CatBoost
- Streamlit
- Google Cloud Platform (GCP)
- Docker

## Source
- [Home Credit Group Dataset](https://www.kaggle.com/c/home-credit-default-risk)

## Summary

1. **Target Feature Identification:**
   - The target feature for model training is the default risk of loan applications, denoted by the *TARGET* feature.
   - The dataset exhibits a lack of clear patterns, as clients are a heterogeneous group with significant diversity.

2. **Important Predictive Features:**
   - The most critical factors for predicting default risk include days employed, credit debt ratio, active credit overdue, and organization/occupation type.
   - External sources (possibly indicating credit score) also play a significant role in prediction.

3. **Model Performance:**
   - The chosen model, a CatBoost classifier with feature selection, achieved a good performance with the ROC AUC of 0.794.
   - The model's primary objective is to minimize false negatives (risky applicants classified as not risky), with a threshold set at 0.4.

4. **Model Interpretability:**
   - SHAP values were used to assess the global relationships of features, highlighting the importance of external sources, credit debt, and days employed.
