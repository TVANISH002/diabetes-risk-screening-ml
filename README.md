# Diabetes Risk Screening using Machine Learning

This project is a complete machine learning application that predicts the risk of diabetes based on basic patient health information. The main goal of this project is to show how a machine learning model can be built, evaluated, and converted into a working application that people can actually use.

Instead of focusing only on model accuracy, this project focuses on the **end-to-end ML workflow** — data preprocessing, model training, evaluation, decision thresholding, and deployment using a simple user interface.

This project is created for **learning and demonstration purposes only** and is not intended to provide medical advice.

## What This Application Does

The application allows a user to enter basic health details such as glucose level, BMI, age, and insulin values. Based on these inputs, the machine learning model estimates the **probability of diabetes risk**.

The app then:
- Shows the predicted risk as a probability value
- Categorizes the result into **Lower Risk**, **Moderate Risk**, or **Higher Risk**
- Displays basic information about the model and evaluation metrics
- Warns the user when medically invalid values (like zero glucose or BMI) are entered

This simulates how a real-world **screening or decision-support tool** might work.

## Problem Statement

Diabetes is a long-term condition where early screening can help with preventive care. Using historical patient data, this project builds a **binary classification model** that predicts whether a person is likely to be at higher or lower risk of diabetes.

Target definition:
- `1` → Higher diabetes risk  
- `0` → Lower diabetes risk
  
## Dataset

The project uses the **PIMA Indians Diabetes dataset**, which contains medical data for female patients. The dataset includes the following features:

- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  
- Outcome (target variable)

Some medical measurements in the dataset contain zero values that are not realistic in practice (for example, glucose or BMI equal to zero). These values are treated as missing during preprocessing.

## Data Preprocessing

Before training the model, the following preprocessing steps are performed:

- Medically invalid zero values are replaced with missing values
- Missing values are filled using median imputation
- All numerical features are standardized to ensure fair contribution to the model

These steps help make the model more reliable and closer to real-world conditions.

---

## Model Building

A **Support Vector Machine (SVM)** classifier is used for this project.

The model is built using a pipeline that includes:
- Data imputation
- Feature scaling
- SVM classification

Model hyperparameters such as kernel type, regularization strength, and class weighting are tuned using cross-validation.

To make the predictions easier to interpret, the model is **calibrated** so that it outputs probability scores instead of just class labels.

## Decision Threshold Selection

Instead of using a default threshold of 0.5, the classification threshold is selected based on **recall-oriented evaluation**. This approach is more suitable for screening problems, where missing a high-risk case can be more costly than a false alarm.

The chosen threshold is saved and reused by the application.

## Model Evaluation

The model is evaluated using multiple metrics to understand its behavior from different angles:

- ROC-AUC  
- Precision-Recall AUC  
- Accuracy  
- Precision and Recall  
- Confusion Matrix  

Evaluation results are stored and displayed inside the application for transparency.

## Streamlit Deployment (Current Implementation)

The trained and calibrated model is deployed using **Streamlit**, providing a simple web-based user interface.

The Streamlit app:
- Collects patient input through form fields
- Validates suspicious or invalid inputs
- Runs the trained ML model for inference
- Displays probability-based risk results
- Shows model metrics and configuration in the sidebar

This deployment demonstrates how a machine learning model can move beyond notebooks and be used in an interactive application.

## Project Structure

- `app.py` – Streamlit application used for prediction and visualization  
- `train.py` – Script used to train and tune the machine learning model  
- `evaluate.py` – Script used to select decision thresholds and evaluate performance  
- `models/` – Saved model and threshold files  
- `reports/` – Stored evaluation metrics  
- `requirements.txt` – Project dependencies  

## What This Project Demonstrates

- Understanding of the full machine learning workflow  
- Handling of real-world data quality issues  
- Use of probability-based predictions instead of hard labels  
- Importance of threshold selection in classification problems  
- Ability to deploy ML models as usable applications
  
## Future Scope (Planned Improvements)

As a future enhancement, this project can be extended in the following simple and practical ways:

- Create a basic backend API using FastAPI for model predictions
- Package the API using Docker
- Deploy the Docker container on AWS (for example, on an EC2 instance)
- Connect the Streamlit app to the AWS-hosted API instead of loading the model locally

These improvements are planned to explore **basic cloud deployment concepts using Docker and AWS**, without adding unnecessary complexity.
## Disclaimer

This project is for educational and demonstration purposes only. It is not intended for medical diagnosis or clinical decision-making.
