# Diabetes Risk Screening using Machine Learning

This project is an end-to-end machine learning application that predicts the risk of diabetes based on patient health indicators. The goal of this project is to demonstrate how a machine learning model can be taken from raw data to a usable application with proper evaluation, interpretation, and deployment.

The project emphasizes practical ML system design rather than just model training.

This application is built for learning and demonstration purposes only and is not intended to provide medical advice.

---

## Problem Statement

Diabetes is a chronic condition that benefits from early screening and risk awareness. Using historical patient data, this project aims to build a binary classification model that estimates whether a person is at higher or lower risk of diabetes based on medical features such as glucose level, BMI, age, and insulin.

Target definition:
- 1 → Higher diabetes risk
- 0 → Lower diabetes risk

---

## Dataset

The project uses the PIMA Indians Diabetes dataset, which contains patient-level medical attributes:

- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  
- Outcome (target variable)

Some features contain medically invalid zero values (for example, glucose or BMI equal to zero). These values are treated as missing during preprocessing.

---

## Approach

The project follows a complete machine learning lifecycle.

First, the data is cleaned by replacing medically invalid zero values with missing values and imputing them using median statistics. All numerical features are standardized to support distance-based models.

A Support Vector Machine (SVM) classifier is trained using a pipeline that includes preprocessing and modeling steps. Hyperparameters such as kernel type, regularization strength, and class weighting are tuned using cross-validation.

To make predictions more interpretable, the model is calibrated to output meaningful probability scores. Instead of relying on a fixed 0.5 decision threshold, the classification threshold is tuned using recall-based evaluation, which is more appropriate for screening-style problems.

---

## Model Evaluation

The model is evaluated using multiple metrics to understand its real-world behavior:

- ROC-AUC  
- Precision-Recall AUC  
- Precision, Recall, and Accuracy  
- Confusion Matrix  

Evaluation results are saved and reused by the application to improve transparency. Threshold tuning helps balance false negatives and false positives, which is especially important in healthcare-style screening problems.

---

## Streamlit Deployment (User Interface)

The trained and calibrated model is deployed using Streamlit, providing an interactive web interface.

The Streamlit application allows users to:
- Enter patient health information
- Receive an estimated diabetes risk probability
- View screening results categorized as lower, moderate, or higher risk
- See model details and evaluation metrics in the sidebar

This demonstrates how a machine learning model can be packaged as a usable decision-support tool rather than remaining as a notebook.

---

## API Design (FastAPI)

In addition to the Streamlit interface, this project includes a FastAPI-based inference service (`api.py`).

The API:
- Loads the same trained and calibrated model used by Streamlit
- Exposes prediction endpoints for programmatic access
- Automatically generates API documentation using Swagger (OpenAPI)
- Separates model inference from the user interface

The API is currently used for local development and architectural demonstration.

---

## Cloud Deployment Roadmap (AWS)

The project is structured to support future cloud deployment. Planned extensions include:

- Containerizing the FastAPI service using Docker
- Deploying the API on AWS (EC2, ECS, or Elastic Beanstalk)
- Storing trained model artifacts in Amazon S3
- Using the Streamlit app as a frontend consuming the cloud-hosted API

While the current deployment focuses on Streamlit for simplicity, the presence of the API layer demonstrates readiness for scalable, production-oriented ML deployment.

---

## Project Structure

- app.py – Streamlit application for interactive inference  
- api.py – FastAPI service for model inference  
- train.py – Model training and hyperparameter tuning  
- evaluate.py – Threshold selection and evaluation  
- models/ – Saved model and threshold artifacts  
- reports/ – Stored evaluation metrics  
- requirements.txt – Project dependencies  

---

## Key Learnings

- Real-world data requires careful preprocessing and validation  
- Probability calibration improves interpretability of ML predictions  
- Threshold selection is critical in screening-style applications  
- Separating UI and backend improves scalability and system design  
- Deployment reveals usability and modeling limitations early  

---

## Future Improvements

- Add baseline models for comparison (Logistic Regression)
- Perform feature importance analysis
- Deploy the FastAPI service on AWS
- Add monitoring and logging for production use

---

## Disclaimer

This project is for educational and demonstration purposes only. It is not intended for medical diagnosis or clinical decision-making.
