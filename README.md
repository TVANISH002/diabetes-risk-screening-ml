# 🩺 Diabetes Risk Screening using Machine Learning

An end-to-end machine learning application that estimates **diabetes risk** from basic
patient health indicators.  
The project focuses on **practical ML system design**, emphasizing evaluation,
interpretability, and deployment rather than model training alone.

⚠️ This application is built for learning and demonstration purposes only and is **not medical advice**.

---

## Problem Statement
Diabetes is a chronic condition where early screening and risk awareness can support
timely intervention. Using historical patient health data, this project builds a
binary classification model to estimate whether an individual is at **higher or lower
risk** of diabetes based on commonly available medical features.

**Target definition**
- `1` → Higher diabetes risk  
- `0` → Lower diabetes risk  

---

## Dataset
The project uses the **PIMA Indians Diabetes Dataset**, which contains patient-level
medical attributes commonly used for diabetes risk modeling.

**Features**
- Pregnancies  
- Glucose  
- Blood Pressure  
- Skin Thickness  
- Insulin  
- BMI  
- Diabetes Pedigree Function  
- Age  

Some medical features contain **invalid zero values** (e.g., glucose or BMI = 0).
These are treated as missing values during preprocessing.

---

## Modeling Approach
The project follows a complete machine learning lifecycle.

- Medical features with invalid zero values are replaced with missing values and
  imputed using median statistics  
- All numerical features are standardized to support distance-based models  
- A **Support Vector Machine (SVM)** classifier is trained using a pipeline that
  combines preprocessing and modeling  
- Hyperparameters such as kernel type, regularization strength, and class weighting
  are tuned using cross-validation  

To improve interpretability, predicted probabilities are **calibrated** before
deployment.  
Instead of relying on a default 0.50 decision threshold, the classification threshold
is selected using **recall-based evaluation**, which better reflects screening-style
requirements.

---

## Model Evaluation
The model is evaluated using multiple metrics to understand real-world behavior:

- ROC-AUC  
- Precision–Recall AUC  
- Precision, Recall, and Accuracy  
- Confusion Matrix  

Evaluation artifacts are stored and reused by the application to improve transparency.
Threshold tuning helps balance false negatives and false positives, which is especially
important in healthcare-style screening problems.

---

## Streamlit Deployment (User Interface)
The trained and calibrated model is deployed using **Streamlit**, providing an
interactive web interface.

The application allows users to:
- Enter patient health information  
- Receive an estimated diabetes risk probability  
- View screening results categorized as **Lower**, **Moderate**, or **Higher** risk  
- Review model details and evaluation metrics in the sidebar  

This demonstrates how a machine learning model can be packaged as a **usable
decision-support tool**, rather than remaining as a notebook.

---

## API Design (FastAPI)
In addition to the Streamlit interface, the project includes a **FastAPI-based inference
service** (`api.py`) to demonstrate scalable system design.

The API:
- Loads the same trained and calibrated model used by Streamlit  
- Exposes prediction endpoints for programmatic access  
- Automatically generates OpenAPI (Swagger) documentation  
- Separates model inference from the user interface  

This API is currently used for architectural demonstration and local development.

---

## Cloud Deployment Roadmap (AWS)
The project is structured to support future cloud deployment. Planned extensions include:

- Containerizing the FastAPI service using Docker  
- Deploying the API on AWS (EC2, ECS, or Elastic Beanstalk)  
- Storing trained model artifacts in Amazon S3  
- Using the Streamlit app as a frontend consuming the cloud-hosted API  

While the current deployment focuses on Streamlit for simplicity, the presence of an
API layer demonstrates **production readiness**.

---

## Project Structure
- `app.py` – Streamlit application for interactive inference  
- `api.py` – FastAPI service for model inference  
- `train.py` – Model training and hyperparameter tuning  
- `evaluate.py` – Threshold selection and evaluation  
- `models/` – Saved model and threshold artifacts  
- `reports/` – Stored evaluation metrics  
- `requirements.txt` – Project dependencies  

---

## Key Learnings
- Real-world data requires careful preprocessing and validation  
- Probability calibration improves interpretability of predictions  
- Threshold selection is critical in screening-style applications  
- Separating UI and backend improves scalability and maintainability  
- Deployment surfaces usability and modeling limitations early  

---

## Future Improvements
- Add baseline model comparisons (Logistic Regression, Tree-based models)  
- Perform feature importance and explainability analysis  
- Deploy the FastAPI service on AWS  
- Add monitoring and logging for production inference  

---

## Disclaimer
This project is for educational and demonstration purposes only.
It is not intended for medical diagnosis or clinical decision-making.
