# 🚀 MLOps Projects Portfolio

This repository contains **end-to-end MLOps projects** demonstrating model training, CI/CD automation, containerization, and cloud deployment on AWS.

Each project is maintained in a **separate branch** and follows real-world MLOps practices such as:
- automated training
- model persistence
- API-based inference
- Dockerization
- CI/CD with GitHub Actions
- AWS integration (ECR, EC2, SageMaker)

---

## 📂 Projects Overview

| Project | Description | Branch |
|------|------------|--------|
| 🌸 Iris Prediction | Feature-based ML classification with CI/CD | `iris_model` |
| 🐶🐱 Cat vs Dog | Image classification using CNN | `image_classification` |
| 📉 Customer Churn | Continuous retraining-ready churn prediction system | `customer_churn` |

---

## 1️⃣ Iris Classification (Feature-based Prediction)

**Branch:** `iris-mlops`

### 📌 Problem
Predict the Iris flower species based on sepal and petal measurements.

### 🧠 ML Model
- Scikit-learn classifier
- Trained on Iris dataset

### 🏗️ MLOps Features
- Automated training using GitHub Actions
- Model saved and versioned
- FastAPI inference service
- Dockerized application
- Docker image pushed to AWS ECR
- Deployed and tested on EC2

### 🔁 Workflow




---

## 2️⃣ Cat vs Dog Image Classification

**Branch:** `cat-dog-mlops`

### 📌 Problem
Classify an image as either **Cat** or **Dog**.

### 🧠 ML Model
- Convolutional Neural Network (TensorFlow/Keras)
- Trained on Kaggle dataset

### 🏗️ MLOps Features
- Model training on AWS SageMaker Studio
- Saved model in `.keras` format
- FastAPI-based inference API
- Dockerized inference service
- CI/CD pipeline for container build & push
- Image deployment via AWS ECR

### 🔁 Workflow



---

## 3️⃣ Customer Churn Prediction (Continuous ML System)

**Branch:** `customer_churn`

### 📌 Problem
Predict whether a customer will churn based on telecom usage data.

### 🧠 ML Model
- Logistic Regression (Scikit-learn)
- Trained on structured CSV data

### 🏗️ MLOps Features
- CI-based model training using GitHub Actions
- Model automatically committed after training
- FastAPI inference service
- Dockerized deployment
- Image pushed to AWS ECR
- Deployed on EC2
- Designed for **continuous retraining**
