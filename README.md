# Fraud Detection System using Credit Card Dataset
This project detects fraudulent transactions using machine learning on the Credit Card Fraud Detection Dataset from Kaggle. The system uses techniques like SMOTE for handling imbalanced data and a Random Forest model for classification. The project includes model evaluation metrics and a command-line testing interface.

# Dataset Source
Credit Card Fraud Detection Dataset-https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

# Project Steps
1. Data Preprocessing
- Loaded and cleaned the dataset (creditcard.csv)
- Scaled features using StandardScaler
- Handled data imbalance using SMOTE (Synthetic Minority Oversampling Technique)

2. Model Training
- Trained a Random Forest Classifier
- Features: Time, Amount, and anonymized variables V1 to V28
- Target: Class (0 for legitimate, 1 for fraud)

3. Model Evaluation
- Evaluated using Confusion Matrix, Precision, Recall, and F1-score
- Plotted ROC and Precision-Recall curves for better insights

4. Testing Interface
- A simple command-line input system that allows users to enter 30 transaction values and get a prediction (fraud or legit)

# How to Run the Project
- Prerequisites
Make sure you have Python 3 installed. Install the required libraries:
pip install pandas numpy scikit-learn imbalanced-learn matplotlib seaborn
- Running the Main Script
Train and Evaluate the Model:
python fraud_detection.py
# This script:
- Loads and preprocesses the data
- Trains the Random Forest model
- Displays evaluation metrics and plots

After training, the script asks you to enter 30 values (Time, Amount, V1 to V28). The system then tells you whether the transaction is likely fraudulent or legitimate.

# Observations
- The dataset is highly imbalanced (~99.8% legit, ~0.2% fraud). SMOTE helped balance the training data.
- The Random Forest classifier achieved high accuracy, but precision and recall were key for fraud detection:
- High precision ensures that flagged frauds are truly fraud.
- High recall ensures we catch most fraudulent transactions.
- ROC AUC and PR curves confirmed good separation between classes.
