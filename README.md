# ${\color{blue}\text{Churn Project}}$

## Project description
The telecom operator Interconnect would like to be able to forecast their churn of clients. If it's discovered that a user is planning to leave, they will be offered promotional codes and special plan options. Interconnect's marketing team has collected some of their clientele's personal data, including information about their plans and contracts.

Interconnect mainly provides two types of services:

1. Landline communication. The telephone can be connected to several lines simultaneously.
2. Internet. The network can be set up via a telephone line (DSL, digital subscriber line) or through a fiber optic cable.

Some other services the company provides include:

- Internet security: antivirus software (DeviceProtection) and a malicious website blocker (OnlineSecurity)
- A dedicated technical support line (TechSupport)
- Cloud file storage and data backup (OnlineBackup)
- TV streaming (StreamingTV) and a movie directory (StreamingMovies)
- The clients can choose either a monthly payment or sign a 1- or 2-year contract. They can use various payment methods and receive an electronic invoice after a transaction.

## Dataset source
The dataset was provided as part of a data science bootcamp and contains anonymized customer behavior and subscription information used for churn modeling.

## Model performance metrics

Model: CatBoost Classifier

Hyperparameters: depth=4, learning_rate=0.03, l2_leaf_reg: 5

#### Test Set Metrics

- ROC-AUC: 0.92
- Accuracy: 0.82
- Recall (churn): 0.82
- F1-score: 0.71


## Business Context

Customer churn represents a significant threat to subscription-based revenue models. Retaining existing customers is typically more cost-effective than acquiring new ones.

This project develops a churn prediction model to identify customers at risk of leaving, enabling targeted retention interventions and improved marketing efficiency.

## Deployment link
https://churn-project-rodriguez-robcar.streamlit.app/

## Screenshot

<img width="2552" height="2065" alt="screencapture-churn-project-rodriguez-robcar-streamlit-app-2026-02-18-20_51_32" src="https://github.com/user-attachments/assets/42bb52ad-0de4-4297-a37b-a2dcaec84b19" />
