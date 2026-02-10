# Spark ML Pipeline on Amazon EMR — Customer Churn

This project implements an end-to-end Spark ML pipeline for predicting customer churn using Amazon EMR.

## Dataset
Bank Customer Churn Dataset (Kaggle)  
Target variable: **Exited**
- 0 = No churn
- 1 = Churn

Features used:
- CreditScore
- Age
- Tenure
- Balance
- NumOfProducts
- EstimatedSalary
- Geography (categorical)
- Gender (categorical)

## Pipeline Stages
1. Load data from HDFS
2. StringIndexer for categorical features
3. OneHotEncoder
4. VectorAssembler
5. StandardScaler
6. Logistic Regression
7. Evaluation (Accuracy and AUC)

## Distributed Execution: spark-submit --master yarn --deploy-mode client churn_pipeline.py



## Experiment (Option B — Feature Ablation)

Two experiments were performed:

**Experiment A**
- Numeric + categorical features

**Experiment B**
- Numeric features only (categorical features removed)

Metrics compared:
- Accuracy
- AUC
- Runtime

## HDFS Path: hdfs:///user/hadoop/churn_input/Churn_Modelling.csv


The job is executed on Amazon EMR using YARN:

