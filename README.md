# Customer Churn Prediction – Machine Learning Project

## Project Overview
This project focuses on predicting customer churn using machine learning techniques. The objective is to analyze customer data, perform feature engineering, train multiple models, and evaluate their performance while understanding the limitations imposed by the dataset.

The project follows a complete end-to-end machine learning workflow including data exploration, preprocessing, modeling, and evaluation.

---

## Problem Statement
Customer churn refers to customers discontinuing a service. Predicting churn enables organizations to take proactive retention measures.

The goal of this project is to build and evaluate machine learning models that predict whether a customer will churn based on available customer attributes.

---

## Project Structure
├── 01_eda.ipynb
├── 02_feature_engineering.ipynb
├── 03_modeling.ipynb
├── Dataset/
│ ├── customer_churn_prediction_dataset.csv
│ ├── X_processed.csv
│ └── y.csv
└── README.md


---

## Exploratory Data Analysis (EDA)
The following steps were performed during EDA:
- Checked dataset shape, data types, and missing values
- Analyzed categorical and numerical feature distributions
- Examined churn distribution
- Studied relationships between churn and:
  - Contract type
  - Tenure
  - Payment method
  - Internet service
  - Technical support

### Key Observations
- Churn distribution is relatively balanced
- Customers with shorter tenure exhibit higher churn rates
- Contract type and service-related features show noticeable influence on churn
- No major data quality issues were identified

---

## Feature Engineering
The following feature engineering steps were applied:
- Created binary target variable `churn_flag`
- Dropped identifier and leakage columns (`customerID`, `Churn`)
- Created tenure buckets to capture customer lifecycle behavior
- Removed `TotalCharges` due to strong correlation with tenure
- Applied one-hot encoding to categorical variables
- Prepared a clean, model-ready dataset

The final dataset:
- Contains only numerical features
- Has no missing values
- Is suitable for machine learning models

---

## Modeling Approach

### 1. Logistic Regression (Baseline Model)
Logistic Regression was used as the baseline model.

- Low recall for churn class
- ROC-AUC ≈ 0.43
- Indicates churn behavior is not linearly separable

---

### 2. Random Forest Classifier
Random Forest was used to capture non-linear relationships.

- Slight improvement over Logistic Regression
- ROC-AUC ≈ 0.48
- Still limited predictive performance

---

### 3. XGBoost Classifier
XGBoost was evaluated as a more advanced ensemble model.

- Did not significantly outperform Random Forest
- ROC-AUC ≈ 0.43
- Performance remained close to random guessing

---

## Model Comparison

| Model | ROC-AUC | Performance Summary |
|------|---------|---------------------|
| Logistic Regression | ~0.43 | Weak baseline |
| Random Forest | ~0.48 | Slight improvement |
| XGBoost | ~0.43 | No significant gain |

ROC curve analysis showed that none of the models achieved strong discriminative power.

---

## Key Insights
- Model performance remained limited across all algorithms
- Dataset size is relatively small (~300 samples)
- Lack of behavioral and time-based features restricts predictive performance
- Model choice had less impact than feature quality
- Performance is constrained by data rather than algorithm complexity

---

## Final Conclusion
This project demonstrates a complete machine learning workflow including data exploration, feature engineering, modeling, and evaluation.

Despite testing multiple models, performance remained limited due to dataset constraints. This highlights an important real-world insight: model performance is often driven more by data quality and feature richness than by algorithm selection.

The project successfully demonstrates:
- End-to-end ML pipeline implementation
- Model comparison and evaluation
- Proper interpretation of results
- Understanding of real-world ML limitations

---

## Tools and Libraries Used
- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn

---

## Future Improvements
- Use larger datasets with richer behavioral features
- Incorporate time-series or usage-based features
- Apply cross-validation
- Explore cost-sensitive learning
- Perform hyperparameter tuning

---

## Author
Bharath Krishna

Machine Learning Enthusiast | Data Science Learner
