# AI SG 22 PROJECT

## Overview
NA

## How to Run
1. pip install -r requirements.txt
2. bash run.sh


# Task 1 Final EDA Analysis

Based on the exploratory data analysis (EDA) conducted on the phishing-website dataset, several important characteristics of the data informed the choice of predictive models. These characteristics relate to feature distributions, non-linearity, skewness, categorical complexity, and the overall structure of the target–feature relationships. The selected models collectively balance interpretability, robustness, and performance.

## 1. Data Characteristics Informing Model Choice

### 1.1 Strong Non-linear Relationships With Target

Multiple features exhibit clear monotonic but non-linear separation between phishing and legitimate websites:

| Feature | Phishing vs Legitimate Pattern |
|---------|--------------------------------|
| DomainAgeMonths (+0.333) | Legitimate sites tend to be much older |
| NoOfExternalRef, NoOfSelfRef, NoOfiFrame | Legitimate sites have significantly more internal/external components |
| LargestLineLength, NoOfURLRedirect, NoOfSelfRedirect | Higher values slightly increase phishing likelihood |

These relationships are not linear, as many features show threshold effects (e.g., old domains → rarely phishing; high redirects → phishing).

**➡️ Tree-based models are suitable** because they naturally capture thresholds and interactions without requiring feature transformations.

### 1.2 Highly Skewed Distributions and Extreme Outliers

Many numerical features exhibit heavy right skew, extreme maximum values, and large variance gaps:

- **LineOfCode**: range up to 418,650
- **LargestLineLength**: range up to 4.3 million  
- **NoOfImage**: extreme outliers (up to 3.1 million)
- Most features have median = 0 but extremely large upper tails

Such distributions violate the assumptions of linear models and distance-based models.

**➡️ Tree-based ensemble models** (Random Forest, Gradient Boosting, XGBoost) are robust to skewness, outliers, and unscaled numeric features.

### 1.3 Informative High-Cardinality Categorical Features

**Hosting Provider** shows strong patterns:
- Free hosting providers (Freehostia, InfinityFree, 000webhost) → ~70% phishing
- Cloud providers (AWS, Azure, Google Cloud) → ~20% phishing

**Also**:
- Robots.txt present → much less phishing
- IsResponsive = 0 (broken site) → far more phishing

These categorical variables carry significant predictive power and interact with numerical features.

**➡️ Models that handle mixed feature types and categorical interactions are required**
**➡️ Tree-based models again are strong candidates**

### 1.4 Need for Interpretability

As this model will be integrated into a browser extension and influence user warnings, interpretability is important:
- Clear explanation of which features contribute most
- Ability to derive decision rules  
- Ability to explain false positives/negatives

**➡️ Logistic Regression provides a transparent, interpretable baseline**

## 2. Final Model Choices

Given these data-driven considerations, the following three models were selected:

### 📌 1. Logistic Regression (Baseline, Interpretable)

**Why?**
- Provides a clear, explainable baseline
- Coefficients allow straightforward interpretation of feature effects
- Useful for validating trends observed in EDA

**Limitations:**
- Assumes linearity
- Not robust to skew/outliers
- May underperform compared to non-linear models

### 📌 2. Random Forest Classifier (Robust Baseline Tree Model)

**Why?**
- Handles extreme skew and outliers without preprocessing
- Captures non-linear feature relationships
- Easily interpretable through feature importances
- Works well on mixed numerical + categorical data

**Strength:**
- Very stable, low risk of overfitting due to ensembling
- Good benchmark tree-based model

### 📌 3. XGBoost / Gradient Boosting Classifier (High-Performance Model)

**Why?**
- Best suited for datasets with skewed, noisy, and high-variance features
- Learns complex interactions between numeric and categorical variables
- Typically achieves state-of-the-art performance on tabular classification
- Provides SHAP values for detailed interpretability

**Strength:**
Handles subtle patterns such as:
- "Low Domain Age + Free Hosting Provider = high phishing probability"
- "Large number of images but site responsive = legitimate"