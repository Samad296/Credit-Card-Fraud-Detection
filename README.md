# Credit-Card-Fraud-Detection
# 💳 Credit Card Fraud Detection

## 📖 Project Description
This project addresses the **Credit Card Fraud Detection problem**, where the goal is to identify fraudulent transactions among a massive number of legitimate ones.  

The dataset is **highly imbalanced** — only about 0.17% of transactions are fraud. This makes the problem challenging:  
- A naïve model that always predicts "Not Fraud" would achieve **99% accuracy**, but it would **completely fail** to detect fraud.  
- Therefore, traditional accuracy is misleading. We must focus on **recall** (catching fraud cases) and **precision** (avoiding false alarms).  

To solve this, we experimented with multiple machine learning algorithms and carefully evaluated them using metrics that make sense for **imbalanced data**.

---

## 🎯 Problem Statement
- Fraudulent transactions cost banks and customers **millions of dollars every year**.  
- Detecting fraud is critical, but must be done with **high reliability**:  
  - **Missing frauds** (false negatives) cause financial loss.  
  - **Flagging genuine transactions as fraud** (false positives) hurts customer trust.  
- The challenge: **catch as many frauds as possible while minimizing false alarms**.

---

## 📊 Dataset Characteristics
- **Source**: [Kaggle – Credit Card Fraud Detection](https://www.kaggle.com/mlg-ulb/creditcardfraud)  
- **Size**: 284,807 transactions, 492 frauds (~0.17%).  
- **Features**:  
  - Most features are **PCA-transformed** to protect sensitive data.  
  - `Time` and `Amount` were **not transformed** — we normalized them for consistency.  
- **Target variable**:  
  - `0` → Legitimate transaction  
  - `1` → Fraudulent transaction  

---

## ⚙️ Approach

### 1. **Exploratory Data Analysis (EDA)**
- Checked for **class imbalance** (majority vs minority).  
- Visualized distribution of `Amount` and `Time`.  
- Verified PCA features were already scaled.  

👉 **Why?** EDA helps us understand data imbalance and guides our model + metric choices.  

---

### 2. **Data Preprocessing**
- Applied **StandardScaler** on `Amount` and `Time`.  
- Dropped raw `Amount` and `Time` after creating normalized versions.  

👉 **Why?** Scaling ensures features contribute equally during training.  

---

### 3. **Handling Imbalanced Data**
- Used **SMOTE (Synthetic Minority Oversampling Technique)** to generate synthetic fraud samples.  

👉 **Why?** Without SMOTE, the model is biased towards predicting "Not Fraud". SMOTE balances classes and helps the model learn fraud patterns.  

---

### 4. **Model Building**
We trained and compared multiple models:
- **Logistic Regression** → baseline model.  
- **Random Forest** → handled imbalance well, very strong results.  
- **XGBoost** → achieved the **best performance** after tuning.  

👉 **Why multiple models?** To compare strengths/weaknesses and avoid overfitting to one algorithm.  

---

### 5. **Hyperparameter Tuning**
- Applied **RandomizedSearchCV** on Random Forest and XGBoost.  
- Tuned parameters like `n_estimators`, `max_depth`, `learning_rate`.  

👉 **Why?** Default settings are rarely optimal. Tuning improves recall & precision trade-off.  

---

### 6. **Model Evaluation**
Instead of only accuracy, we used:
- **Confusion Matrix**  
- **Precision, Recall, F1-Score**  
- **ROC-AUC Score**  

👉 **Why?** Fraud detection requires maximizing **recall** (catch frauds), while keeping precision reasonably high (avoid false alarms).  

---

### 7. **Final Model Choice**
- **XGBoost (Tuned)** gave the **best balance**:  
  - High recall (catching frauds).  
  - Good precision (avoiding false alarms).  
  - Stable performance compared to other models.  

👉 **Decision:** Save tuned **XGBoost model** for deployment.  

---

## 📦 Deployment

### Model Saving
- Preprocessing (`scaler.pkl`) and trained model (`xgboost_model.pkl`) saved using **Pickle**.  
- Ensures same transformations are applied during inference.  

### Streamlit App
- Built a **Streamlit dashboard** for easy testing.  
- Users can input transaction details and check if it’s predicted as **Fraud** or **Not Fraud**.  

👉 **Why Streamlit?** Lightweight, interactive, and great for ML model demos.  

---

## 🚀 How to Run

### 1. Clone the repo
```bash
git clone https://github.com/yourusername/credit-card-fraud-detection.git
cd credit-card-fraud-detection
