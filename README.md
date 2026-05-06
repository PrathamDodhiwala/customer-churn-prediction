# 📊 Customer Churn Prediction App

A complete **Machine Learning + Streamlit Web App** that predicts whether a customer is likely to **churn (leave)** or **stay**, based on their data.

This project is designed to be **beginner-friendly**, **resume-ready**, and **fully functional** with real-world data.


## 🚀 Features

* ✅ Train a Machine Learning model (Random Forest)
* ✅ Upload CSV files to predict customer churn
* ✅ Automatic data preprocessing (encoding + scaling)
* ✅ Handles missing and extra columns automatically
* ✅ Displays churn predictions with probability (%)
* ✅ Download predictions as CSV
* ✅ Interactive Streamlit UI
* ✅ Built-in visualization (bar chart)


## 🧠 Tech Stack

* **Python**
* **Pandas**
* **Scikit-learn**
* **Streamlit**
* **Joblib**


## 📂 Project Structure

```
customer_churn_project/
│
├── customer_churn_app.py          # Main Streamlit app
├── churn_model.pkl                # Saved ML model (auto-generated)
├── scaler.pkl                     # Scaler file (auto-generated)
├── feature_names.pkl              # Feature list (auto-generated)
├── WA_Fn-UseC_-Telco-Customer-Churn.csv   # Dataset (download manually)
└── README.md                      # Project documentation
```


## 📥 Dataset

Download the dataset from Kaggle:

👉 https://www.kaggle.com/datasets/blastchar/telco-customer-churn

After downloading:

* Extract the file
* Place `WA_Fn-UseC_-Telco-Customer-Churn.csv` in the project folder


## ⚙️ Installation & Setup

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/customer-churn-prediction.git
cd customer-churn-prediction
```


### 2️⃣ Install Dependencies

```bash
pip install streamlit pandas scikit-learn joblib
```


### 3️⃣ Run the Application

```bash
streamlit run customer_churn_app.py
```


## 🖥️ How to Use

1. Run the app using Streamlit
2. Click on **"Train Churn Model"** (only first time)
3. Upload a CSV file with customer data
4. View predictions:

   * Churn (Yes/No)
   * Churn Probability (%)
5. Download results as CSV


## 📊 Input Format

Your uploaded CSV should contain columns similar to the dataset, such as:

* `gender`
* `SeniorCitizen`
* `tenure`
* `MonthlyCharges`
* `Contract`
* `InternetService`
* etc.

⚠️ Don’t worry if:

* Some columns are missing → handled automatically
* Extra columns exist → dropped automatically


## 📈 Output Example

| Churn_Prediction | Churn_Probability (%) |
| ---------------- | --------------------- |
| Yes              | 82.45                 |
| No               | 12.30                 |


## 🎯 Model Details

* Algorithm: **Random Forest Classifier**
* Preprocessing:

  * Label Encoding (categorical data)
  * Standard Scaling
* Train/Test Split: 80/20
* Evaluation Metric: Accuracy


## 💡 Key Highlights

* Fully automated preprocessing pipeline
* Works with real-world messy datasets
* Clean UI for demonstration
* Great for **resume projects** and **internship portfolios**


## 🚀 Future Improvements

* Add manual input form (no CSV needed)
* Deploy on Streamlit Cloud / Render / AWS
* Add multiple ML models (Logistic Regression, XGBoost)
* Improve accuracy with feature engineering
* Add model explainability (SHAP)


## 🤝 Contributing

Feel free to fork this repo and improve it!


## 📜 License

This project is open-source and free to use.

## Author

Pratham Dodhiwala


## ⭐ If You Like This Project

Give it a ⭐ on GitHub and share it!

