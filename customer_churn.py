import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------
# STREAMLIT PAGE SETUP
# -------------------------
st.set_page_config(page_title="Customer Churn Prediction", layout="wide")
st.title("📊 Customer Churn Prediction App")

st.write(
    """
This app predicts whether a customer will **churn (leave)** or **stay** based on their details.  
You can **train a model** and then **upload customer data** to get predictions.
"""
)


# -------------------------
# STEP 1: MODEL TRAINING FUNCTION
# -------------------------
def train_model():
    st.info("🔄 Training model... please wait")

    # ✅ Load dataset locally (download this from Kaggle and place in same folder)
    # Kaggle dataset name: WA_Fn-UseC_-Telco-Customer-Churn.csv
    data_path = "WA_Fn-UseC_-Telco-Customer-Churn.csv"

    if not os.path.exists(data_path):
        st.error(
            """
        ❌ Dataset not found!
        Please download it from Kaggle:
        👉 https://www.kaggle.com/datasets/blastchar/telco-customer-churn  
        and place it in the same folder as this app.
        """
        )
        st.stop()

    data = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    data = data.dropna()

    # Encode categorical variables
    label_encoder = LabelEncoder()
    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = label_encoder.fit_transform(data[col].astype(str))

    # Split into features and labels
    X = data.drop("Churn", axis=1)
    y = data["Churn"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Scale features
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Train model
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)

    # Evaluate
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    # Save model, scaler, and feature names
    joblib.dump(model, "churn_model.pkl")
    joblib.dump(scaler, "scaler.pkl")
    joblib.dump(list(X.columns), "feature_names.pkl")

    st.success(f"✅ Model trained successfully! Accuracy: **{acc*100:.2f}%**")

    return model, scaler, list(X.columns)


# -------------------------
# STEP 2: LOAD OR TRAIN MODEL
# -------------------------
if os.path.exists("churn_model.pkl"):
    model = joblib.load("churn_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_names = joblib.load("feature_names.pkl")
    st.success("✅ Pre-trained model loaded successfully.")
else:
    st.warning("⚠️ No model found. Click the button below to train one.")
    if st.button("Train Churn Model"):
        model, scaler, feature_names = train_model()
    else:
        st.stop()

# -------------------------
# STEP 3: PREDICTION SECTION
# -------------------------
st.subheader("📂 Upload Customer Data to Predict Churn")

uploaded_file = st.file_uploader("Upload a CSV file with customer data", type=["csv"])

if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.write("### 🧾 Uploaded Data Preview")
    st.dataframe(data.head())

    # 1️⃣ Encode categorical features (convert text → numbers)
    from sklearn.preprocessing import LabelEncoder

    for col in data.select_dtypes(include=["object"]).columns:
        data[col] = LabelEncoder().fit_transform(data[col].astype(str))

    # 2️⃣ Add missing columns (fill with zeros)
    missing_cols = set(feature_names) - set(data.columns)
    for col in missing_cols:
        data[col] = 0

    # 3️⃣ Drop extra columns not used by the model
    extra_cols = set(data.columns) - set(feature_names)
    if extra_cols:
        st.warning(f"🧹 Dropping extra columns: {list(extra_cols)}")
        data = data.drop(columns=list(extra_cols))

    # 4️⃣ Reorder columns to match training
    data = data[feature_names]

    # 5️ Scale data
    data_scaled = scaler.transform(data)

    # 6️ Make predictions
    predictions = model.predict(data_scaled)
    prediction_probs = model.predict_proba(data_scaled)[:, 1]

    data["Churn_Prediction"] = ["Yes" if p == 1 else "No" for p in predictions]
    data["Churn_Probability (%)"] = (prediction_probs * 100).round(2)

    st.write("### 🔍 Prediction Results")
    st.dataframe(data[["Churn_Prediction", "Churn_Probability (%)"]])

    # 7️⃣ Downloadable results
    csv = data.to_csv(index=False).encode("utf-8")
    st.download_button(
        "📥 Download Predictions", csv, "churn_predictions.csv", "text/csv"
    )

    # 8️⃣ Optional: Bar chart visualization
    st.write("### 📊 Churn Probability Chart")
    sorted_data = data.sort_values(by="Churn_Probability (%)", ascending=False)
    st.bar_chart(sorted_data["Churn_Probability (%)"])

else:
    st.info("👆 Upload a CSV file with customer data to generate churn predictions.")
