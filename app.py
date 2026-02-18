import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# Load model and helpers
model = joblib.load("models/churn_model.pkl")
features_names = joblib.load("models/features_names.pkl")
cat_features = joblib.load("models/cat_features.pkl")

st.title("Customer Churn Prediction App")

st.write("""
This application predicts the probability of customer churn.
The model was trained using historical customer behavior data.
""")

# Model metrics table
col1, col2, col3, col4 = st.columns(4)

col1.metric("ROC-AUC", "0.92")
col2.metric("Accuracy", "0.82")
col3.metric("Recall (Churn)", "0.82")
col4.metric("F1-Score", "0.71")



# Input fields
with st.sidebar:
    st.header("Customer Information")
    st.write("Enter customer information below to predict churn probability")

    tenure = st.number_input("Tenure (months)", min_value=0, max_value=72, value=12)
    monthly_charges = st.number_input("Monthly charges", min_value=18.25, max_value=118.0)
    total_charges = st.number_input("Total charges", min_value=monthly_charges, max_value=8684.80)
    contract_type = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Is paperless billing?", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Bank transfer (automatic)", "Credit card (automatic)", "Electronic check", "Mailed check"])
    gender = st.selectbox("Gender", ["Male", "Female"])
    senior = st.selectbox("Is senior citizen? (1 - Yes, 0 - No)", [1, 0])
    has_partner = st.selectbox("Has partner?", ["Yes", "No"])
    has_dependents = st.selectbox("Has dependents?", ["Yes", "No"])

    has_internet = st.selectbox("Has internet? (1 - Yes, 0 - No)", [1, 0])

    if has_internet == 1:
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic"])
        online_security = st.selectbox("Has online security?", ["Yes", "No"])
        online_backup = st.selectbox("Has online backup?", ["Yes", "No"])
        device_protection = st.selectbox("Has device protection?", ["Yes", "No"])
        tech_support = st.selectbox("Has tech support?", ["Yes", "No"])
        streaming_tv = st.selectbox("Has streaming tv?", ["Yes", "No"])
        streaming_movies = st.selectbox("Has streaming movies?", ["Yes", "No"])
    else:
        internet_service = "No"
        online_security = "No"
        online_backup = "No"
        device_protection = "No"
        tech_support = "No"
        streaming_tv = "No"
        streaming_movies = "No"

    has_phone = st.selectbox("Has phone? (1 - Yes, 0 - No)", [1, 0])

    if has_phone == 1:
        multiple_lines = st.selectbox("Has multiple lines?", ["Yes", "No"])
    else:
        multiple_lines = "No"


# Build input data frame
input_data = pd.DataFrame([{
    "TenureMonths": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "PaperlessBilling": paperless_billing,
    "PaymentMethod": payment_method,
    "Type": contract_type,
    "Gender": gender,
    "SeniorCitizen": senior,
    "Partner": has_partner,
    "Dependents": has_dependents,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "OnlineBackup": online_backup,
    "DeviceProtection": device_protection,
    "TechSupport": tech_support,
    "StreamingTV": streaming_tv,
    "StreamingMovies": streaming_movies,
    "MultipleLines": multiple_lines,
    "HasInternet": has_internet,
    "HasPhone": has_phone

}])

#st.write(input_data)

input_data = input_data.reindex(columns=features_names)

prediction_proba = model.predict_proba(input_data)[:,1][0]

# Threshold slider
threshold = st.slider(
    "Decision Threshold",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.01
)

st.write("""
- Lower threshold → Higher Recall (catch more churners)
- Higher threshold → Higher Precision (fewer wasted retention offers)
""")

# Prediction button
if st.sidebar.button("Predict Churn"):

    #proba = model.predict_proba(input_data)
    #st.write(proba)

    #prediction_proba = model.predict_proba(input_data)[:,1][0]

    st.subheader("Prediction Result")
    st.write(f"Churn Probability: {prediction_proba:.2%}")

    if prediction_proba > threshold:
        st.error("High Risk of Churn")
    else:
        st.success("Low Risk of Churn")


# Confusion Matrix
st.subheader("Confusion Matrix (Test Set)")

cm = [[850, 186],
      [65, 308]]

cm_df = pd.DataFrame(
    cm,
    index=["Actual: No Churn", "Actual: Churn"],
    columns=["Predicted: No", "Predicted: Churn"]
)

#st.dataframe(cm_df)

fig, ax = plt.subplots()

im = ax.imshow(cm, cmap="Blues")

for i in range(2):
    for j in range(2):
        color = "white" if cm[i][j] > np.max(cm)/2 else "black"
        ax.text(j, i, cm[i][j], ha="center", va="center", color=color, fontsize=12)

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])

ax.set_xticklabels(["No Churn", "Churn"])
ax.set_yticklabels(["No Churn", "Churn"])

ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")

fig.colorbar(im)
st.pyplot(fig)

st.write("""
- True Positives (850): Correctly identified churners.
- False Negatives (186): Missed churners (business risk).
- False Positives (65): Unnecessary retention offers.
- True Negatives (308): Correctly identified non-churners.
""")

st.write("""
Business Strategy:
Since missing a churner (False Negative) is more costly than sending an unnecessary retention offer (False Positive),
we prefer a lower threshold to increase recall.
""")

# Feature Importance
importance = model.get_feature_importance()
importance_df = pd.DataFrame({
    "Feature": features_names,
    "Importance": importance
}).sort_values(by="Importance", ascending=False)

st.subheader("Top Feature Importances")
st.bar_chart(importance_df.head(10).set_index("Feature"))
