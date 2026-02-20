import joblib
import pandas as pd

# load trained pipeline
model = joblib.load("churn_model.pkl")

# create a NEW customer (raw format, same as dataset columns)
new_customer = pd.DataFrame([{
    'gender': 'Male',
    'SeniorCitizen': 0,
    'Partner': 'No',
    'Dependents': 'No',
    'tenure': 20,
    'PhoneService': 'Yes',
    'MultipleLines': 'No',
    'InternetService': 'DSL',
    'OnlineSecurity': 'Yes',
    'OnlineBackup': 'No',
    'DeviceProtection': 'Yes',
    'TechSupport': 'No',
    'StreamingTV': 'No',
    'StreamingMovies': 'No',
    'Contract': 'Month-to-month',
    'PaperlessBilling': 'No',
    'PaymentMethod': 'Electronic check',
    'MonthlyCharges': 56.95,
    'TotalCharges': 56.95
}])

# predict class
prediction = model.predict(new_customer)[0]

# predict probability
probability = model.predict_proba(new_customer)[0][1]

print("Churn Prediction:", "YES (will leave)" if prediction == 1 else "NO (will stay)")
print("Churn Risk Probability:", round(probability*100,2), "%")
