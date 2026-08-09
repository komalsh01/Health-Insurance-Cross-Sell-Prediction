import joblib
import pandas as pd

#load saved model
model = joblib.load("models/random_forest_model.pkl")
scaler = joblib.load("models/scaler.pkl")
threshold = joblib.load("models/threshold.pkl")


def predict_response(
    Gender,
    Age,
    Driving_License,
    Region_Code,
    Previously_Insured,
    Vehicle_Age,
    Vehicle_Damage,
    Annual_Premium,
    Policy_Sales_Channel,
    Vintage
):

    input_data = pd.DataFrame({
        "Gender": [Gender],
        "Age": [Age],
        "Driving_License": [Driving_License],
        "Region_Code": [Region_Code],
        "Previously_Insured": [Previously_Insured],
        "Vehicle_Age": [Vehicle_Age],
        "Vehicle_Damage": [Vehicle_Damage],
        "Annual_Premium": [Annual_Premium],
        "Policy_Sales_Channel": [Policy_Sales_Channel],
        "Vintage": [Vintage]
    })

    probability = model.predict_proba(input_data)[0][1]

    prediction = int(probability >= threshold)

    return prediction, probability

