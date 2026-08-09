import streamlit as st
import pandas as pd
import joblib 
import os
st.set_page_config(
    page_title="Health Insurance Prediction",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>

/* Hide Streamlit Menu */
#MainMenu{
    visibility:hidden;
}

footer{
    visibility:hidden;
}

header{
    visibility:hidden;
}

/* Main Background */

.stApp{
    background:#F5F7FB;
}

/* Remove top padding */

.block-container{
    padding-top:2rem;
    padding-left:4rem;
    padding-right:4rem;
}

/* Navigation Bar */

.navbar{

    background:white;

    border-radius:18px;

    padding:18px 35px;

    border:1px solid #E2E8F0;

    box-shadow:0px 4px 18px rgba(15,23,42,0.08);

    margin-bottom:40px;

}

/* Hero Title */

.hero-title{

    font-size:44px;

    font-weight:700;

    color:#1E293B;

    margin-bottom:10px;

}

/* Hero Subtitle */

.hero-subtitle{

    font-size:18px;

    color:#64748B;

    line-height:1.7;

}

/* Input Labels */

.stSelectbox label,
.stNumberInput label,
.stTextInput label,
.stSlider label,
.stRadio label{

    color:#1E293B !important;

    font-weight:600 !important;

}

/* Streamlit Labels */

[data-testid="stWidgetLabel"] label{

    color:#1E293B !important;

    font-size:16px !important;

    font-weight:600 !important;

}

/* Predict Button */

.stButton > button{

    background:#2563EB;

    color:white;

    border:none;

    border-radius:12px;

    height:55px;

    font-size:18px;

    font-weight:600;

    transition:0.3s;

    box-shadow:0px 6px 16px rgba(37,99,235,0.25);

}

.stButton > button:hover{

    background:#1D4ED8 !important;

    color:white !important;

    border:none !important;

    transform:translateY(-2px);

    box-shadow:0px 10px 22px rgba(37,99,235,0.35);

}

.stButton > button:focus{

    background:#1D4ED8 !important;

    color:white !important;

    border:none !important;

}

.stButton > button:active{

    background:#1E40AF !important;

    color:white !important;

}

/* Customer Card */

.customer-card{

    background:white;

    padding:35px;

    border-radius:20px;

    border:1px solid #E2E8F0;

    box-shadow:0px 6px 22px rgba(15,23,42,0.08);

    margin-top:25px;

}

/* Input Fields */

.stSelectbox > div > div,
.stNumberInput > div > div > input{

    background:white !important;

    border:1px solid #D1D5DB !important;

    border-radius:12px !important;

    color:#1E293B !important;

    box-shadow:0px 3px 10px rgba(15,23,42,0.06);

}

/* Number Input */

.stNumberInput input{

    color:#1E293B !important;

}

/* Selectbox */

.stSelectbox div[data-baseweb="select"]{

    background:white !important;

    border-radius:12px !important;

}

/* Focus */

.stSelectbox div[data-baseweb="select"]:focus-within{

    border:2px solid #2563EB !important;

}

.stNumberInput input:focus{

    border:2px solid #2563EB !important;

    outline:none !important;

}

/* Selected value inside Selectbox */

.stSelectbox div[data-baseweb="select"] span{
    color:#1E293B !important;
    font-weight:500 !important;
}

/* Dropdown text */

.stSelectbox div[data-baseweb="select"] *{
    color:#1E293B !important;
}

/* Dropdown menu options */

div[role="listbox"] div{
    color:#1E293B !important;
    background:white !important;
}

/* Selected option */

div[role="option"]{
    color:#1E293B !important;
}

/* Input text */

input{
    color:#1E293B !important;
}

/* Prediction Result */

.prediction-section{
    background:#FFFFFF;
    border:1px solid #E2E8F0;
    border-radius:20px;
    padding:30px;
    margin-top:30px;
    box-shadow:0px 8px 24px rgba(15,23,42,0.08);
}

.prediction-heading{
    color:#1E293B;
    font-size:28px;
    font-weight:700;
    margin-bottom:20px;
}

.success-box{
    background:#ECFDF5;
    border:1px solid #A7F3D0;
    border-left:6px solid #10B981;
    border-radius:14px;
    padding:18px 20px;
    color:#047857;
    font-size:20px;
    font-weight:600;
}

.danger-box{
    background:#FEF2F2;
    border:1px solid #FECACA;
    border-left:6px solid #EF4444;
    border-radius:14px;
    padding:18px 20px;
    color:#B91C1C;
    font-size:20px;
    font-weight:600;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="navbar">

<table width="100%">
<tr>

<td width="55%">

<h2 style="margin:0;color:#2563EB;">
🛡️ Health Insurance Prediction
</h2>

</td>

<td align="right">

<span style="color:#334155;font-size:17px;font-weight:500;">
Home
&nbsp;&nbsp;&nbsp;&nbsp;
About
&nbsp;&nbsp;&nbsp;&nbsp;
Model
&nbsp;&nbsp;&nbsp;&nbsp;
Contact
</span>

</td>

</tr>

</table>

</div>
""", unsafe_allow_html=True)

left,right=st.columns([2,1])

with left:

    st.markdown("""

<div class="hero-title">

Health Insurance Cross-Sell Prediction

</div>

<div class="hero-subtitle">

Predict whether a customer is likely to purchase vehicle insurance using an optimized Random Forest Machine Learning model.

</div>

""",unsafe_allow_html=True)

with right:

    st.info(
        """
Model

Random Forest

ROC-AUC : 0.83

Threshold : 0.25
"""
    )

st.markdown("<br>", unsafe_allow_html=True)

st.markdown("""
<div class="customer-card">

<h2 style="color:#1E293B;margin-bottom:8px;">
Customer Details
</h2>

<p style="color:#64748B;margin-bottom:30px;">
Fill in the customer information to predict whether the customer is likely to purchase vehicle insurance.
</p>

""", unsafe_allow_html=True)

st.markdown("---")

left_col, right_col = st.columns(2)

with left_col:

    gender = st.selectbox(
        "Gender",
        ["Male", "Female"],
        index=None,
        placeholder="Select Gender",
        key="gender"
    )

    age = st.number_input(
        "Age",
        min_value=18,
        max_value=100,
        value=30,
        key="age"
    )

    driving_license = st.selectbox(
        "Driving License",
        ["Yes", "No"],
        index=None,
        placeholder="Select Driving License Status",
        key="driving_license"
    )

    previously_insured = st.selectbox(
        "Previously Insured",
        ["Yes", "No"],
        index=None,
        placeholder="Select Previous Insurance Status",
        key="previously_insured"
    )

    vehicle_age = st.selectbox(
        "Vehicle Age",
        [
            "Less than 1 Year",
            "1-2 Year",
            "More than 2 Years"
        ],
        index=None,
        placeholder="Select Vehicle Age",
        key="vehicle_age"
    )

with right_col:

    region_code = st.number_input(
        "Region Code",
        min_value=0,
        value=28,
        key="region_code"
    )

    vehicle_damage = st.selectbox(
        "Vehicle Damage",
        ["Yes", "No"],
        index=None,
        placeholder="Select Vehicle Damage Status",
        key="vehicle_damage"
    )

    annual_premium = st.number_input(
        "Annual Premium",
        min_value=0,
        value=30000,
        key="annual_premium"
    )

    policy_sales_channel = st.number_input(
        "Policy Sales Channel",
        min_value=0,
        value=152,
        key="policy_sales_channel"
    )

    vintage = st.number_input(
        "Vintage",
        min_value=0,
        max_value=365,
        value=150,
        key="vintage"
    )


# add button
st.markdown("<br>", unsafe_allow_html=True)

predict = st.button(
    "Predict Insurance Purchase",
    use_container_width=True
)
# st.markdown("</div>", unsafe_allow_html=True)

if predict:

    # Load trained model and threshold
    model = joblib.load("models/random_forest_model.pkl")
    threshold = joblib.load("models/threshold.pkl")

    # Convert categorical values into model-compatible values
    gender_value = 1 if gender == "Male" else 0

    driving_license_value = 1 if driving_license == "Yes" else 0

    previously_insured_value = 1 if previously_insured == "Yes" else 0

    vehicle_damage_value = 1 if vehicle_damage == "Yes" else 0

    vehicle_age_value = {
        "Less than 1 Year": 0,
        "1-2 Year": 1,
        "More than 2 Years": 2
    }[vehicle_age]

    # Create input DataFrame
    input_data = pd.DataFrame({

        "Gender": [gender_value],
        "Age": [age],
        "Driving_License": [driving_license_value],
        "Region_Code": [region_code],
        "Previously_Insured": [previously_insured_value],
        "Vehicle_Age": [vehicle_age_value],
        "Vehicle_Damage": [vehicle_damage_value],
        "Annual_Premium": [annual_premium],
        "Policy_Sales_Channel": [policy_sales_channel],
        "Vintage": [vintage]

    })

    # Model prediction
    probability = model.predict_proba(input_data)[0][1]

    prediction = int(probability >= threshold)

    # Prediction Result
    st.markdown("<br>", unsafe_allow_html=True)

    with st.container(border=True):

        st.markdown(
            "## Prediction Result"
        )

        if prediction == 1:

            st.success(
                "Customer is likely to purchase Vehicle Insurance"
            )

        else:

            st.error(
                "Customer is NOT likely to purchase Vehicle Insurance"
            )

        st.metric(
            label="Prediction Probability",
            value=f"{probability * 100:.2f}%"
        )

# if predict:

#     model = joblib.load("models/random_forest_model.pkl")
#     threshold = joblib.load("models/threshold.pkl")

#     gender_value = 1 if gender == "Male" else 0

#     driving_license_value = 1 if driving_license == "Yes" else 0

#     previously_insured_value = 1 if previously_insured == "Yes" else 0

#     vehicle_damage_value = 1 if vehicle_damage == "Yes" else 0

#     vehicle_age_value = {
#         "Less than 1 Year": 0,
#         "1-2 Year": 1,
#         "More than 2 Years": 2
#     }[vehicle_age]

#     input_data = pd.DataFrame({

#         "Gender":[gender_value],
#         "Age":[age],
#         "Driving_License":[driving_license_value],
#         "Region_Code":[region_code],
#         "Previously_Insured":[previously_insured_value],
#         "Vehicle_Age":[vehicle_age_value],
#         "Vehicle_Damage":[vehicle_damage_value],
#         "Annual_Premium":[annual_premium],
#         "Policy_Sales_Channel":[policy_sales_channel],
#         "Vintage":[vintage]

#     })

#     probability = model.predict_proba(input_data)[0][1]

#     prediction = int(probability >= threshold)

#     st.markdown("<br>", unsafe_allow_html=True)

#     st.markdown(
#         '<div class="prediction-section">',
#         unsafe_allow_html=True
#     )

#     st.markdown(
#         '<div class="prediction-heading">Prediction Result</div>',
#         unsafe_allow_html=True
#     )

#     if prediction == 1:

#         st.markdown(
#             '<div class="success-box">'
#             '✔ Customer is likely to purchase Vehicle Insurance'
#             '</div>',
#             unsafe_allow_html=True
#         )

#     else:

#         st.markdown(
#             '<div class="danger-box">'
#             '✘ Customer is NOT likely to purchase Vehicle Insurance'
#             '</div>',
#             unsafe_allow_html=True
#         )

#     st.markdown("<br>", unsafe_allow_html=True)

#     st.metric(
#         label="Prediction Probability",
#         value=f"{probability * 100:.2f}%"
#     )

#     st.markdown("</div>", unsafe_allow_html=True)