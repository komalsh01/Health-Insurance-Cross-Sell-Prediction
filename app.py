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

html{
    scroll-behavior:smooth;
}

.block-container{
    padding-top:6.5rem;
    padding-left:4rem;
    padding-right:4rem;
}

/* Navigation Bar */

.navbar{

    position:fixed;
    top:0;
    left:0;
    right:0;
    width:100%;

    background:white;

    border-radius:0;

    padding:14px 4rem;

    border-bottom:1px solid #E2E8F0;

    box-shadow:0px 4px 18px rgba(15,23,42,0.08);

    z-index:9999;

}

.navbar h2{
    font-size:24px !important;
}

.navbar-links a{
    color:#334155 !important;
    font-size:16px;
    font-weight:500;
    text-decoration:none;
    margin-left:26px;
}

.navbar-links a:hover{
    color:#2563EB !important;
}

#top, #model-section, #about-section, #contact-section{
    scroll-margin-top:110px;
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

div[data-testid="stVerticalBlockBorderWrapper"]{

    background:white !important;

    padding:20px 35px 35px 35px !important;

    border-radius:20px !important;

    border:1px solid #E2E8F0 !important;

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
    padding:28px 32px;
    margin:30px auto 0 auto;
    max-width:560px;
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

.probability-box{
    background:#EFF6FF;
    border:1px solid #BFDBFE;
    border-radius:14px;
    padding:20px;
    margin-top:18px;
    text-align:center;
}

.probability-label{
    color:#475569 !important;
    font-size:16px;
    font-weight:600;
}

.probability-value{
    color:#2563EB !important;
    font-size:34px;
    font-weight:700;
    margin-top:6px;
}

/* Footer */

.app-footer{
    background:#1E293B;
    border-radius:20px;
    padding:36px 45px;
    margin-top:45px;
}

.app-footer h4{
    color:#FFFFFF !important;
    font-size:18px;
    font-weight:700;
    margin-bottom:10px;
}

.app-footer p{
    color:#CBD5E1 !important;
    font-size:14px;
    line-height:1.7;
    margin-bottom:4px;
}

.app-footer a{
    color:#93C5FD !important;
    text-decoration:none;
    font-size:14px;
}

.app-footer a:hover{
    text-decoration:underline;
}

.app-footer .footer-bottom{
    color:#64748B !important;
    font-size:13px;
    text-align:center;
    margin-top:28px;
    padding-top:18px;
    border-top:1px solid #334155;
}

</style>
""", unsafe_allow_html=True)

st.markdown("""
<div id="top"></div>

<div class="navbar">

<table width="100%">
<tr>

<td width="55%">

<h2 style="margin:0;color:#2563EB;">
🛡️ Health Insurance Prediction
</h2>

</td>

<td align="right">

<span class="navbar-links">
<a href="#top">Home</a>
<a href="#about-section">About</a>
<a href="#model-section">Model</a>
<a href="#contact-section">Contact</a>
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

    st.markdown('<div id="model-section"></div>', unsafe_allow_html=True)

    st.info(
        """
Model

Random Forest

ROC-AUC : 0.83

Threshold : 0.25
"""
    )

st.markdown("<br>", unsafe_allow_html=True)

with st.container(border=True):

    st.markdown("""
<h2 style="color:#1E293B;margin-bottom:8px;">
Customer Details
</h2>

<p style="color:#64748B;margin-bottom:20px;">
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

    st.markdown("<br>", unsafe_allow_html=True)

    predict = st.button(
        "Predict Insurance Purchase",
        use_container_width=True,
        key="predict_button"
    )


if predict:

    try:

        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "models", "random_forest_model.pkl")
        threshold_path = os.path.join(BASE_DIR, "models", "threshold.pkl")

        if not os.path.exists(model_path):
            st.error(f"Model file not found at: {model_path}")
            st.stop()

        if not os.path.exists(threshold_path):
            st.error(f"Threshold file not found at: {threshold_path}")
            st.stop()

        # Load model and threshold
        model = joblib.load(model_path)
        threshold = joblib.load(threshold_path)


        # Check whether all required fields are selected
        if (
            gender is None
            or driving_license is None
            or previously_insured is None
            or vehicle_age is None
            or vehicle_damage is None
        ):

            st.warning(
                "Please fill in all customer details before making a prediction."
            )

        else:

            # Convert categorical values
            gender_value = 1 if gender == "Male" else 0

            driving_license_value = (
                1 if driving_license == "Yes" else 0
            )

            previously_insured_value = (
                1 if previously_insured == "Yes" else 0
            )

            vehicle_damage_value = (
                1 if vehicle_damage == "Yes" else 0
            )

            vehicle_age_value = {
                "Less than 1 Year": 0,
                "1-2 Year": 1,
                "More than 2 Years": 2
            }[vehicle_age]


            # Create input dataframe
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


            # Prediction probability
            probability = model.predict_proba(
                input_data
            )[0][1]


            # Final prediction
            prediction = int(
                probability >= threshold
            )

            # Prediction Result Card

            if prediction == 1:
                result_box_html = (
                    '<div class="success-box">'
                    "Customer is likely to purchase Vehicle Insurance"
                    "</div>"
                )
            else:
                result_box_html = (
                    '<div class="danger-box">'
                    "Customer is NOT likely to purchase Vehicle Insurance"
                    "</div>"
                )

            prediction_card_html = f"""<div class="prediction-section">
<div class="prediction-heading">Prediction Result</div>
{result_box_html}
<div class="probability-box">
<div class="probability-label">Prediction Probability</div>
<div class="probability-value">{probability * 100:.2f}%</div>
</div>
</div>"""

            st.markdown(prediction_card_html, unsafe_allow_html=True)


    except Exception as e:

        st.error(
            f"Prediction Error: {e}"
        )

# FOOTER 
st.markdown("""
<div class="app-footer">

<div id="about-section">
<h4>About this project</h4>
<p>This app predicts whether an existing health insurance customer is
likely to purchase vehicle insurance, using a Random Forest model
trained on customer demographics, vehicle details, and policy
information. Built to help insurance providers target the right
customers and plan outreach more efficiently.</p>
</div>

<br>

<div id="contact-section">
<h4>Contact</h4>
<p>Email: <a href="mailto:your.email@example.com">your.email@example.com</a></p>
<p>GitHub: <a href="https://github.com/your-username" target="_blank">github.com/your-username</a></p>
<p>LinkedIn: <a href="https://linkedin.com/in/your-profile" target="_blank">linkedin.com/in/your-profile</a></p>
</div>

<div class="footer-bottom">
Built with Streamlit · Data science portfolio project
</div>

</div>
""", unsafe_allow_html=True)