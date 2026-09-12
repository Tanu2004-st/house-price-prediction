import streamlit as st
import pandas as pd
import joblib

# =====================================================
# LOAD MODEL
# =====================================================

model = joblib.load("house_price_model.pkl")
pipeline = joblib.load("house_price_pipeline.pkl")


# =====================================================
# PAGE SETTINGS
# =====================================================

st.set_page_config(
    page_title="DreamHome - House Price Prediction",
    page_icon="🏚️",
    layout="wide"
)


# =====================================================
# COLORFUL CSS
# =====================================================

st.markdown("""
<style>

.stApp {
    background: linear-gradient(135deg, #f4f0ff, #eef8ff, #fff2fa);
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #35218c, #693bc2, #a13d91);
}

[data-testid="stSidebar"] * {
    color: white !important;
}

/* BUTTON */
.stButton > button {
    width: 100%;
    border-radius: 15px;
    height: 55px;
    font-size: 18px;
    font-weight: bold;
    color: white;
    background: linear-gradient(90deg, #4055f5, #a62be2);
    border: none;
}

.stButton > button:hover {
    color: white;
    background: linear-gradient(90deg, #293fe8, #8c1dca);
}

/* METRIC CARDS */
[data-testid="stMetric"] {
    background-color: white;
    padding: 18px;
    border-radius: 15px;
    border: 1px solid #ddd8f2;
    box-shadow: 0px 5px 15px rgba(50,50,100,0.08);
}

[data-testid="stMetricValue"] {
    color: #5835c5;
}

/* INPUTS */
div[data-baseweb="input"] > div,
div[data-baseweb="select"] > div {
    border-radius: 12px;
    background-color: white;
}

</style>
""", unsafe_allow_html=True)


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.title("🏠 DreamHome")

    st.caption("Find Value in Every Place")

    st.divider()

    page = st.radio(
        "Navigation",
        [
            "🏠 Predict Price",
            "📊 Dataset",
            "🤖 ML Model",
            "ℹ️ About"
        ]
    )

    st.divider()

    st.info(
        "✨ Smart Prediction\n\n"
        "Random Forest Regressor\n\n"
        "Boston Housing Dataset"
    )

    st.divider()

    st.caption("❤️ Built with Python & Streamlit")


# =====================================================
# PAGE 1 - PREDICT PRICE
# =====================================================

if page == "🏠 Predict Price":

    st.title("🏠 House Price Prediction")

    st.subheader(
        "Predict the estimated price of a house using Machine Learning."
    )

    st.write(
        "Enter the house details below and click "
        "**Predict House Price**."
    )

    st.divider()

    # ---------------- PROJECT OVERVIEW ----------------

    st.subheader("📊 Project Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Dataset Records", "506")

    with col2:
        st.metric("Input Features", "13")

    with col3:
        st.metric("Trees in Forest", "100")

    with col4:
        st.metric("Test RMSE", "2.81")

    st.write("")

    # ---------------- HOUSE DETAILS ----------------

    st.subheader("📋 Enter House Details")

    st.info(
        "💡 Enter the characteristics of the house. "
        "The Random Forest model will estimate its price."
    )

    col1, col2, col3 = st.columns(3)

    # COLUMN 1
    with col1:

        st.markdown("### 🏘️ Location & Area")

        CRIM = st.number_input(
            "Crime Rate (CRIM)",
            value=0.05
        )

        ZN = st.number_input(
            "Residential Land Zone (ZN)",
            value=0.0
        )

        INDUS = st.number_input(
            "Industrial Area (INDUS)",
            value=5.0
        )

        CHAS = st.selectbox(
            "Charles River (CHAS)",
            [0, 1],
            format_func=lambda x:
            "0 - No" if x == 0 else "1 - Yes"
        )

    # COLUMN 2
    with col2:

        st.markdown("### 🏠 House Features")

        NOX = st.number_input(
            "Nitric Oxide (NOX)",
            value=0.5
        )

        RM = st.number_input(
            "Number of Rooms (RM)",
            value=6.0
        )

        AGE = st.number_input(
            "Age of House (AGE)",
            value=50.0
        )

        DIS = st.number_input(
            "Distance to Employment Centres (DIS)",
            value=4.0
        )

    # COLUMN 3
    with col3:

        st.markdown("### 🛣️ Other Features")

        RAD = st.number_input(
            "Accessibility to Highways (RAD)",
            value=5.0
        )

        TAX = st.number_input(
            "Property Tax (TAX)",
            value=300.0
        )

        PTRATIO = st.number_input(
            "Pupil-Teacher Ratio (PTRATIO)",
            value=18.0
        )

        B = st.number_input(
            "Black Population Index (B)",
            value=350.0
        )

        LSTAT = st.number_input(
            "Lower Status Population (%) (LSTAT)",
            value=10.0
        )

    st.write("")

    predict = st.button("✨ PREDICT HOUSE PRICE ✨")

    # ---------------- PREDICTION ----------------

    if predict:

        house_data = pd.DataFrame({

            "CRIM": [CRIM],
            "ZN": [ZN],
            "INDUS": [INDUS],
            "CHAS": [CHAS],
            "NOX": [NOX],
            "RM": [RM],
            "AGE": [AGE],
            "DIS": [DIS],
            "RAD": [RAD],
            "TAX": [TAX],
            "PTRATIO": [PTRATIO],
            "B": [B],
            "LSTAT": [LSTAT]

        })

        prepared_data = pipeline.transform(house_data)

        prediction = model.predict(
            prepared_data
        )[0]
        usd_to_inr = 85

        prediction_inr = prediction * 1000 * usd_to_inr 

        st.divider()

        st.subheader("🎉 Prediction Result")

        result1, result2 = st.columns(2)

        with result1:

            st.success(
                f"🏠 Predicted House Price\n\n"
                f"## ₹{prediction_inr:,.0f}"
            )

        with result2:

            st.info(
                "🤖 **Model Used**\n\n"
                "Random Forest Regressor\n\n"
                "Test RMSE: **2.8082**"
            )

        st.subheader("🧾 Entered House Details")

        st.dataframe(
            house_data,
            use_container_width=True,
            hide_index=True
        )

        st.warning(
            "💡 This is an estimated price generated by "
            "the Machine Learning model. Actual market "
            "prices may vary."
        )


# =====================================================
# PAGE 2 - DATASET
# =====================================================

elif page == "📊 Dataset":

    st.title("📊 Boston Housing Dataset")

    st.write(
        "This project uses the Boston Housing Dataset "
        "for house price prediction."
    )

    st.divider()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total Records", "506")

    with col2:
        st.metric("Input Features", "13")

    with col3:
        st.metric("Target", "MEDV")

    st.divider()

    st.subheader("📋 Dataset Features")

    features = pd.DataFrame({
        "Feature": [
            "CRIM",
            "ZN",
            "INDUS",
            "CHAS",
            "NOX",
            "RM",
            "AGE",
            "DIS",
            "RAD",
            "TAX",
            "PTRATIO",
            "B",
            "LSTAT"
        ],

        "Meaning": [
            "Crime rate",
            "Residential land zone",
            "Industrial area",
            "Charles River proximity",
            "Nitric oxide concentration",
            "Average number of rooms",
            "Age of houses",
            "Distance to employment centres",
            "Highway accessibility",
            "Property tax",
            "Pupil-teacher ratio",
            "Black population index",
            "Lower status population"
        ]
    })

    st.dataframe(
        features,
        use_container_width=True,
        hide_index=True
    )

    st.info(
        "🎯 MEDV is the target variable representing "
        "the median value of owner-occupied homes."
    )


# =====================================================
# PAGE 3 - ML MODEL
# =====================================================

elif page == "🤖 ML Model":

    st.title("🤖 Machine Learning Model")

    st.subheader("Random Forest Regressor")

    st.write(
        "The final prediction is generated using a "
        "Random Forest Regressor."
    )

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.success("🌳 Random Forest")

        st.write(
            "Random Forest combines multiple decision "
            "trees to make a more accurate prediction."
        )

        st.write("**Number of Trees:** 100")

        st.write("**Training Data:** 80%")

        st.write("**Testing Data:** 20%")

    with col2:

        st.info("📈 Model Performance")

        st.metric(
            "Final Test RMSE",
            "2.8082"
        )

        st.metric(
            "Cross Validation RMSE",
            "3.6657"
        )

        st.metric(
            "Standard Deviation",
            "0.9711"
        )

    st.divider()

    st.subheader("🔄 Machine Learning Process")

    st.write("1️⃣ Load Dataset")

    st.write("2️⃣ Split Training and Testing Data")

    st.write("3️⃣ Handle Missing Values")

    st.write("4️⃣ Preprocess Data")

    st.write("5️⃣ Train Machine Learning Models")

    st.write("6️⃣ Compare Models")

    st.write("7️⃣ Select Random Forest")

    st.write("8️⃣ Test Final Model")

    st.write("9️⃣ Predict House Price")


# =====================================================
# PAGE 4 - ABOUT
# =====================================================

elif page == "ℹ️ About":

    st.title("ℹ️ About This Project")

    st.subheader("🏠 House Price Prediction")

    st.write(
        """
        This is a Machine Learning based House Price Prediction
        project.

        The purpose of this project is to predict the estimated
        value of a house using different characteristics such as
        number of rooms, crime rate, property tax, age of house,
        accessibility to highways and other features.

        The project uses the Boston Housing Dataset and a
        Random Forest Regressor for prediction.
        """
    )

    st.divider()

    st.subheader("🛠️ Technologies Used")

    tech1, tech2, tech3, tech4 = st.columns(4)

    with tech1:
        st.info("🐍 Python")

    with tech2:
        st.info("🐼 Pandas")

    with tech3:
        st.info("🤖 Scikit-learn")

    with tech4:
        st.info("🎈 Streamlit")

    st.divider()

    st.subheader("📌 Project Result")

    st.success(
        "The final Random Forest model achieved a "
        "Test RMSE of 2.8082."
    )

    st.write(
        "The Streamlit interface allows a user without "
        "programming knowledge to enter house details and "
        "get a predicted price."
    )