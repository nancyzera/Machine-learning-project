import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import streamlit as st
import pandas as pd
from app import ml_models, utils, cycling_simulator



st.set_page_config(page_title="💡 ML & Cycling Dashboard", layout="wide")


page = st.sidebar.selectbox("Choose Page", ["ML System", "Cycling Simulator"])

if page == "ML System":
    st.markdown("<h1 style='color:#4B0082;'> Interactive ML System</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload CSV dataset", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.dataframe(df.head())
        st.subheader("Summary Statistics")
        st.write(df.describe())
        st.subheader("Correlation Heatmap")
        utils.plot_correlation(df)

     
        st.subheader("Select Model")
        model_choice = st.selectbox("Choose ML Model", ["Linear Regression", "Decision Tree", "Random Forest"])
        target_column = st.selectbox("Select Target Column", df.columns)

        st.subheader("Model Parameters (Optional)")
        params = {}
        if model_choice == "Decision Tree":
            params["max_depth"] = st.slider("Max Depth", 1, 20, 5)
            params["min_samples_split"] = st.slider("Min Samples Split", 2, 10, 2)
        elif model_choice == "Random Forest":
            params["n_estimators"] = st.slider("Number of Trees", 10, 200, 50)
            params["max_depth"] = st.slider("Max Depth", 1, 20, 5)
            params["min_samples_split"] = st.slider("Min Samples Split", 2, 10, 2)

        if st.button("Train Model"):
            with st.spinner("Training model..."):
                model, metrics, data_split = ml_models.train_model(df, target_column, model_choice, **params)
            st.success("Model Trained Successfully!")

          
            st.subheader("Metrics")
            st.json(metrics)

           
            X_train, X_test, y_train, y_test, y_train_pred, y_test_pred = data_split
            st.subheader("Predicted vs Actual (Test Set)")
            utils.plot_predictions(y_test, y_test_pred)

            st.subheader("Overfitting Check")
            st.write(f"Train R2: {metrics['Train R2']:.4f} | Test R2: {metrics['Test R2']:.4f}")
            st.write(f"Train MSE: {metrics['Train MSE']:.4f} | Test MSE: {metrics['Test MSE']:.4f}")

         
            st.subheader("Formulas Used")
            utils.show_formula(model_choice)

           
            st.subheader("AI Notes / Observations")
            utils.generate_notes(model_choice, metrics)

            st.info("Your ML model is ready. You can explore data, predictions, and overfitting!")

elif page == "Cycling Simulator":
    st.markdown("<h1 style='color:#006400;'>🚴 Kigali Cycling Demand Simulator</h1>", unsafe_allow_html=True)

    st.subheader("Input Baseline Trips ")
    baseline_trips = {}
    stations = ["Gikondo", "CBD", "Nyabugogo"]
    for s in stations:
        baseline_trips[s] = st.number_input(f"{s} baseline trips", value=500, step=50)

    st.subheader("Scenario Parameters")
    slope = st.slider("Road Slope (%)", 0, 20, 5)
    infra_score = st.slider("Infrastructure Score", 0, 5, 1)
    income_level = st.slider("Income Level", 0, 5, 1)
    safety_score = st.slider("Safety Score", 0, 5, 1)
    rain = st.number_input("Rainfall (mm)", 0, 100, 10)
    wind = st.number_input("Wind Speed (m/s)", 0, 20, 3)

  
    st.subheader("Pollution Data")
    co = st.number_input("CO (ppm)", 0, 100, 30)
    pm10 = st.number_input("PM10 (µg/m³)", 0, 200, 50)
    o3 = st.number_input("O3 (ppb)", 0, 300, 100)

    if st.button("Run Scenario"):
        results = cycling_simulator.calculate_demand(
            baseline_trips=baseline_trips,
            slope=slope,
            infra_score=infra_score,
            income_level=income_level,
            safety_score=safety_score,
            rain=rain,
            wind=wind,
            co=co,
            pm10=pm10,
            o3=o3
        )

        st.subheader("Simulation Results")
        for station, data in results.items():
            st.markdown(f"**{station}**: {data['Adjusted Trips']:.1f} trips "
                        f"(<span style='color:blue;'>{data['Change (%)']:.2f}% change</span>)",
                        unsafe_allow_html=True)

        st.subheader("Formula Used")
        st.code("""
Ds_t = Dsbase * exp(
    SLOPE_BETA*slope + INFRA_BETA*infra_score + 
    INCOME_BETA*income_level + SAFETY_BETA*safety_score + 
    RAIN_THETA*rain + WIND_THETA*wind + POLLUTION_THETA*PI
)
Percentage Change = ((Ds_t - Dsbase)/Dsbase) * 100
""")
        st.info("💡 You can modify any parameter to test different scenarios and observe changes in demand.")

        
        st.subheader("Visualized Results")
        df_results = pd.DataFrame(results).T.reset_index().rename(columns={"index": "Station"})
        st.bar_chart(df_results.set_index("Station")["Adjusted Trips"])
