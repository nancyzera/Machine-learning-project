import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

# Negative Binomial
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial

# ================= PAGE CONFIG =================
st.set_page_config(page_title=" Intelligent ML Learning & Analysis Platform", layout="wide")
st.title(" Intelligent Machine Learning & Analysis Platform")

# ================= MODEL SAVE FOLDER =================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ================= SIDEBAR =================
st.sidebar.header("⚙️ Controls")
uploaded_file1 = st.sidebar.file_uploader("Upload Dataset 1 (CSV)", type="csv")
uploaded_file2 = st.sidebar.file_uploader("Upload Dataset 2 (CSV, optional)", type="csv")

model_option = st.sidebar.selectbox(
    "Choose Model",
    ["Linear Regression", "Logistic Regression", "Decision Tree",
     "Random Forest", "KNN", "Negative Binomial"]
)

target_column = st.sidebar.text_input("Target Column (for prediction)", "")
run_button = st.sidebar.button("🚀 Run Process")
forecast_days = st.sidebar.slider("Forecast Simulation Days", 1, 30, 7)
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20)

# ================= SAFE CSV LOADER =================
def load_csv(file):
    try:
        return pd.read_csv(file)
    except UnicodeDecodeError:
        return pd.read_csv(file, encoding="latin1")

df1 = df2 = None

# ================= LOAD DATASETS =================
if uploaded_file1:
    df1 = load_csv(uploaded_file1)
    df1 = df1.fillna(df1.mean(numeric_only=True))
    st.subheader(" Dataset 1 Preview")
    st.dataframe(df1.head())
    st.write("Missing values:", df1.isnull().sum())

if uploaded_file2:
    df2 = load_csv(uploaded_file2)
    df2 = df2.fillna(df2.mean(numeric_only=True))
    st.subheader(" Dataset 2 Preview")
    st.dataframe(df2.head())
    st.write("Missing values:", df2.isnull().sum())

# ================= FEATURE SELECTION =================
selected_features = []
if df1 is not None:
    selected_features = st.sidebar.multiselect(
        "Select Features",
        df1.columns.tolist()
    )

# ================= DATASET COMPARISON =================
if df1 is not None and df2 is not None:
    st.subheader(" Dataset Comparison")
    comparison = pd.DataFrame({
        "Dataset1_mean": df1.mean(numeric_only=True),
        "Dataset2_mean": df2.mean(numeric_only=True),
        "Difference": df1.mean(numeric_only=True) - df2.mean(numeric_only=True)
    })
    st.dataframe(comparison)

# ================= RUN ML PIPELINE =================
if run_button:
    if df1 is None:
        st.error(" Upload Dataset 1 first!")
    elif target_column not in df1.columns:
        st.error(" Target column not found!")
    else:

        # Features & Target
        X = df1[selected_features] if selected_features else df1.drop(columns=[target_column])
        y = df1[target_column]

        # Dataset description
        st.subheader(" Dataset Description")
        st.write(df1.describe())
        st.write(f"Features used: {X.shape[1]}")

        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size / 100, random_state=42
        )

        # ================= MODEL SELECTION =================
        st.subheader(f" Training Model: {model_option}")

        if model_option == "Linear Regression":
            model = LinearRegression()
            formula_text = r"y = b_0 + b_1x_1 + b_2x_2 + ... + b_nx_n"
            description_text = "Predicts continuous values using linear relationships."

        elif model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
            formula_text = r"p = \frac{1}{1 + e^{-(b_0 + b_1x_1 + ... + b_nx_n)}}"
            description_text = "Predicts probability of binary outcomes."

        elif model_option == "Decision Tree":
            model = DecisionTreeRegressor()
            formula_text = "Prediction based on decision rules"
            description_text = "Splits data into branches to predict values."

        elif model_option == "Random Forest":
            model = RandomForestRegressor()
            formula_text = "Average of many decision trees"
            description_text = "Ensemble model for higher accuracy."

        elif model_option == "KNN":
            model = KNeighborsRegressor()
            formula_text = "Average of nearest neighbors"
            description_text = "Predicts using closest data points."

        elif model_option == "Negative Binomial":
            X_train_nb = sm.add_constant(X_train)
            X_test_nb = sm.add_constant(X_test)
            model = sm.GLM(y_train, X_train_nb, family=NegativeBinomial()).fit()
            y_pred_train = model.predict(X_train_nb)
            y_pred_test = model.predict(X_test_nb)

            formula_text = r"\log(E(Y)) = b_0 + b_1x_1 + ... + b_nx_n"
            description_text = "Used for count data with over-dispersion."

        # ================= TRAIN NORMAL MODELS =================
        if model_option != "Negative Binomial":
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

        st.success(" Model Trained Successfully!")

        # ================= MODEL THEORY =================
        st.subheader(" Model Architecture & Mathematical Representation")
        st.write(description_text)
        st.latex(formula_text)

        # ================= EVALUATION =================
        st.subheader("Model Evaluation & Overfitting Check")

        train_score = r2_score(y_train, y_pred_train)
        test_score = r2_score(y_test, y_pred_test)

        st.write(f"Training R² Score: {train_score:.4f}")
        st.write(f"Testing R² Score: {test_score:.4f}")

        if train_score > test_score + 0.1:
            st.warning(" Possible Overfitting Detected!")

        mse_test = mean_squared_error(y_test, y_pred_test)
        st.write(f"Testing MSE: {mse_test:.4f}")

        # ================= PLOTS =================
        st.subheader("Predicted vs Actual")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred_test)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

        st.subheader("Residual Distribution")
        residuals = y_test - y_pred_test
        fig2, ax2 = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax2)
        st.pyplot(fig2)

        # ================= LEARNING CURVE =================
        st.subheader(" Learning Curves (Model Stability)")
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)

        curve_df = pd.DataFrame({
            "Train Score": np.mean(train_scores, axis=1),
            "Test Score": np.mean(test_scores, axis=1)
        })
        st.line_chart(curve_df)

        # ================= FORECAST =================
        st.subheader(f" {forecast_days}-Day Forecast Simulation")
        last_input = X.tail(1)
        forecast_results = []

        for _ in range(forecast_days):
            if model_option == "Negative Binomial":
                last_input_nb = sm.add_constant(last_input)
                pred_day = model.predict(last_input_nb)[0]
            else:
                pred_day = model.predict(last_input)[0]

            forecast_results.append(pred_day)

        st.line_chart(forecast_results)

        # ================= LOSS LANDSCAPE =================
        if model_option == "Linear Regression":
            st.subheader("Loss Landscape Visualization")
            coefs = np.linspace(model.coef_.min() - 1, model.coef_.max() + 1, 50)
            losses = []
            X_matrix = X_train.to_numpy()
            y_vector = y_train.to_numpy()

            for c in coefs:
                y_pred_temp = X_matrix[:, 0] * c + model.intercept_
                loss = ((y_vector - y_pred_temp) ** 2).mean()
                losses.append(loss)

            fig3, ax3 = plt.subplots()
            ax3.plot(coefs, losses)
            ax3.set_xlabel("Coefficient Value")
            ax3.set_ylabel("Loss (MSE)")
            st.pyplot(fig3)

        # ================= SAVE MODEL =================
        if st.button("💾 Save Model"):
            model_path = os.path.join(MODEL_DIR, f"{model_option.replace(' ','_')}.pkl")
            joblib.dump(model, model_path)
            st.success(f"Model saved at {model_path}")