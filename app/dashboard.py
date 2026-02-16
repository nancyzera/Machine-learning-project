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

st.set_page_config(page_title="💡 Intelligent ML Learning & Analysis Platform", layout="wide")
st.title("💡 Intelligent Machine Learning Learning & Analysis Platform")

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


st.sidebar.header("Controls")
uploaded_file1 = st.sidebar.file_uploader("Upload Dataset 1 (CSV)", type="csv")
uploaded_file2 = st.sidebar.file_uploader("Upload Dataset 2 (CSV, optional)", type="csv")
model_option = st.sidebar.selectbox("Choose Model", ["Linear Regression", "Logistic Regression",
                                                     "Decision Tree", "Random Forest", "KNN"])
target_column = st.sidebar.text_input("Target Column (for prediction)", "")
selected_features = st.sidebar.multiselect("Select Features (multi-column support)", [])
run_button = st.sidebar.button("🚀 Run Process")
forecast_days = st.sidebar.slider("7-Day Forecast Simulation", 1, 30, 7)


df1 = df2 = None
if uploaded_file1:
    df1 = pd.read_csv(uploaded_file1)
    df1 = df1.fillna(df1.mean(numeric_only=True))
    st.subheader("✅ Dataset 1 Preview")
    st.dataframe(df1.head())
    st.write("Missing values per column:", df1.isnull().sum())

if uploaded_file2:
    df2 = pd.read_csv(uploaded_file2)
    df2 = df2.fillna(df2.mean(numeric_only=True))
    st.subheader("✅ Dataset 2 Preview")
    st.dataframe(df2.head())
    st.write("Missing values per column:", df2.isnull().sum())


if df1 is not None and df2 is not None:
    st.subheader("📊 Dataset Comparison")
    comparison = pd.DataFrame({
        "Dataset1_mean": df1.mean(numeric_only=True),
        "Dataset2_mean": df2.mean(numeric_only=True),
        "Difference": df1.mean(numeric_only=True) - df2.mean(numeric_only=True)
    })
    st.dataframe(comparison)

if run_button:
    if df1 is None:
        st.error("Please upload at least Dataset 1!")
    elif target_column not in df1.columns:
        st.error("Target column not found in Dataset 1!")
    else:
       
        X = df1[selected_features] if selected_features else df1.drop(columns=[target_column])
        y = df1[target_column]

     
        st.subheader("🗂 Dataset Description")
        st.write(df1.describe())
        st.write(f"Number of features used: {X.shape[1]}")

        test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42)

        st.subheader(f"🛠 Training Model: {model_option}")
        if model_option == "Linear Regression":
            model = LinearRegression()
            formula_text = "y = b0 + b1*x1 + b2*x2 + ... + bn*xn"
            description_text = ("Linear Regression predicts a continuous target variable "
                                "based on linear relationships between features and target.")
        elif model_option == "Logistic Regression":
            model = LogisticRegression(max_iter=1000)
            formula_text = "p = 1 / (1 + e^-(b0 + b1*x1 + ... + bn*xn))"
            description_text = ("Logistic Regression predicts the probability of a binary outcome "
                                "using the logistic function.")
        elif model_option == "Decision Tree":
            model = DecisionTreeRegressor()
            formula_text = "y = leaf value based on feature splits"
            description_text = ("Decision Tree splits the data based on feature thresholds to predict values.")
        elif model_option == "Random Forest":
            model = RandomForestRegressor()
            formula_text = "y = average predictions of multiple decision trees"
            description_text = ("Random Forest is an ensemble of Decision Trees to improve prediction accuracy.")
        elif model_option == "KNN":
            model = KNeighborsRegressor()
            formula_text = "y = average of k nearest neighbors"
            description_text = ("K-Nearest Neighbors predicts by averaging outcomes of closest k points.")

        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

        st.success("Model Trained Successfully!")

        st.subheader("📝 Model Architecture & Mathematical Representation")
        st.write(description_text)
        st.latex(formula_text)

      
        st.subheader("📊 Model Evaluation & Overfitting Check")
        train_score = r2_score(y_train, y_pred_train)
        test_score = r2_score(y_test, y_pred_test)
        st.write(f"Training R2 Score: {train_score:.4f}")
        st.write(f"Testing R2 Score: {test_score:.4f}")
        if train_score > test_score + 0.1:
            st.warning("⚠️ Possible Overfitting Detected!")

        mse_test = mean_squared_error(y_test, y_pred_test)
        st.write(f"Testing MSE: {mse_test:.4f}")

        st.subheader("📈 Visual Testing (Predicted vs Actual)")
        fig, ax = plt.subplots()
        ax.scatter(y_test, y_pred_test)
        ax.set_xlabel("Actual")
        ax.set_ylabel("Predicted")
        st.pyplot(fig)

        st.subheader("📉 Error Analysis (Residuals Distribution)")
        residuals = y_test - y_pred_test
        fig2, ax2 = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax2)
        ax2.set_xlabel("Residuals")
        st.pyplot(fig2)

       
        st.subheader("📊 Learning Curves (Stability Test)")
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
        st.line_chart({
            'Train Score': np.mean(train_scores, axis=1),
            'Test Score': np.mean(test_scores, axis=1)
        })

        st.subheader(f"📅 {forecast_days}-Day Forecast Simulation")
        if forecast_days > 0:
            last_input = X.tail(1)
            forecast_results = []
            for i in range(forecast_days):
                pred_day = model.predict(last_input)[0]
                forecast_results.append(pred_day)
              
            st.line_chart(forecast_results)

        if model_option == "Linear Regression":
            st.subheader("🔍 Loss Landscape (Grid Search Visualization)")
            coefs = np.linspace(model.coef_.min() - 1, model.coef_.max() + 1, 50)
            losses = []
            X_matrix = X_train.to_numpy()
            y_vector = y_train.to_numpy()
            for c in coefs:
                y_pred_temp = X_matrix[:,0] * c + model.intercept_
                loss = ((y_vector - y_pred_temp) ** 2).mean()
                losses.append(loss)
            fig3, ax3 = plt.subplots()
            ax3.plot(coefs, losses)
            ax3.set_xlabel("Coefficient Value")
            ax3.set_ylabel("Loss (MSE)")
            st.pyplot(fig3)

        if st.button("💾 Save Model"):
            model_path = os.path.join(MODEL_DIR, f"{model_option.replace(' ','_')}.pkl")
            joblib.dump(model, model_path)
            st.success(f"Model saved at {model_path}")
