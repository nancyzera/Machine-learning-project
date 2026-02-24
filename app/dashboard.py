# ===== IMPORTS (UNCHANGED) =====
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
import joblib, os
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial

# ================= PAGE CONFIG =================
st.set_page_config(page_title="AI ML Research Platform", layout="wide")
st.title("Intelligent Machine Learning Research & Analysis Platform")

# ================= SAVE FOLDER =================
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

# ================= SIDEBAR =================
st.sidebar.header("Controls")
uploaded_file1 = st.sidebar.file_uploader("Upload Dataset 1", type="csv")
uploaded_file2 = st.sidebar.file_uploader("Upload Dataset 2 (optional)", type="csv")

model_option = st.sidebar.selectbox(
    "Choose Model",
    ["Linear Regression", "Logistic Regression", "Decision Tree",
     "Random Forest", "KNN", "Negative Binomial"]
)

target_column = st.sidebar.text_input("Target Column")
forecast_days = st.sidebar.slider("Forecast Days", 1, 30, 7)
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20)
run_button = st.sidebar.button("Run AI Pipeline")

# ================= SAFE LOAD =================
def load_csv(file):
    try:
        return pd.read_csv(file)
    except:
        return pd.read_csv(file, encoding="latin1")

df1 = df2 = None

if uploaded_file1:
    df1 = load_csv(uploaded_file1)
    st.subheader("Dataset 1 Preview")
    st.dataframe(df1.head())

if uploaded_file2:
    df2 = load_csv(uploaded_file2)
    st.subheader("Dataset 2 Preview")
    st.dataframe(df2.head())

# Feature selection
selected_features = []
if df1 is not None:
    selected_features = st.sidebar.multiselect("Select Features", df1.columns.tolist())

# ================= DATASET COMPARISON =================
if df1 is not None and df2 is not None:
    st.subheader("Dataset Statistical Comparison")
    comparison = pd.DataFrame({
        "Dataset1 Mean": df1.mean(numeric_only=True),
        "Dataset2 Mean": df2.mean(numeric_only=True),
        "Difference": df1.mean(numeric_only=True) - df2.mean(numeric_only=True)
    })
    st.dataframe(comparison)

# ================= RUN PIPELINE =================
if run_button:
    if df1 is None:
        st.error("Upload dataset first!")
    elif target_column not in df1.columns:
        st.error("Target column not found!")
    else:

        # ================= DATA PREPROCESSING =================
        X = df1[selected_features] if selected_features else df1.drop(columns=[target_column])
        y = df1[target_column]

        # Fill missing values
        X = X.fillna(X.mean(numeric_only=True))
        y = y.fillna(y.mean())

        # Convert categorical columns safely
        X_encoded = pd.get_dummies(X, drop_first=True)

        # Save columns for prediction alignment
        TRAIN_COLUMNS = X_encoded.columns

        y = y.values.ravel()

        st.subheader("Dataset Summary")
        st.write(df1.describe())

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X_encoded, y, test_size=test_size/100, random_state=42
        )

        # ================= MODEL SELECTION =================
        if model_option == "Linear Regression":
            model = LinearRegression()

        elif model_option == "Logistic Regression":
            # Auto convert target to binary if needed
            y_train = (y_train > np.median(y_train)).astype(int)
            y_test = (y_test > np.median(y_test)).astype(int)
            model = LogisticRegression(max_iter=1000)

        elif model_option == "Decision Tree":
            model = DecisionTreeRegressor()

        elif model_option == "Random Forest":
            model = RandomForestRegressor(n_estimators=100)

        elif model_option == "KNN":
            model = KNeighborsRegressor(n_neighbors=5)

        elif model_option == "Negative Binomial":
            X_train_nb = sm.add_constant(X_train)
            X_test_nb = sm.add_constant(X_test)
            model = sm.GLM(y_train, X_train_nb, family=NegativeBinomial()).fit()
            y_pred_train = model.predict(X_train_nb)
            y_pred_test = model.predict(X_test_nb)

        # Train sklearn models
        if model_option != "Negative Binomial":
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

        st.success("Model Trained Successfully")

        # ================= EVALUATION =================
        st.subheader("Model Evaluation")
        train_score = r2_score(y_train, y_pred_train)
        test_score = r2_score(y_test, y_pred_test)
        mse_test = mean_squared_error(y_test, y_pred_test)

        st.metric("Train R²", train_score)
        st.metric("Test R²", test_score)
        st.metric("Test MSE", mse_test)

        # ================= LIVE GRAPH =================
        st.subheader("Live Prediction vs Actual")
        fig, ax = plt.subplots()
        ax.plot(y_test, label="Actual")
        ax.plot(y_pred_test, label="Predicted")
        ax.legend()
        st.pyplot(fig)

        # ================= RESIDUALS =================
        st.subheader("Residual Distribution")
        residuals = y_test - y_pred_test
        fig2, ax2 = plt.subplots()
        sns.histplot(residuals, kde=True, ax=ax2)
        st.pyplot(fig2)

        # ================= LEARNING CURVE =================
        st.subheader("Learning Curve (Bias-Variance Test)")
        if model_option != "Negative Binomial":
            train_sizes, train_scores, test_scores = learning_curve(model, X_encoded, y, cv=5)
            curve_df = pd.DataFrame({
                "Train Score": np.mean(train_scores, axis=1),
                "Test Score": np.mean(test_scores, axis=1)
            })
            st.line_chart(curve_df)
        else:
            st.warning("Learning curve not supported for Negative Binomial")

        # ================= FORECAST =================
        st.subheader("Future Forecast Simulation")
        last_input = X_encoded.tail(1)
        future_preds = []
        for _ in range(forecast_days):
            if model_option == "Negative Binomial":
                pred = model.predict(sm.add_constant(last_input))[0]
            else:
                pred = model.predict(last_input)[0]
            future_preds.append(pred)
        st.line_chart(future_preds)

        # ================= CLEAR CUSTOM INPUT FORM =================
        st.subheader("🔮 Predict with Your Own Data")

        with st.form("prediction_form"):
            user_input = {}
            for col in X.columns:
                if np.issubdtype(X[col].dtype, np.number):
                    user_input[col] = st.number_input(f"{col}", value=float(X[col].mean()))
                else:
                    user_input[col] = st.selectbox(f"{col}", X[col].unique())

            predict_btn = st.form_submit_button("Predict Now")

        if predict_btn:
            input_df = pd.DataFrame([user_input])
            input_df = pd.get_dummies(input_df)
            input_df = input_df.reindex(columns=TRAIN_COLUMNS, fill_value=0)

            if model_option == "Negative Binomial":
                input_df = sm.add_constant(input_df)

            pred = model.predict(input_df)[0]
            st.success(f"✅ Predicted {target_column}: {pred}")

        # ================= AUTO MODEL COMPARISON =================
        st.subheader("Automatic Model Comparison")
        models = {
            "Linear Regression": LinearRegression(),
            "Decision Tree": DecisionTreeRegressor(),
            "Random Forest": RandomForestRegressor(),
            "KNN": KNeighborsRegressor()
        }

        results = []
        for name, m in models.items():
            m.fit(X_train, y_train)
            pred = m.predict(X_test)
            results.append([name, r2_score(y_test, pred), mean_squared_error(y_test, pred)])

        compare_df = pd.DataFrame(results, columns=["Model", "R2 Score", "MSE"])
        st.dataframe(compare_df)

        best_model = compare_df.sort_values("R2 Score", ascending=False).iloc[0]
        st.success(f"Best Model: {best_model['Model']} (R²={best_model['R2 Score']:.4f})")

        # ================= SAVE MODEL =================
        if st.button("Save Model"):
            path = os.path.join(MODEL_DIR, f"{model_option.replace(' ','_')}.pkl")
            joblib.dump(model, path)
            st.success(f"Model saved: {path}")
