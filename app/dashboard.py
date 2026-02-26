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
    ["Linear Regression", "Decision Tree",
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
        st.stop()

    if target_column not in df1.columns:
        st.error("Target column not found!")
        st.stop()

    # ================= DATA PREPROCESSING =================
    X = df1[selected_features] if selected_features else df1.drop(columns=[target_column])
    y = df1[target_column]

    # Convert categorical columns
    for col in X.columns:
        if X[col].dtype == "object":
            X[col] = pd.factorize(X[col])[0]

    X = X.fillna(X.mean())
    y = y.fillna(y.mean()).values.ravel()

    st.subheader("Dataset Summary")
    st.write(df1.describe())

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )

    # ================= MODEL SELECTION =================
    if model_option == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

    elif model_option == "Logistic Regression":
        # Convert y to binary for logistic
        y_bin = (y > np.median(y)).astype(int)
        X_train, X_test, y_train, y_test = train_test_split(X, y_bin, test_size=test_size/100, random_state=42)
        model = LogisticRegression(max_iter=1000)
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

    elif model_option == "Decision Tree":
        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

    elif model_option == "Random Forest":
        model = RandomForestRegressor()
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

    elif model_option == "KNN":
        model = KNeighborsRegressor()
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)

    # elif model_option == "Negative Binomial":
    #     X_train_nb = sm.add_constant(X_train)
    #     X_test_nb = sm.add_constant(X_test)
    #     model = sm.GLM(y_train, X_train_nb, family=NegativeBinomial()).fit()
    #     y_pred_train = model.predict(X_train_nb)
    #     y_pred_test = model.predict(X_test_nb)
elif model_option == "Negative Binomial":

    # ================= NEGATIVE BINOMIAL SAFETY CHECKS =================
    
    # 1️⃣ Ensure target is numeric
    if not np.issubdtype(y_train.dtype, np.number):
        st.error("Negative Binomial requires numeric target values.")
        st.stop()

    # 2️⃣ Ensure no negative values (NB is for count data)
    if (y_train < 0).any():
        st.error("Negative Binomial cannot be used with negative target values.")
        st.stop()

    # 3️⃣ Ensure target is not constant
    if len(np.unique(y_train)) <= 1:
        st.error("Target column must contain more than one unique value.")
        st.stop()

    # ================= NB PARAMETER INPUT =================
    
    alpha_value = st.sidebar.number_input(
        "Dispersion Parameter (alpha)",
        min_value=0.01,
        max_value=10.0,
        value=1.0,
        step=0.1
    )

    try:
        # 4️⃣ Add constant term properly
        X_train_nb = sm.add_constant(X_train, has_constant='add')
        X_test_nb = sm.add_constant(X_test, has_constant='add')

        # 5️⃣ Define GLM model
        nb_model = sm.GLM(
            y_train,
            X_train_nb,
            family=NegativeBinomial(alpha=alpha_value)
        )

        # 6️⃣ Fit model
        model = nb_model.fit()

        # 7️⃣ Predictions
        y_pred_train = model.predict(X_train_nb)
        y_pred_test = model.predict(X_test_nb)

    except Exception as e:
        st.error(f"Negative Binomial model failed: {e}")
        st.stop()
    # ================= THEORY =================
    st.subheader("Mathematical Representation & Theory")

    formulas = {
        "Linear Regression": r"y = \beta_0 + \beta_1x_1 + ... + \beta_nx_n",
        "Logistic Regression": r"P(y=1)=\frac{1}{1+e^{-(\beta_0+\beta_1x_1+...+\beta_nx_n)}}",
        "Decision Tree": r"Gini = 1 - \sum p_i^2",
        "Random Forest": r"\hat{y} = \frac{1}{T}\sum h_t(x)",
        "KNN": r"\hat{y} = \frac{1}{k}\sum y_i",
        "Negative Binomial": r"\log(E(Y)) = \beta_0 + \beta_1x_1 + ..."
    }

    theory = {
        "Linear Regression": "Predicts continuous values.",
        "Logistic Regression": "Binary classification model.",
        "Decision Tree": "Tree-based splits.",
        "Random Forest": "Multiple trees ensemble.",
        "KNN": "Neighbor-based prediction.",
        "Negative Binomial": "Count data regression."
    }

    st.write(theory[model_option])
    st.latex(formulas[model_option])

    # ================= EVALUATION =================
    st.subheader("Model Evaluation")
    train_score = r2_score(y_train, y_pred_train)
    test_score = r2_score(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)

    st.metric("Train R²", train_score)
    st.metric("Test R²", test_score)
    st.metric("Test MSE", mse_test)

    if train_score > test_score + 0.1:
        st.warning("Overfitting detected")
    elif test_score < 0.5:
        st.warning("Weak model performance")
    else:
        st.success("Model generalizes well")

    # ================= PLOT =================
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
    st.subheader("Learning Curve")
    if model_option != "Negative Binomial":
        train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5)
        curve_df = pd.DataFrame({
            "Train Score": np.mean(train_scores, axis=1),
            "Test Score": np.mean(test_scores, axis=1)
        })
        st.line_chart(curve_df)

    # ================= FORECAST =================
    st.subheader("Future Forecast Simulation")
    last_input = X.tail(1)
    future_preds = []
    for _ in range(forecast_days):
        if model_option == "Negative Binomial":
            pred = model.predict(sm.add_constant(last_input))[0]
        else:
            pred = model.predict(last_input)[0]
        future_preds.append(pred)
    st.line_chart(future_preds)

    # ================= CUSTOM INPUT =================
    st.subheader("Predict with Custom Inputs")
    user_input = {}
    for col in X.columns:
        val = st.number_input(col, value=float(X[col].mean()))
        user_input[col] = val

    if st.button("Predict Custom Input"):
        input_df = pd.DataFrame([user_input])
        if model_option == "Negative Binomial":
            input_df = sm.add_constant(input_df)
        pred = model.predict(input_df)[0]
        st.success(f"Predicted {target_column}: {pred:.4f}")

    # ================= SAVE MODEL =================
    if st.button("Save Model"):
        path = os.path.join(MODEL_DIR, f"{model_option.replace(' ','_')}.pkl")
        joblib.dump(model, path)
        st.success(f"Model saved: {path}")
