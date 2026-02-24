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
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score
import joblib, os
import statsmodels.api as sm
from statsmodels.genmod.families import NegativeBinomial

# ================= PAGE CONFIG =================
st.set_page_config(page_title="AI ML Research Platform", layout="wide")
st.title("🚀 Intelligent Machine Learning Research & Analysis Platform")

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

target_column = st.sidebar.text_input("Target Column Name")
forecast_days = st.sidebar.slider("Forecast Simulation Steps", 1, 30, 7)
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
    st.subheader("📊 Dataset 1 Preview")
    st.dataframe(df1.head())

if uploaded_file2:
    df2 = load_csv(uploaded_file2)
    st.subheader("📊 Dataset 2 Preview")
    st.dataframe(df2.head())

# Feature selection
selected_features = []
if df1 is not None:
    all_cols = df1.columns.tolist()
    # Default selection: everything except target
    default_features = [c for c in all_cols if c != target_column]
    selected_features = st.sidebar.multiselect("Select Features", all_cols, default=default_features)

# ================= DATASET COMPARISON =================
if df1 is not None and df2 is not None:
    st.subheader("🔄 Dataset Statistical Comparison")
    # Align columns for comparison
    common_cols = df1.select_dtypes(include=[np.number]).columns.intersection(df2.select_dtypes(include=[np.number]).columns)
    if not common_cols.empty:
        comparison = pd.DataFrame({
            "Dataset1 Mean": df1[common_cols].mean(),
            "Dataset2 Mean": df2[common_cols].mean(),
            "Difference": df1[common_cols].mean() - df2[common_cols].mean()
        })
        st.dataframe(comparison)

# ================= RUN PIPELINE =================
if run_button:
    if df1 is None:
        st.error("Please upload Dataset 1 first!")
    elif target_column not in df1.columns:
        st.error(f"Target column '{target_column}' not found in dataset!")
    elif not selected_features:
        st.error("Please select at least one feature column!")
    else:
        # ================= DATA PREPROCESSING =================
        X = df1[selected_features].copy()
        y = df1[target_column].copy()

        # Convert categorical/text to numeric for features
        for col in X.columns:
            if X[col].dtype == 'object':
                X[col] = pd.factorize(X[col])[0]

        # Fill missing values
        X = X.fillna(X.median(numeric_only=True))
        
        # Target preprocessing based on model
        if model_option == "Logistic Regression":
            # Ensure target is discrete
            if y.dtype == 'float' or y.nunique() > 20:
                st.warning("Logistic Regression detected a continuous target. Converting to binary (above/below mean).")
                y = (y > y.mean()).astype(int)
            else:
                y = pd.factorize(y)[0]
        else:
            y = y.fillna(y.mean())

        y_vals = y.values.ravel()

        st.subheader("📈 Dataset Summary")
        st.write(df1.describe())

        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_vals, test_size=test_size/100, random_state=42
        )

        # ================= SELECT MODEL =================
        try:
            if model_option == "Linear Regression":
                model = LinearRegression()
            elif model_option == "Logistic Regression":
                model = LogisticRegression(max_iter=1000)
            elif model_option == "Decision Tree":
                model = DecisionTreeRegressor()
            elif model_option == "Random Forest":
                model = RandomForestRegressor()
            elif model_option == "KNN":
                model = KNeighborsRegressor()
            elif model_option == "Negative Binomial":
                X_train_nb = sm.add_constant(X_train, has_constant='add')
                X_test_nb = sm.add_constant(X_test, has_constant='add')
                model = sm.GLM(y_train, X_train_nb, family=NegativeBinomial()).fit()
                y_pred_train = model.predict(X_train_nb)
                y_pred_test = model.predict(X_test_nb)

            # Train Scikit-Learn models
            if model_option != "Negative Binomial":
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

            st.success(f"✅ {model_option} Trained Successfully")

            # ================= EVALUATION =================
            st.subheader("⚖️ Model Evaluation")
            col1, col2, col3 = st.columns(3)
            
            # Use appropriate metrics
            if model_option == "Logistic Regression":
                train_acc = accuracy_score(y_train, y_pred_train.round())
                test_acc = accuracy_score(y_test, y_pred_test.round())
                col1.metric("Train Accuracy", f"{train_acc:.2%}")
                col2.metric("Test Accuracy", f"{test_acc:.2%}")
                test_score = test_acc # for logic below
                train_score = train_acc
            else:
                train_r2 = r2_score(y_train, y_pred_train)
                test_r2 = r2_score(y_test, y_pred_test)
                mse = mean_squared_error(y_test, y_pred_test)
                col1.metric("Train R²", f"{train_r2:.4f}")
                col2.metric("Test R²", f"{test_r2:.4f}")
                col3.metric("Test MSE", f"{mse:.4f}")
                test_score = test_r2
                train_score = train_r2

            # ================= VISUALS =================
            c1, c2 = st.columns(2)
            with c1:
                st.write("**Actual vs Predicted**")
                fig, ax = plt.subplots()
                # Sort for better visualization if small sample, otherwise scatter
                ax.scatter(range(len(y_test[:100])), y_test[:100], label="Actual", alpha=0.6)
                ax.scatter(range(len(y_pred_test[:100])), y_pred_test[:100], label="Predicted", alpha=0.6)
                ax.legend()
                st.pyplot(fig)
            
            with c2:
                st.write("**Residual Distribution**")
                residuals = y_test - y_pred_test
                fig2, ax2 = plt.subplots()
                sns.histplot(residuals, kde=True, ax=ax2, color="purple")
                st.pyplot(fig2)

            # ================= CUSTOM INPUT PREDICTION =================
            st.markdown("---")
            st.subheader("🔮 Predict with Custom Inputs")
            with st.expander("Configure Input Features", expanded=True):
                user_input = {}
                cols = st.columns(3)
                for i, col_name in enumerate(selected_features):
                    with cols[i % 3]:
                        if np.issubdtype(X[col_name].dtype, np.number):
                            val = st.number_input(f"{col_name}", value=float(X[col_name].median()))
                            user_input[col_name] = val
                        else:
                            unique_vals = df1[col_name].unique().tolist()
                            val = st.selectbox(f"{col_name}", unique_vals)
                            user_input[col_name] = pd.factorize([val])[0][0]

                if st.button("Generate Prediction"):
                    input_df = pd.DataFrame([user_input])
                    if model_option == "Negative Binomial":
                        input_df_nb = sm.add_constant(input_df, has_constant='add', prepend=True)
                        # Ensure columns match training (constant first)
                        pred = model.predict(input_df_nb)[0]
                    else:
                        pred = model.predict(input_df)[0]
                    st.info(f"**Predicted {target_column}:** {pred:.4f}")

            # ================= AUTO MODEL COMPARISON =================
            st.subheader("🏁 Automatic Model Comparison")
            comp_models = {
                "Linear": LinearRegression(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "KNN": KNeighborsRegressor()
            }
            results = []
            for name, m in comp_models.items():
                m.fit(X_train, y_train)
                p = m.predict(X_test)
                results.append([name, r2_score(y_test, p), mean_squared_error(y_test, p)])
            
            compare_df = pd.DataFrame(results, columns=["Model", "R2 Score", "MSE"])
            st.table(compare_df.sort_values("R2 Score", ascending=False))

            # Store model in session state for saving
            st.session_state['current_model'] = model
            if st.button("💾 Save Trained Model"):
                path = os.path.join(MODEL_DIR, f"{model_option.replace(' ','_')}.pkl")
                joblib.dump(model, path)
                st.success(f"Model saved to {path}")

        except Exception as e:
            st.error(f"Error during execution: {e}")

else:
    st.info("Upload a dataset and click 'Run AI Pipeline' in the sidebar to begin.")
