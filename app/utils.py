import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import streamlit as st


def plot_correlation(df):
 
    for col in df.select_dtypes(include='object').columns:
        try:
            df[col] = pd.to_datetime(df[col])
            df[col + "_timestamp"] = df[col].astype('int64') // 1_000_000_000  
        except:
            continue 
  
    numeric_df = df.select_dtypes(include='number')
    if numeric_df.empty:
        st.warning("No numeric columns available for correlation plot.")
        return

    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

def plot_predictions(y_true, y_pred, title="Predicted vs Actual"):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred, alpha=0.7)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    st.pyplot(fig)

def show_formula(model_name):
    formulas = {
        "Linear Regression": "y = β0 + β1*x1 + β2*x2 + ... + βn*xn",
        "Decision Tree": "Decision Trees split data based on feature thresholds to minimize impurity (Gini/Entropy).",
        "Random Forest": "Random Forest averages multiple decision trees trained on random subsets of data/features."
    }
    st.info(formulas.get(model_name, "Formula not available"))


def generate_notes(model_name, metrics):
    st.subheader("📓 Notes about this model")
    st.write(f"Model used: **{model_name}**")
    for key, value in metrics.items():
        st.write(f"- **{key}:** {value:.4f}")
