import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st


def plot_correlation(df):
   
    numeric_df = df.select_dtypes(include=["number"])

    if numeric_df.empty:
        st.warning("No numeric columns available for correlation.")
        return

    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)



def plot_predictions(y_true, y_pred):
    fig, ax = plt.subplots()
    ax.scatter(y_true, y_pred)
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Predicted vs Actual")
    st.pyplot(fig)



def plot_feature_importance(importances):
    if importances:
        fig, ax = plt.subplots()
        ax.barh(list(importances.keys()), list(importances.values()))
        ax.set_title("Feature Importance")
        st.pyplot(fig)
    else:
        st.warning("No feature importance data available.")



def show_formula(model):
    formulas = {
        "Linear Regression": "y = β0 + β1x1 + β2x2 + ... + ε",
        "Decision Tree": "Recursive feature splits based on impurity",
        "Random Forest": "Average of multiple decision trees"
    }
    st.code(formulas.get(model, "Unknown model"))



def generate_notes(model, metrics):
    note = f"""
    Model: {model}  
    Train R2: {metrics.get('Train R2', 0):.3f}  
    Test R2: {metrics.get('Test R2', 0):.3f}  

    If Train R2 >> Test R2 → Overfitting  
    If Train R2 ≈ Test R2 → Good model  
    If both low → Underfitting  
    """
    st.write(note)
