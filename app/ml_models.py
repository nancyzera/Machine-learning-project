from sklearn.model_selection import train_test_split, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib, os
import numpy as np

MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)


def train_model(df, target_column, model_type="Linear Regression", **params):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if model_type == "Linear Regression":
        model = LinearRegression(**params)
    elif model_type == "Decision Tree":
        model = DecisionTreeRegressor(**params)
    elif model_type == "Random Forest":
        model = RandomForestRegressor(**params)
    else:
        raise ValueError("Unsupported model type")

    model.fit(X_train, y_train)

    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    metrics = {
        "Train R2": r2_score(y_train, y_train_pred),
        "Test R2": r2_score(y_test, y_test_pred),
        "Train MSE": mean_squared_error(y_train, y_train_pred),
        "Test MSE": mean_squared_error(y_test, y_test_pred),
    }

    return model, metrics, (X_train, X_test, y_train, y_test, y_train_pred, y_test_pred)

def save_model(model, name):
    path = os.path.join(MODEL_DIR, f"{name}.pkl")
    joblib.dump(model, path)
    return path


def get_feature_importance(model, X):
    if hasattr(model, "feature_importances_"):
        return dict(zip(X.columns, model.feature_importances_))
    elif hasattr(model, "coef_"):
        return dict(zip(X.columns, model.coef_))
    return None


def compute_learning_curve(model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(model, X, y, cv=5, scoring="r2")
    return train_sizes, np.mean(train_scores, axis=1), np.mean(test_scores, axis=1)

def check_overfitting(metrics):
    if metrics["Train R2"] - metrics["Test R2"] > 0.2:
        return "⚠️ Model is OVERFITTING"
    return "✅ Model generalizes well"


def auto_train(df, target_column):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(),
        "Random Forest": RandomForestRegressor()
    }

    X = df.drop(columns=[target_column])
    y = df[target_column]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        results[name] = r2_score(y_test, pred)

    best_model = max(results, key=results.get)
    return best_model, results
