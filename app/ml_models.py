from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score


def train_model(df, target_column, model_type="Linear Regression", **params):
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

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
