from sklearn.metrics import mean_squared_error, r2_score

def evaluate(model, X_train, y_train, X_test, y_test):
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    train_r2 = r2_score(y_train, train_pred)
    test_r2 = r2_score(y_test, test_pred)

    gap = train_r2 - test_r2

    return train_r2, test_r2, gap
