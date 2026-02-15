import matplotlib.pyplot as plt

def plot_actual_vs_pred(y_test, y_pred):
    plt.scatter(y_test, y_pred)
    plt.xlabel("Actual")
    plt.ylabel("Predicted")
    plt.show()
