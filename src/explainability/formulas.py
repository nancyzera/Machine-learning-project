def random_forest_formula():
    return """
    Y(x) = (1/K) * Σ Tk(x)
    Where:
    K = number of trees
    Tk(x) = prediction of kth tree
    """
