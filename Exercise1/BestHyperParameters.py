from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor


def print_best_decision_tree_classifier(X, Y):
    dt = DecisionTreeClassifier()

    # Hyperparameters grid
    param_grid = {
        'max_depth': [3, 5, 10, 20],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['auto', 'sqrt', 'log2'],
        'criterion': ['gini', 'entropy'],
    }

    # Grid Search
    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5, scoring='accuracy')
    grid_search.fit(X, Y)

    # Best parameters
    print(grid_search.best_params_)


def print_best_decision_tree_regressor(X, Y):
    dt = DecisionTreeRegressor()

    param_grid = {
        'max_depth': [3, 5, 10, 20],
        'min_samples_split': [2, 10, 20],
        'min_samples_leaf': [1, 5, 10],
        'max_features': ['sqrt', 'log2', None],  # None is valid for regressor
        'criterion': ['squared_error', 'friedman_mse', 'absolute_error'],  # Correct criteria for regression
    }

    grid_search = GridSearchCV(estimator=dt, param_grid=param_grid, cv=5,
                               scoring='neg_mean_squared_error')  # Use 'neg_mean_squared_error' for regression
    grid_search.fit(X, Y)

    print(grid_search.best_params_)
