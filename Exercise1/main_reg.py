import pandas as pd
from sklearn.model_selection import cross_validate
from sklearn.tree import DecisionTreeRegressor

from Exercise1.BestHyperParameters import print_best_decision_tree_regressor
from Exercise1.DecisionTrees.SoftDecisionTreeRegressor import SoftDecisionTreeRegressor
from Exercise1.SensitivityAnalysis import find_best_hyperparameters_for_soft_regressor

path_to_datasets = r"datasets\for_regression"


def compare_soft_and_regular_regression(X, y, max_depth=None, min_samples_leaf=1, min_samples_split=2,
                                        alpha=0.1, n_samples=100, random_state=None, criterion='squared_error',
                                        max_features=None):
    """
    Compare performance of a regular DecisionTreeRegressor and a SoftDecisionTreeRegressor.
    This includes MSE and R² for train and test sets.
    """
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # Initialize regressors
    reg_regular = DecisionTreeRegressor(random_state=random_state, max_depth=max_depth,
                                        min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                        criterion=criterion, max_features=max_features)

    reg_basic_soft = SoftDecisionTreeRegressor(alpha=alpha, n_samples=n_samples, random_state=random_state,
                                               max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                               min_samples_split=min_samples_split, criterion=criterion,
                                               max_features=max_features)

    reg_soft = SoftDecisionTreeRegressor(alpha=alpha, n_samples=n_samples, random_state=random_state,
                                         max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                         min_samples_split=min_samples_split, criterion=criterion,
                                         max_features=max_features)

    print('Regular DecisionTreeRegressor:')
    evaluate_regressor(reg_regular, X, y)

    print('\nBasic SoftDecisionTreeRegressor:')
    evaluate_regressor(reg_basic_soft, X, y)

    print('\nSoftDecisionTreeRegressor:')
    evaluate_regressor(reg_soft, X, y)


def evaluate_regressor(reg, X, y):
    """
    Evaluate regressor using cross-validation and print MSE and R² scores.
    """
    cv_results = cross_validate(reg, X, y, cv=5, return_train_score=True,
                                scoring=('neg_mean_squared_error', 'neg_mean_absolute_error', 'r2'))

    # Print Train and Test Scores
    print("Train MSE (CV):", -cv_results['train_neg_mean_squared_error'].mean())
    print("Train MAE (CV):", -cv_results['train_neg_mean_absolute_error'].mean())
    print("Train R² (CV):", cv_results['train_r2'].mean())

    print("Test MSE (CV):", -cv_results['test_neg_mean_squared_error'].mean())
    print("Test MAE (CV):", -cv_results['test_neg_mean_absolute_error'].mean())
    print("Test R² (CV):", cv_results['test_r2'].mean())


if __name__ == '__main__':
    # 1 - Wine Quality Dataset - V
    data = pd.read_csv(f'{path_to_datasets}\\WineQuality_dataset.csv', delimiter=',')
    X, y = data.drop(columns=['quality']), data['quality']
    print("Wine Quality Dataset:")
    print_best_decision_tree_regressor(X, y)
    find_best_hyperparameters_for_soft_regressor(X, y)
    compare_soft_and_regular_regression(X, y, max_depth=5, min_samples_leaf=10, min_samples_split=2,
                                        criterion='friedman_mse', alpha=0.03, n_samples=100)
    print("\n")

    # 2 - Bike Sharing Dataset - V
    data = pd.read_csv(f'{path_to_datasets}\\BikeSharingDatasetHourly_dataset.csv')
    X, y = data.drop(columns=['cnt', 'dteday']), data['cnt']
    print("Bike Sharing Dataset:")
    print_best_decision_tree_regressor(X, y)
    find_best_hyperparameters_for_soft_regressor(X, y)
    compare_soft_and_regular_regression(X, y, max_depth=20, min_samples_leaf=1, min_samples_split=2,
                                        criterion='squared_error', alpha=0.01, n_samples=50)
    print("\n")

    # 3 - Student Performance Dataset - V
    data = pd.read_csv(f'{path_to_datasets}\\StudentsPerformance_dataset_encoding.csv', delimiter=',')
    data.columns = data.columns.str.strip()
    X, y = data.drop(columns=['math score']), data['math score']
    print("Student Performance Dataset:")
    print_best_decision_tree_regressor(X, y)
    find_best_hyperparameters_for_soft_regressor(X, y)
    compare_soft_and_regular_regression(X, y, max_depth=10, min_samples_leaf=10,
                                        min_samples_split=2, criterion='squared_error', alpha=0.09, n_samples=50)
    print("\n")

    # 4 - Student Performance Dataset - V
    data = pd.read_csv(f'{path_to_datasets}\\MergedStudentPerformance_dataset_encoding.csv', delimiter=',')
    data.columns = data.columns.str.strip()
    X, y = data.drop(columns=['G3']), data['G3']
    print("Merged Student Performance Dataset:")
    print_best_decision_tree_regressor(X, y)
    find_best_hyperparameters_for_soft_regressor(X, y)
    compare_soft_and_regular_regression(X, y, max_depth=10, min_samples_leaf=5, min_samples_split=4, alpha=0.07,
                                        n_samples=150)
    print("\n")

    # 5 - California Housing Prices Dataset - V
    data = pd.read_csv(f'{path_to_datasets}\\CaliforniaHousingPrices_dataset_encoding.csv', delimiter=',')
    data.columns = data.columns.str.strip()
    X, y = data.drop(columns=['median_house_value']), data['median_house_value']
    print("California Housing Prices Dataset:")
    print_best_decision_tree_regressor(X, y)
    find_best_hyperparameters_for_soft_regressor(X, y)
    compare_soft_and_regular_regression(X, y, max_depth=3, min_samples_leaf=2, min_samples_split=2,
                                        criterion='squared_error', alpha=0.03, n_samples=150)
    print("\n")
