import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.tree import DecisionTreeRegressor

# Assuming SoftDecisionTreeRegressor is defined as per your provided code
from SoftDecisionTreeRegressor import SoftDecisionTreeRegressor

import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
import numpy as np
from scipy import stats
from sklearn.feature_selection import SelectKBest, f_regression


path_to_datasets = r"C:\Users\Noa\Documents\Studies\ML_exercises\Exercise1\datasets\regression"



def compare_soft_and_regular_regression(X, y, max_depth=None, min_samples_leaf=1, min_samples_split=2,
                                        alpha=0.1, n_samples=100, random_state=None):
    """
    Compare performance of a regular DecisionTreeRegressor and a SoftDecisionTreeRegressor.
    This includes MSE and R² for train and test sets.
    """
    X = pd.DataFrame(X)
    y = pd.Series(y)

    # Initialize regressors
    reg_regular = DecisionTreeRegressor(random_state=random_state, max_depth=max_depth,
                                        min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)

    reg_soft = SoftDecisionTreeRegressor(alpha=alpha, n_samples=n_samples, random_state=random_state,
                                         max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                         min_samples_split=min_samples_split)

    print('Regular DecisionTreeRegressor:')
    evaluate_regressor(reg_regular, X, y)

    print('\nSoftDecisionTreeRegressor:')
    evaluate_regressor(reg_soft, X, y)

def evaluate_regressor(reg, X, y):
    """
    Evaluate regressor using cross-validation and print MSE and R² scores.
    """
    cv_results = cross_validate(reg, X, y, cv=5, return_train_score=True,
                                scoring=('neg_mean_squared_error', 'r2'))

    # Print Train and Test Scores
    print("Test MSE (CV):", -cv_results['test_neg_mean_squared_error'].mean())
    print("Train MSE (CV):", -cv_results['train_neg_mean_squared_error'].mean())
    print("Test R² (CV):", cv_results['test_r2'].mean())
    print("Train R² (CV):", cv_results['train_r2'].mean())

    # Predict all data using cross-validation
    preds = cross_val_predict(reg, X, y, cv=5)

    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)

    print("MSE on full dataset (CV):", mse)
    print("R² on full dataset (CV):", r2)


def tune_soft_decision_tree(X, y, max_depth_range, alpha_range, n_samples_range, min_samples_leaf_range, min_samples_split_range, cv=5):
    param_grid = {
        'max_depth': max_depth_range,
        'alpha': alpha_range,
        'n_samples': n_samples_range,
        'min_samples_leaf': min_samples_leaf_range,
        'min_samples_split': min_samples_split_range 
    }
    
    model = SoftDecisionTreeRegressor(alpha=0.1, n_samples=100)
    
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring='neg_mean_squared_error',
        cv=cv,
        n_jobs=-1
    )
    
    grid_search.fit(X, y)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    print("Best Parameters Found:")
    print(best_params)
    
    preds = grid_search.predict(X)
    mse = mean_squared_error(y, preds)
    r2 = r2_score(y, preds)
    
    print(f"Best Model MSE: {mse:.2f}")
    print(f"Best Model R²: {r2:.4f}")
    
    return best_model, best_params, grid_search




if __name__ == '__main__':
    
    # # 1 - Wine Quality Dataset - V
    # data = pd.read_csv(f'{path_to_datasets}\\WineQuality_dataset.csv', delimiter=',')
    # X, y = data.drop(columns=['quality']), data['quality']
    # print("Wine Quality Dataset:")
    # compare_soft_and_regular_regression(X, y, max_depth=10, min_samples_leaf=5, min_samples_split=4)
    # print("\n")


    # # 2 - Bike Sharing Dataset - V
    # data = pd.read_csv(f'{path_to_datasets}\\BikeSharingDatasetHourly_dataset.csv')
    # print(data)
    # X, y = data.drop(columns=['cnt', 'dteday']), data['cnt']
    # print("Bike Sharing Dataset:")
    # compare_soft_and_regular_regression(X, y, max_depth=15, n_samples=200, alpha=0.01, )
    # print("\n")


    # # Define parameter ranges for tuning
    # max_depth_range = [5, 10, 15]
    # alpha_range = [0.01, 0.05, 0.1]
    # n_samples_range = [100, 200]

    # best_model, best_params, grid_search = tune_soft_decision_tree(X, y, max_depth_range, alpha_range, n_samples_range)


    # results = pd.DataFrame(grid_search.cv_results_)

    # Best Parameters Found:
    # {'alpha': 0.01, 'max_depth': 15, 'n_samples': 200}

    # plt.figure(figsize=(10, 6))
    # plt.plot(results['param_alpha'], -results['mean_test_score'], marker='o')
    # plt.xlabel('Alpha')
    # plt.ylabel('Negative MSE')
    # plt.title('Hyperparameter Tuning for Soft Decision Tree')
    # plt.show()



    # # 3 - Student Performance Dataset - V
    # data = pd.read_csv(f'{path_to_datasets}\\StudentsPerformance_dataset_encoding.csv', delimiter=',')
    # data.columns = data.columns.str.strip()
    # print(data)
    # X, y = data.drop(columns=['math score']), data['math score']
    # print("Student Performance Dataset:")
    # compare_soft_and_regular_regression(X, y, max_depth=10, min_samples_leaf=5, min_samples_split=4)
    # print("\n")


    # # 4 - Online News Popularity Dataset - X
    # data = pd.read_csv(f'{path_to_datasets}\\OnlineNewsPopularity_dataset.csv')
    # sampled_data = data.sample(n=10000, random_state=42)
    # sampled_data.columns = sampled_data.columns.str.strip()
    # selector = SelectKBest(score_func=f_regression, k=40)  # Select top 30 features
    # X, y = sampled_data.drop(columns=['shares', 'url']), sampled_data['shares']
    # # Clip target values at 1st and 99th percentile
    # y = np.clip(y, np.percentile(y, 2), np.percentile(y, 98))
    # X_selected = selector.fit_transform(X, y)
    # X = pd.DataFrame(X_selected, columns=X.columns[selector.get_support()])
    # print("Online News Popularity Dataset:")
    # compare_soft_and_regular_regression(X, y, alpha=0.05, n_samples=300, max_depth=16, min_samples_leaf=25)
    # print("\n")


    # # 5 - Superconductivity Dataset - ?
    # data = pd.read_csv(f'{path_to_datasets}\\Superconductivity_dataset.csv')
    # data.columns = data.columns.str.strip()
    # X, y = data.drop(columns=['critical_temp']), data['critical_temp']
    # print("Superconductivity Dataset:")
    # compare_soft_and_regular_regression(X, y, max_depth=10, n_samples=300, alpha=0.03)
    # print("\n")

    # # Define parameter ranges for tuning
    # max_depth_range = [8, 10, 12]
    # alpha_range = [0.005, 0.01, 0.03]
    # n_samples_range = [100, 200, 300]
    

    # best_model, best_params, grid_search = tune_soft_decision_tree(X, y, max_depth_range, alpha_range, n_samples_range)


    # results = pd.DataFrame(grid_search.cv_results_)

    # 6 - Adult Income Dataset - V
    data = pd.read_csv(f'{path_to_datasets}\\AdultIncome_dataset_encoding.csv')
    data.columns = data.columns.str.strip()
    X, y = data.drop(columns=['income']), data['income']
    print("Adult Income Dataset:")
    compare_soft_and_regular_regression(X, y, max_depth=10, min_samples_leaf=5, min_samples_split=4, alpha=0.01)
    print("\n")

    # # Define parameter ranges for tuning
    # max_depth_range = [10, 12]
    # alpha_range = [0.01, 0.05]
    # n_samples_range = [100, 200]
    

    # best_model, best_params, grid_search = tune_soft_decision_tree(X, y, max_depth_range, alpha_range, n_samples_range)

    # Best Parameters Found:
    # {'alpha': 0.01, 'max_depth': 10, 'n_samples': 100}
    # Best Model MSE: 0.09
    # Best Model R²: 0.5036

    # # 7 - Student Performance Dataset - V
    # data = pd.read_csv(f'{path_to_datasets}\\MergedStudentPerformance_dataset_encoding.csv', delimiter=',')
    # data.columns = data.columns.str.strip()
    # print(data)
    # X, y = data.drop(columns=['G3']), data['G3']
    # print("Merged Student Performance Dataset:")
    # compare_soft_and_regular_regression(X, y, max_depth=10, min_samples_leaf=5, min_samples_split=4)
    # print("\n")


    # # 8 - California Housing Prices Dataset - V
    # data = pd.read_csv(f'{path_to_datasets}\\CaliforniaHousingPrices_dataset_encoding.csv', delimiter=',')
    # data.columns = data.columns.str.strip()
    # X, y = data.drop(columns=['median_house_value']), data['median_house_value']
    # print("California Housing Prices Dataset:")
    # compare_soft_and_regular_regression(X, y, max_depth=10, min_samples_leaf=5, min_samples_split=4)
    # print("\n")

    # # Define parameter ranges for tuning
    # max_depth_range = [5, 10, 15]
    # alpha_range = [0.001, 0.005, 0.01, 0.05]
    # n_samples_range = [100, 200, 300]

    # best_model, best_params, grid_search = tune_soft_decision_tree(X, y, max_depth_range, alpha_range, n_samples_range)


    # # 9 - Parkinson's Telemonitoring Dataset - ?
    # data = pd.read_csv(f'{path_to_datasets}\\ParkinsonsTelemonitoring_dataset.csv', delimiter=',')
    # data.columns = data.columns.str.strip()
    # X, y = data.drop(columns=['motor_UPDRS', 'subject#', 'test_time', 'sex', 'total_UPDRS']), data['motor_UPDRS']
    # print("Parkinson's Telemonitoring Dataset:")
    # compare_soft_and_regular_regression(X, y, max_depth=7, min_samples_leaf=10, min_samples_split=50, alpha=0.05, n_samples=400)
    # print("\n")