import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from Exercise1.SoftDecisionTreeClassifier import SoftDecisionTreeClassifier


def find_best_hyperparameters(X, Y):
    """
    Function to find the best hyperparameters for Soft Decision Tree using cross-validation with grid search.
    """
    # Define parameter grid for alpha and n_samples
    param_grid = {
        'alpha': np.logspace(-3, np.log10(0.9), 5),  # Testing alpha in log scale from 0.001 to 0.9
        'n_samples': [10, 50, 100, 200, 500, 1000]  # Number of samples to test
    }

    # Initialize the Soft Decision Tree classifier
    clf = SoftDecisionTreeClassifier(alpha=0.1, n_samples=100)

    # Setup Stratified K-Fold cross-validation for class imbalance handling
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Grid search with cross-validation
    grid_search = GridSearchCV(
        estimator=clf,
        param_grid=param_grid,
        cv=skf,
        scoring='accuracy',
        n_jobs=-1,  # Parallelize the search across available cores
        verbose=1
    )

    # Perform the grid search
    grid_search.fit(X, Y)

    # Get the best parameters and the best cross-validation score
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"Best Alpha: {best_params['alpha']}")
    print(f"Best Number of Samples: {best_params['n_samples']}")
    print(f"Best Cross-Validation Accuracy: {best_score:.4f}")

    # Now let's visualize the results
    mean_accuracies = grid_search.cv_results_['mean_test_score']
    param_alpha = [params['alpha'] for params in grid_search.cv_results_['params']]
    param_n_samples = [params['n_samples'] for params in grid_search.cv_results_['params']]

    alpha_values, sample_values = np.meshgrid(
        np.logspace(-3, np.log10(0.9), 5), [10, 50, 100, 200, 500, 1000]
    )

    mean_accuracies = np.array(mean_accuracies).reshape(alpha_values.shape)

    # Plot accuracy surface
    plot_accuracy_surface(alpha_values, sample_values, mean_accuracies)

    return best_params, best_score


def plot_accuracy_surface(alpha_values, sample_values, mean_accuracies):
    """
    Plots surface plots of mean accuracy over different hyperparameter values.
    """
    # Flatten the results for plotting
    mesh_data = {
        'alpha': alpha_values.flatten(),
        'n_samples': sample_values.flatten(),
        'mean_accuracy': mean_accuracies.flatten()
    }

    df = pd.DataFrame(mesh_data)

    # Create a 3D scatter plot of the hyperparameter space
    fig = px.scatter_3d(df, x='alpha', y='n_samples', z='mean_accuracy',
                        title='Mean Cross-Validation Accuracy for Soft Decision Tree',
                        labels={'alpha': 'Alpha (Regularization)', 'n_samples': 'Number of Samples',
                                'mean_accuracy': 'Mean Accuracy'},
                        opacity=0.7)

    fig.update_traces(marker=dict(size=5))
    fig.show()
