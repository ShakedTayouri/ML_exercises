import numpy as np
import pandas as pd
import plotly.express as px
from sklearn.metrics import accuracy_score

from Exercise1.SoftDecisionTreeClassifier import SoftDecisionTreeClassifier


def get_accuracy(X_train, Y_train, X_test, Y_test, alpha, n_samples):
    clf = SoftDecisionTreeClassifier(alpha, n_samples)
    clf = clf.fit(X_train, Y_train)

    y_pred = clf.predict_proba(X_test)
    y_pred_labels = np.argmax(y_pred, axis=1)

    return accuracy_score(Y_test, y_pred_labels)


def evaluate_hyperparameters(X_train, Y_train, X_test, Y_test, alphas, n_samples):
    """
    Evaluate the Soft Decision Tree for different combinations of `alpha` and `n_samples`
    and return training and testing accuracies.
    """
    train_accuracies = np.zeros((len(n_samples), len(alphas)))
    test_accuracies = np.zeros((len(n_samples), len(alphas)))

    for i, n_sample in enumerate(n_samples):
        for j, alpha in enumerate(alphas):
            # Get training accuracy
            train_accuracy = get_accuracy(X_train, Y_train, X_train, Y_train, alpha, n_sample)
            train_accuracies[i, j] = train_accuracy

            # Get testing accuracy
            test_accuracy = get_accuracy(X_train, Y_train, X_test, Y_test, alpha, n_sample)
            test_accuracies[i, j] = test_accuracy

    return train_accuracies, test_accuracies


def plot_accuracy_surface(alpha_values, sample_values, train_accuracy_values, test_accuracy_values):
    """
    Plots surface plots of accuracy over different hyperparameter values.
    """
    # Flatten for plotly
    mesh_data = {
        'alpha': alpha_values.flatten(),
        'n_samples': sample_values.flatten(),
        'train_accuracy': train_accuracy_values.flatten(),
        'test_accuracy': test_accuracy_values.flatten()
    }

    df = pd.DataFrame(mesh_data)

    # Training Accuracy Plot
    fig_train = px.scatter_3d(df, x='alpha', y='n_samples', z='train_accuracy',
                              title='Training Soft Decision Tree Accuracy',
                              labels={'alpha': 'Alpha (Regularization)', 'n_samples': 'Number of Samples',
                                      'train_accuracy': 'Train Accuracy'},
                              opacity=0.7)

    # Testing Accuracy Plot
    fig_test = px.scatter_3d(df, x='alpha', y='n_samples', z='test_accuracy',
                             title='Testing Soft Decision Tree Accuracy',
                             labels={'alpha': 'Alpha (Regularization)', 'n_samples': 'Number of Samples',
                                     'test_accuracy': 'Test Accuracy'},
                             opacity=0.7)

    fig_train.show()
    fig_test.show()


def find_best_hyperparameters(X_train, X_test, Y_train, Y_test):
    """
    Function to find the best hyperparameters for Soft Decision Tree based on accuracy.
    """
    alphas = np.arange(0.1, 1.1, 0.1)
    n_samples = [2, 10, 50, 100, 300, 700, 1000]

    train_accuracies, test_accuracies = evaluate_hyperparameters(X_train, Y_train, X_test, Y_test, alphas, n_samples)

    alpha_values, sample_values = np.meshgrid(alphas, n_samples)
    plot_accuracy_surface(alpha_values, sample_values, train_accuracies, test_accuracies)
