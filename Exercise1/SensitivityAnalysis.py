import numpy as np
import plotly.graph_objects as go
from sklearn.metrics import accuracy_score, r2_score
from sklearn.model_selection import train_test_split

from Exercise1.DecisionTrees.SoftDecisionTreeClassifier import SoftDecisionTreeClassifier
from Exercise1.DecisionTrees.SoftDecisionTreeRegressor import SoftDecisionTreeRegressor
from Exercise1.DecisionTrees.WeightedDecisionTreeClassifier import WeightedDecisionTreeClassifier


def get_soft_decision_tree_classifier_accuracy(X_train, Y_train, X_test, Y_test, alpha, n_samples):
    clf = SoftDecisionTreeClassifier(alpha=alpha, n_samples=n_samples)
    print(alpha)
    clf.fit(X_train, Y_train)

    preds_proba = clf.predict_proba(X_test)
    y_pred_labels = np.argmax(preds_proba, axis=1)

    return accuracy_score(Y_test, y_pred_labels)


def get_weighted_decision_tree_classifier_accuracy(X_train, Y_train, X_test, Y_test, alpha, n_samples):
    clf = WeightedDecisionTreeClassifier(alpha=alpha, n_samples=n_samples)
    print(alpha)
    clf.fit(X_train, Y_train)

    y_pred_labels = clf.predict(X_test)

    return accuracy_score(Y_test, y_pred_labels)


def get_soft_decision_tree_regressor_r2(X_train, Y_train, X_test, Y_test, alpha, n_samples):
    clf = SoftDecisionTreeRegressor(alpha=alpha, n_samples=n_samples)
    print(f"Training with alpha={alpha}, n_samples={n_samples}")

    # Fit the model
    clf.fit(X_train, Y_train)

    # Get predictions
    train_predictions = clf.predict(X_train)
    test_predictions = clf.predict(X_test)

    # Compute R² score
    train_r2 = r2_score(Y_train, train_predictions)
    test_r2 = r2_score(Y_test, test_predictions)

    return train_r2, test_r2


def find_best_hyperparameters_for_soft_regressor(X, Y):
    """
    Function to find the best hyperparameters for Soft Decision Tree using cross-validation with grid search.
    This version is for regression (using R² score).
    """

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3)

    alphas = np.concatenate((np.arange(0.01, 0.1, 0.02), np.arange(0.1, 0.4, 0.1)))
    print("Alphas:", alphas)
    n_samples = [10, 50, 100, 150]

    train_r2_scores = np.zeros((len(alphas), len(n_samples)))
    test_r2_scores = np.zeros((len(alphas), len(n_samples)))

    # Train decision trees for each combination of parameters
    for i, alpha in enumerate(alphas):
        for j, n_sample in enumerate(n_samples):
            # Get R² scores for training and testing
            train_r2, test_r2 = get_soft_decision_tree_regressor_r2(X_train, Y_train, X_test, Y_test, alpha=alpha,
                                                                    n_samples=n_sample)

            train_r2_scores[i, j] = train_r2
            test_r2_scores[i, j] = test_r2

    # Find the best hyperparameters based on R² score
    max_train_idx = np.unravel_index(np.argmin(train_r2_scores), train_r2_scores.shape)
    print(f"Maximum training R²: {train_r2_scores[max_train_idx]}")
    print(f"Best alpha for training: {alphas[max_train_idx[0]]}")
    print(f"Best n_samples for training: {n_samples[max_train_idx[1]]}")

    max_test_idx = np.unravel_index(np.argmax(test_r2_scores), test_r2_scores.shape)
    print(f"Min test R²: {test_r2_scores[max_test_idx]}")
    print(f"Best alpha for testing: {alphas[max_test_idx[0]]}")
    print(f"Best n_samples for testing: {n_samples[max_test_idx[1]]}")

    plot_r2_surface(test_r2_scores, alphas, n_samples)


def find_best_hyperparameters_for_soft_classifier(X, Y):


    def get_soft_decision_tree_regressor_r2(X_train, Y_train, X_test, Y_test, alpha, n_samples):
        clf = SoftDecisionTreeRegressor(alpha=alpha, n_samples=n_samples)
        print(f"Training with alpha={alpha}, n_samples={n_samples}")

        # Fit the model
        clf.fit(X_train, Y_train)

        # Get predictions
        train_predictions = clf.predict(X_train)
        test_predictions = clf.predict(X_test)

        # Compute R² score
        train_r2 = r2_score(Y_train, train_predictions)
        test_r2 = r2_score(Y_test, test_predictions)

        return train_r2, test_r2


def find_best_hyperparameters_for_soft_regressor(X, Y):
    """
    Function to find the best hyperparameters for Soft Decision Tree using cross-validation with grid search.
    This version is for regression (using R² score).
    """

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3)

    alphas = np.concatenate((np.arange(0.01, 0.1, 0.02), np.arange(0.1, 0.4, 0.1)))
    print("Alphas:", alphas)
    n_samples = [10, 50, 100, 150]

    train_r2_scores = np.zeros((len(alphas), len(n_samples)))
    test_r2_scores = np.zeros((len(alphas), len(n_samples)))

    # Train decision trees for each combination of parameters
    for i, alpha in enumerate(alphas):
        for j, n_sample in enumerate(n_samples):
            # Get R² scores for training and testing
            train_r2, test_r2 = get_soft_decision_tree_regressor_r2(X_train, Y_train, X_test, Y_test, alpha=alpha,
                                                                    n_samples=n_sample)

            train_r2_scores[i, j] = train_r2
            test_r2_scores[i, j] = test_r2

    # Find the best hyperparameters based on R² score
    max_train_idx = np.unravel_index(np.argmin(train_r2_scores), train_r2_scores.shape)
    print(f"Maximum training R²: {train_r2_scores[max_train_idx]}")
    print(f"Best alpha for training: {alphas[max_train_idx[0]]}")
    print(f"Best n_samples for training: {n_samples[max_train_idx[1]]}")

    max_test_idx = np.unravel_index(np.argmax(test_r2_scores), test_r2_scores.shape)
    print(f"Min test R²: {test_r2_scores[max_test_idx]}")
    print(f"Best alpha for testing: {alphas[max_test_idx[0]]}")
    print(f"Best n_samples for testing: {n_samples[max_test_idx[1]]}")

    plot_r2_surface(test_r2_scores, alphas, n_samples)


def find_best_hyperparameters_for_soft_classifier(X, Y):
    """
    Function to find the best hyperparameters for Soft Decision Tree using cross-validation with grid search.
    """

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3)

    alphas = np.concatenate((np.arange(0.01, 0.1, 0.01), np.arange(0.1, 0.6, 0.1)))
    print(alphas)
    n_samples = [10, 50, 100, 200, 500]

    train_accuracies = np.zeros((len(alphas), len(n_samples)))
    test_accuracies = np.zeros((len(alphas), len(n_samples)))

    # Train decision trees for each combination of parameters
    for i, alpha in enumerate(alphas):
        for j, n_sample in enumerate(n_samples):
            train_accuracies[i, j] = get_soft_decision_tree_classifier_accuracy(X_train, Y_train, X_train, Y_train,
                                                                                alpha=alpha,
                                                                                n_samples=n_sample)
            test_accuracies[i, j] = get_soft_decision_tree_classifier_accuracy(X_train, Y_train, X_test, Y_test,
                                                                               alpha=alpha,
                                                                               n_samples=n_sample)

    max_train_idx = np.unravel_index(np.argmax(train_accuracies), train_accuracies.shape)
    print(f"Maximum training accuracy: {train_accuracies[max_train_idx]}")
    print(f"Best alpha for training: {alphas[max_train_idx[0]]}")
    print(f"Best n_samples for training: {n_samples[max_train_idx[1]]}")

    max_test_idx = np.unravel_index(np.argmax(test_accuracies), test_accuracies.shape)
    print(f"Maximum test accuracy: {test_accuracies[max_test_idx]}")
    print(f"Best alpha for testing: {alphas[max_test_idx[0]]}")
    print(f"Best n_samples for testing: {n_samples[max_test_idx[1]]}")

    plot_accuracy_surface(test_accuracies, alphas, n_samples)


def find_best_hyperparameters_for_WD_classifier(X, Y):
    """
    Function to find the best hyperparameters for Weighted Decision Tree using cross-validation with grid search.
    """

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, train_size=0.7, test_size=0.3)

    alphas = np.concatenate((np.arange(0.01, 0.1, 0.01), np.arange(0.1, 0.6, 0.1)))
    print(alphas)
    n_samples = [10, 50, 100, 200, 500]

    train_accuracies = np.zeros((len(alphas), len(n_samples)))
    test_accuracies = np.zeros((len(alphas), len(n_samples)))

    # Train decision trees for each combination of parameters
    for i, alpha in enumerate(alphas):
        for j, n_sample in enumerate(n_samples):
            train_accuracies[i, j] = get_weighted_decision_tree_classifier_accuracy(X_train, Y_train, X_train, Y_train,
                                                                                    alpha=alpha,
                                                                                    n_samples=n_sample)
            test_accuracies[i, j] = get_weighted_decision_tree_classifier_accuracy(X_train, Y_train, X_test, Y_test,
                                                                                   alpha=alpha,
                                                                                   n_samples=n_sample)

    max_train_idx = np.unravel_index(np.argmax(train_accuracies), train_accuracies.shape)
    print(f"Maximum training accuracy: {train_accuracies[max_train_idx]}")
    print(f"Best alpha for training: {alphas[max_train_idx[0]]}")
    print(f"Best n_samples for training: {n_samples[max_train_idx[1]]}")

    max_test_idx = np.unravel_index(np.argmax(test_accuracies), test_accuracies.shape)
    print(f"Maximum test accuracy: {test_accuracies[max_test_idx]}")
    print(f"Best alpha for testing: {alphas[max_test_idx[0]]}")
    print(f"Best n_samples for testing: {n_samples[max_test_idx[1]]}")

    plot_accuracy_surface(test_accuracies, alphas, n_samples)


def plot_accuracy_surface(accuracies, alphas, n_samples):
    """
    Plots surface plots of mean accuracy over different hyperparameter values.
    """

    fig = go.Figure(data=[go.Surface(z=accuracies, x=n_samples, y=alphas, colorscale='Viridis')])

    # Add labels and title
    fig.update_layout(
        scene=dict(
            xaxis_title='n_samples',
            yaxis_title='Alpha',
            zaxis_title='Accuracy'
        ),
        width=800,
        height=800,
        coloraxis_colorbar=dict(
            len=0.5,  # Set the length of the color bar (relative to the plot's height)
            thickness=15,  # Make the color bar narrower
            tickvals=[0, 0.25, 0.5, 0.75, 1],  # Customize tick marks on the color bar
            ticktext=['0', '0.25', '0.5', '0.75', '1'],  # Customize tick labels
            title='Accuracy',  # Title of the color bar
            ticks='outside',  # Position the ticks outside of the color bar
            ticklen=5,  # Length of ticks
            tickwidth=2,  # Width of ticks
        ),

    )

    fig.show()


def plot_r2_surface(r2_scores, alphas, n_samples):
    """
    Plots surface plots of R² score over different hyperparameter values.
    """

    # Create surface plot with R² scores
    fig = go.Figure(data=[go.Surface(z=r2_scores, x=n_samples, y=alphas, colorscale='Viridis')])

    # Add labels and title for R² plot
    fig.update_layout(
        scene=dict(
            xaxis_title='n_samples',
            yaxis_title='Alpha',
            zaxis_title='R² Score'
        ),
        width=800,
        height=800,
        coloraxis_colorbar=dict(
            len=0.5,  # Set the length of the color bar (relative to the plot's height)
            thickness=15,  # Make the color bar narrower
            tickvals=[-1, 0, 0.25, 0.5, 0.75, 1],  # Customize tick marks on the color bar
            ticktext=['-1', '0', '0.25', '0.5', '0.75', '1'],  # Customize tick labels
            title='R² Score',  # Title of the color bar
            ticks='outside',  # Position the ticks outside of the color bar
            ticklen=5,  # Length of ticks
            tickwidth=2,  # Width of ticks
        ),
    )

    fig.show()
