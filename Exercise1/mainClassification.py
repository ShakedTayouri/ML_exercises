import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, auc, roc_curve
from sklearn.model_selection import cross_validate, cross_val_predict
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from Exercise1.BestHyperParameters import print_best_decision_tree_classifier
from Exercise1.DecisionTrees.SoftDecisionTreeClassifier import SoftDecisionTreeClassifier
from Exercise1.SensitivityAnalysis import find_best_hyperparameters


def compare_soft_and_regular_decision_tree(x, y, criterion='gini', max_depth=None, min_samples_leaf=1,
                                           min_samples_split=2,
                                           max_features=None, alpha=0.3, n_samples=100, random_state=None):
    """
    Compare performance of a regular DecisionTreeClassifier and a SoftDecisionTreeClassifier.
    This includes accuracy scores and predicted probabilities for train and test sets.
    """
    # Convert inputs to appropriate types
    x = pd.DataFrame(x)
    y = pd.Series(y)

    # Initialize classifiers
    clf_regular = DecisionTreeClassifier(random_state=random_state, criterion=criterion, max_depth=max_depth,
                                         min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split,
                                         max_features=max_features)
    basic_clf_soft = SoftDecisionTreeClassifier(alpha=0.1, n_samples=100, criterion=criterion,
                                                random_state=random_state,
                                                max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                                min_samples_split=min_samples_split, max_features=max_features)
    clf_soft = SoftDecisionTreeClassifier(alpha=alpha, n_samples=n_samples, criterion=criterion,
                                          random_state=random_state,
                                          max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                          min_samples_split=min_samples_split, max_features=max_features)

    print('Regular DecisionTreeClassifier:')
    evaluate_classifier(clf_regular, x, y)

    print('\nBasic SoftDecisionTreeClassifier:')
    evaluate_classifier(basic_clf_soft, x, y)

    print('\nSoftDecisionTreeClassifier:')
    evaluate_classifier(clf_soft, x, y)


def plot_roc_curve(y, preds_proba):
    """
    Plot the ROC curve and calculate the AUC for both binary and multi-class classification.
    """
    if preds_proba.shape[1] > 2:  # Multi-class case
        # Compute ROC curve and AUC for each class
        fpr, tpr, roc_auc = {}, {}, {}
        for i in range(preds_proba.shape[1]):
            fpr[i], tpr[i], _ = roc_curve(y, preds_proba[:, i], pos_label=i)
            roc_auc[i] = auc(fpr[i], tpr[i])

        # Calculate macro-average AUC
        all_fpr = np.unique(np.concatenate([fpr[i] for i in range(preds_proba.shape[1])]))
        mean_tpr = np.zeros_like(all_fpr)
        for i in range(preds_proba.shape[1]):
            mean_tpr += np.interp(all_fpr, fpr[i], tpr[i])
        mean_tpr /= preds_proba.shape[1]

        # Calculate the macro-average AUC
        macro_roc_auc = auc(all_fpr, mean_tpr)
        print(f"Macro-Average AUC: {macro_roc_auc:.2f}")

        # Plot ROC curve for each class
        plt.figure()
        for i in range(preds_proba.shape[1]):
            plt.plot(fpr[i], tpr[i], lw=2, label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - Multi-Class')
        plt.legend(loc="lower right")
        plt.show()

    else:  # Binary classification case
        fpr, tpr, _ = roc_curve(y,
                                preds_proba[:, 1])  # For binary, use the second column (positive class probabilities)
        roc_auc = auc(fpr, tpr)
        print("AUC (Area Under Curve):", roc_auc)

        # Plot ROC curve for binary classification
        plt.figure()
        plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='gray', linestyle='--')  # Random classifier line
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) - Binary')
        plt.legend(loc="lower right")
        plt.show()


def evaluate_classifier(clf, x, y):
    """
    Evaluate classifier using cross-validation and print accuracy scores and predicted probabilities.
    """

    cv_results = cross_validate(clf, x, y, cv=5, return_train_score=True)
    print("Test Score (CV):", cv_results['test_score'].mean())
    print("Train Score (CV):", cv_results['train_score'].mean())

    preds_proba = cross_val_predict(clf, x, y, cv=5, method='predict_proba')
    y_pred_labels = np.argmax(preds_proba, axis=1)

    predict_proba_accuracy = accuracy_score(y, y_pred_labels)
    print("Accuracy from predicted probabilities (all data):", predict_proba_accuracy)

    plot_roc_curve(y, preds_proba)


if __name__ == '__main__':
    # 42 features, 2000 samples
    data = pd.read_csv(
        'C:\\Users\\shake\\PycharmProjects\\ML_exs\\Exercise1\\datasets\\mobile_price.csv')
    X, Y = data.drop(columns=['price_range']), data['price_range']

    print("Mobile Price Dataset\n")
    print_best_decision_tree_classifier(X, Y)
    find_best_hyperparameters(X, Y)
    compare_soft_and_regular_decision_tree(X, Y, max_depth=10, max_features='sqrt', min_samples_leaf=1,
                                           min_samples_split=10, alpha=0.1, n_samples=50)
    print("\n")

    # 35 features, 2149 samples
    data = pd.read_csv(
        'C:\\Users\\shake\\PycharmProjects\\ML_exs\\Exercise1\\datasets\\alzheimers_disease_data.csv')
    X, Y = data.drop(columns=['Diagnosis', 'DoctorInCharge']), data['Diagnosis']

    print("alzheimers_disease Dataset\n")
    # print_best_decision_tree_classifier(X, Y)
    # find_best_hyperparameters(X, Y)
    compare_soft_and_regular_decision_tree(X, Y, criterion='entropy', max_depth=10, max_features='sqrt',
                                           min_samples_leaf=1,
                                           min_samples_split=10, alpha=0.06, n_samples=200)
    print("\n")

    # 15 features, 2392 samples
    data = pd.read_csv(
        'C:\\Users\\shake\\PycharmProjects\\ML_exs\\Exercise1\\datasets\\Student_performance_data _.csv')
    X, Y = data.drop(columns=['GradeClass']), data['GradeClass']

    print("Student Performance Dataset\n")
    print_best_decision_tree_classifier(X, Y)
    find_best_hyperparameters(X, Y)
    compare_soft_and_regular_decision_tree(X, Y, max_depth=20, max_features='log2', min_samples_leaf=1,
                                           min_samples_split=20, alpha=0.03, n_samples=500)
    print("\n")

    # 13 features, 52444 samples
    data = pd.read_csv(
        'C:\\Users\\shake\\PycharmProjects\\ML_exs\\Exercise1\\datasets\\mountains_vs_beaches_preferences.csv')
    X, Y = data.drop(columns=['Preference']), data['Preference']

    label_encoder = LabelEncoder()
    X['Gender'] = label_encoder.fit_transform(X['Gender'])
    X['Education_Level'] = label_encoder.fit_transform(X['Education_Level'])
    X['Preferred_Activities'] = label_encoder.fit_transform(X['Preferred_Activities'])
    X['Location'] = label_encoder.fit_transform(X['Location'])
    X['Favorite_Season'] = label_encoder.fit_transform(X['Favorite_Season'])

    print("mountains_vs_beaches_preferences Dataset\n")

    print_best_decision_tree_classifier(X, Y)
    find_best_hyperparameters(X, Y)
    compare_soft_and_regular_decision_tree(X, Y, criterion='entropy', max_depth=20, max_features='sqrt',
                                           min_samples_leaf=1,
                                           min_samples_split=10, alpha=0.02, n_samples=500)
    print("\n")

    # 50 features, 103904 samples
    data = pd.read_csv(
        'C:\\Users\\shake\\PycharmProjects\\ML_exs\\Exercise1\\datasets\\Airline.csv')
    X, Y = data.drop(columns=['satisfaction']), data['satisfaction']

    label_encoder = LabelEncoder()
    X['Gender'] = label_encoder.fit_transform(X['Gender'])
    X['Customer Type'] = label_encoder.fit_transform(X['Customer Type'])
    X['Type of Travel'] = label_encoder.fit_transform(X['Type of Travel'])
    X['Class'] = label_encoder.fit_transform(X['Class'])
    Y = label_encoder.fit_transform(Y)

    print("Airline Dataset\n")
    print_best_decision_tree_classifier(X, Y)
    find_best_hyperparameters(X, Y)
    compare_soft_and_regular_decision_tree(X, Y, criterion='entropy', max_depth=20, max_features='log2',
                                           min_samples_leaf=1,
                                           min_samples_split=20, alpha=0.01, n_samples=10)
