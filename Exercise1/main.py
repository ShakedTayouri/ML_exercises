import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_validate, cross_val_predict, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from Exercise1.SoftDecisionTreeClassifier import SoftDecisionTreeClassifier


def compare_soft_and_regular_decision_tree(x, y, max_depth=None, min_samples_leaf=1, min_samples_split=2,
                                           alpha=0.1, n_samples=100, random_state=None):
    """
    Compare performance of a regular DecisionTreeClassifier and a SoftDecisionTreeClassifier.
    This includes accuracy scores and predicted probabilities for train and test sets.
    """
    # Convert inputs to appropriate types
    x = pd.DataFrame(x)
    y = pd.Series(y)

    # Initialize classifiers
    clf_regular = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth,
                                         min_samples_leaf=min_samples_leaf, min_samples_split=min_samples_split)
    clf_soft = SoftDecisionTreeClassifier(alpha=alpha, n_samples=n_samples, random_state=random_state,
                                          max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                          min_samples_split=min_samples_split)

    print('Regular DecisionTreeClassifier:')
    evaluate_classifier(clf_regular, x, y)

    print('\nSoftDecisionTreeClassifier:')
    evaluate_classifier(clf_soft, x, y)


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


if __name__ == '__main__':
    # 2 features, 150 samples
    data = load_iris()
    X, Y = data.data, data.target
    print("Iris Dataset:")
    compare_soft_and_regular_decision_tree(X, Y, 6, 1, 2)
    print("\n")

    # 31 features, 569 samples
    data = pd.read_csv(
        'C:\\Users\\shake\\PycharmProjects\\ML_exs\\Exercise1\\datasets\\breast_cancer_wisconsin_diagnostic.csv')
    X, Y = data.drop(columns=['diagnosis']), data['diagnosis']
    label_encoder = LabelEncoder()
    Y = label_encoder.fit_transform(Y)

    print("Breast Cancer Wisconsin Diagnostic Dataset:")
    compare_soft_and_regular_decision_tree(X, Y, 6, 1, 2)
    print("\n")

    # 13 features and 303 samples
    data = pd.read_csv(
        'C:\\Users\\shake\\PycharmProjects\\ML_exs\\Exercise1\\datasets\\Heart_disease_cleveland_new.csv')
    X, Y = data.drop(columns=['target']), data['target']

    print("Heart Disease Cleveland Dataset\n")
    compare_soft_and_regular_decision_tree(X, Y, 7, 5, 2)
    print("\n")

    # 8 features, 768 samples
    data = pd.read_csv(
        'C:\\Users\\shake\\PycharmProjects\\ML_exs\\Exercise1\\datasets\\diabetes.csv')
    X, Y = data.drop(columns=['Outcome']), data['Outcome']

    print("Diabetes Dataset\n")
    compare_soft_and_regular_decision_tree(X, Y, 6, 1, 2)
    print("\n")

    # 54 features, 1315 samples
    data = pd.read_csv(
        'C:\\Users\\shake\\PycharmProjects\\ML_exs\\Exercise1\\datasets\\covtype.csv')
    X, Y = data.drop(columns=['Cover_Type']), data['Cover_Type']

    print("Covtype Dataset\n")
    compare_soft_and_regular_decision_tree(X, Y)
    print("\n")
