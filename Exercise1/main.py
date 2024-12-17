import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier

from Exercise1.SoftDecisionTreeClassifier import SoftDecisionTreeClassifier


def compare_soft_and_regular_decision_tree(x, y, max_depth=None, min_samples_leaf=1, min_samples_split=2,
                                           random_state=None):
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.3, random_state=42)

    # origin classifier to compare
    clf = DecisionTreeClassifier(random_state=random_state, max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                 min_samples_split=min_samples_split, )
    predict_accuries(clf, x_test, x_train, y_test, y_train)

    clf = SoftDecisionTreeClassifier(alpha=0.1, n_samples=10, random_state=random_state, max_depth=max_depth,
                                     min_samples_leaf=min_samples_leaf,
                                     min_samples_split=10)
    predict_accuries(clf, x_test, x_train, y_test, y_train)


def predict_accuries(clf, x_test, x_train, y_test, y_train):
    clf.fit(x_train, y_train)
    proba_train = clf.predict_proba(x_train)
    y_pred_labels_train = np.argmax(proba_train, axis=1)
    proba_test = clf.predict_proba(x_test)
    y_pred_labels_test = np.argmax(proba_test, axis=1)
    print('score on training: {score}'.format(score=accuracy_score(y_train, y_pred_labels_train)))
    print('score on test: {score}'.format(score=accuracy_score(y_test, y_pred_labels_test)))


if __name__ == '__main__':
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

    # 54 features, 1315 samples
    data = pd.read_csv(
        'C:\\Users\\shake\\PycharmProjects\\ML_exs\\Exercise1\\datasets\\covtype.csv')
    X, Y = data.drop(columns=['Cover_Type']), data['Cover_Type']

    print("Covtype Dataset\n")
    compare_soft_and_regular_decision_tree(X, Y, 6, 1, 2)
    print("\n")

    # 8 features, 768 samples
    data = pd.read_csv(
        'C:\\Users\\shake\\PycharmProjects\\ML_exs\\Exercise1\\datasets\\diabetes.csv')
    X, Y = data.drop(columns=['Outcome']), data['Outcome']

    print("Diabetes Dataset\n")
    compare_soft_and_regular_decision_tree(X, Y, 6, 1, 2)
    print("\n")
