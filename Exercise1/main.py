import numpy as np
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from Exercise1.SoftDecisionTreeClassifier import SoftDecisionTreeClassifier

if __name__ == '__main__':
    data = load_iris()
    X = data.data
    y = data.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    # Use the origin classifier to compare
    # clf = DecisionTreeClassifier(random_state=42, max_depth=5, min_samples_leaf=1, min_samples_split=10)

    # When alpha high the accuracy is low
    clf = SoftDecisionTreeClassifier(alpha=0.1, n_samples=10, random_state=42, max_depth=5, min_samples_leaf=1,
                                     min_samples_split=10)
    clf.fit(X_train, y_train)

    proba_train = clf.predict_proba(X_train)
    y_pred_labels_train = np.argmax(proba_train, axis=1)

    proba_test = clf.predict_proba(X_test)
    y_pred_labels_test = np.argmax(proba_test, axis=1)

    print('score on training: {score}'.format(score=accuracy_score(y_train, y_pred_labels_train)))
    print('score on test: {score}'.format(score=accuracy_score(y_test, y_pred_labels_test)))
