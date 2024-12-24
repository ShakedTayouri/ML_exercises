import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


class SoftDistanceDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, alpha, n_samples, **kwargs):
        """
        alpha: The probability of routing a sample in the opposite direction at each split.
        n_samples: The number of times to run the prediction for each sample to average probabilities.
        """
        super().__init__(**kwargs)

        # Store the additional parameters for the soft decision tree
        self.alpha = alpha
        self.n_samples = n_samples
        self.node_proportions_ = None

    def fit(self, X, y):
        super().fit(X, y)
        
        # Calculate class proportions at each node
        self.node_proportions_ = []
        for i in range(self.tree_.node_count):
            node_value = self.tree_.value[i].flatten()
            node_proportion = node_value / np.sum(node_value)
            self.node_proportions_.append(node_proportion)
            
        return self

    def _predict_proba_soft(self, X):
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        prob = np.zeros((n_samples, n_classes))

        for i in range(n_samples):
            node = 0
            while self.tree_.children_left[node] != -1:
                feature = self.tree_.feature[node]
                threshold = self.tree_.threshold[node]
                feature_value = X[i, feature]

                # Calculate distance from threshold
                distance = np.abs(feature_value - threshold)
                soft_prob = np.exp(-self.alpha * distance)  # Softness decreases with distance

                # Decide based on distance
                if feature_value <= threshold:
                    if np.random.rand() > soft_prob:
                        node = self.tree_.children_left[node]
                    else:
                        node = self.tree_.children_right[node]
                else:
                    if np.random.rand() > soft_prob:
                        node = self.tree_.children_right[node]
                    else:
                        node = self.tree_.children_left[node]

            # Use node proportions to create a soft class distribution
            prob[i] = self.node_proportions_[node]

        return prob

    def predict_proba(self, X, check_input=True):
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        return self._predict_proba_soft(X)

    def predict(self, X):
        return np.argmax(self.predict_proba(X), axis=1)

    def _get_default_requests(self):
        from sklearn.utils._metadata_requests import MetadataRequest
        return MetadataRequest(self)
