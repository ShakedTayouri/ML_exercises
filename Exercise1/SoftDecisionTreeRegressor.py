from sklearn.tree import DecisionTreeRegressor
from sklearn.utils.validation import check_is_fitted
import numpy as np


class SoftDecisionTreeRegressor(DecisionTreeRegressor):
    def __init__(self, alpha, n_samples, max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=None):
        """
        alpha: Probability of taking the opposite split at each node.
        n_samples: Number of times to average predictions during inference.
        max_depth, min_samples_leaf, min_samples_split: Passed to the base DecisionTreeRegressor.
        """
        super().__init__(max_depth=max_depth,
                         min_samples_leaf=min_samples_leaf,
                         min_samples_split=min_samples_split,
                         random_state=random_state)
        self.alpha = alpha
        self.n_samples = n_samples

    def predict(self, X, check_input=True):
        """
        Override predict method to introduce soft splits during inference.
        """
        check_is_fitted(self)
        X = self._validate_X_predict(X, check_input)
        n_samples = X.shape[0]

        # Accumulate predictions for each pass
        predictions = np.zeros(n_samples)

        # Perform soft split predictions n_samples times
        for _ in range(self.n_samples):
            predictions += self._predict_soft(X)

        # Average predictions over all passes
        predictions /= self.n_samples
        return predictions

    def _predict_soft(self, X):
        """
        Perform a soft prediction for each sample, traversing the tree with probabilistic splits.
        """
        n_samples = X.shape[0]
        predictions = np.zeros(n_samples)

        for i in range(n_samples):
            node = 0  # Start at the root node
            while self.tree_.children_left[node] != -1:  # Continue until reaching a leaf
                feature = self.tree_.feature[node]
                threshold = self.tree_.threshold[node]
                feature_value = X[i, feature]

                # Perform soft split (probabilistic routing)
                if feature_value <= threshold:
                    if np.random.rand() > self.alpha:
                        node = self.tree_.children_left[node]
                    else:
                        node = self.tree_.children_right[node]
                else:
                    if np.random.rand() > self.alpha:
                        node = self.tree_.children_right[node]
                    else:
                        node = self.tree_.children_left[node]

            # At the leaf node, predict the value
            predictions[i] = self.tree_.value[node][0][0]

        return predictions