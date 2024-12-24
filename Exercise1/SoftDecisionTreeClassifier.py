import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.utils.validation import check_is_fitted


class SoftDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, alpha, n_samples, **kwargs):
        """
        alpha: The probability of routing a sample in the opposite direction at each split.
        n_samples: The number of times to run the prediction for each sample to average probabilities.
        """
        super().__init__(**kwargs)

        # Store the additional parameters for the soft decision tree
        self.alpha = alpha
        self.n_samples = n_samples

    def predict_proba(self, X, check_input=True):
        """
        Override the predict_proba function to implement soft splits.
        """

        # Checks that the model has been fitted
        check_is_fitted(self)
        # Checks that X is in the correct format for making predictions
        X = self._validate_X_predict(X, check_input)

        n_samples = X.shape[0]
        n_classes = len(self.classes_)

        # Initialize an array to accumulate probability vectors for each sample
        prob_accumulated = np.zeros((n_samples, n_classes))

        # Run the prediction `n_samples` times
        for _ in range(self.n_samples):
            prob_accumulated += self._predict_proba_soft(X)

        # Average the probabilities over the n_samples
        prob_accumulated /= self.n_samples

        # Handle multi-output classification
        if self.n_outputs_ == 1:
            return prob_accumulated
        else:
            all_proba = []
            for k in range(self.n_outputs_):
                # Slice to get probabilities for each output
                prob_k = prob_accumulated[:, k, :self.n_classes_[k]]
                all_proba.append(prob_k)
            return all_proba

    def _predict_proba_soft(self, X):
        """
        Perform a soft prediction for each sample, with random splits.
        """
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        prob = np.zeros((n_samples, n_classes))

        # Traverse each sample through the tree with soft splits
        for i in range(n_samples):
            node = 0  # Start at the root node
            while self.tree_.children_left[node] != -1:  # Until reaching a leaf
                feature = self.tree_.feature[node]
                threshold = self.tree_.threshold[node]
                feature_value = X[i, feature]

                # Apply soft split: decide probabilistically which direction to take
                if feature_value <= threshold:
                    # With probability (1-alpha), go left, and with alpha, go right
                    if np.random.rand() > self.alpha:
                        node = self.tree_.children_left[node]
                    else:
                        node = self.tree_.children_right[node]
                else:
                    if np.random.rand() > self.alpha:
                        node = self.tree_.children_right[node]
                    else:
                        node = self.tree_.children_left[node]

            # Once a leaf node is reached, store the class probabilities from that leaf
            prob[i] = self.tree_.value[node].flatten()

        return prob