from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._tags import Tags, ClassifierTags, TargetTags
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
import pandas as pd



class WeightedDecisionTreeClassifier(DecisionTreeClassifier):
    def __init__(self, max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=None, ccp_alpha=0.0):
        super().__init__(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                         min_samples_split=min_samples_split, random_state=random_state, ccp_alpha=ccp_alpha)
        self.weights_ = None

    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)  # Ensure X is float32
        super().fit(X, y)

        self.weights_ = []
        for i in range(self.tree_.node_count):
            if self.tree_.children_left[i] == -1:  # Leaf node
                leaf_y = y[self.tree_.apply(X) == i]
                class_counts = np.bincount(leaf_y)
                majority_class = np.argmax(class_counts)
                gini = self.gini_impurity(leaf_y)

                # Confidence-based flipping
                if len(class_counts) > 1 and class_counts[majority_class] / np.sum(class_counts) < 0.7 and gini > 0.55:
                    flipped_class = np.random.choice(np.unique(leaf_y))
                    self.weights_.append((flipped_class, gini))
                else:
                    self.weights_.append((majority_class, gini))
            else:
                self.weights_.append(None)
        return self

    def predict(self, X):
        # Consistently handle DataFrame inputs during prediction
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.array(X, dtype=np.float32)
        leaf_indices = self.tree_.apply(X)

        # Generate predictions based on the leaf weights
        predictions = np.array([self.weights_[idx][0] for idx in leaf_indices])
        return predictions

    def score(self, X, y):
        # Compute accuracy based on predictions
        y_pred = self.predict(X)
        return accuracy_score(y, y_pred)

    def gini_impurity(self, y):
        # Calculate Gini impurity for a given label distribution
        class_counts = np.bincount(y)
        proportions = class_counts / len(y)
        return 1 - np.sum(proportions ** 2)

    def __sklearn_tags__(self):
        # Update sklearn tags to ensure compatibility
        tags = super().__sklearn_tags__()
        tags.estimator_type = "classifier"
        tags.classifier_tags = ClassifierTags()
        tags.target_tags = TargetTags(required=True)
        tags.non_deterministic = False
        return tags
    
    def _get_default_requests(self):
        # Handle metadata requests correctly for cross-validation
        from sklearn.utils._metadata_requests import MetadataRequest
        return MetadataRequest(self)
