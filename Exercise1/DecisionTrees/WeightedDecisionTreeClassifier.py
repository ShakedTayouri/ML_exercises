from sklearn.tree import DecisionTreeClassifier
from sklearn.utils._tags import Tags, ClassifierTags, TargetTags
import numpy as np
from sklearn.metrics import accuracy_score
import warnings
import pandas as pd
from SoftDecisionTreeClassifier import SoftDecisionTreeClassifier


class WeightedDecisionTreeClassifier(SoftDecisionTreeClassifier):
    def __init__(self, criterion='gini', max_depth=None, min_samples_leaf=1, min_samples_split=2, random_state=None, max_features=None, alpha=0.1, n_samples=100):
        super().__init__(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                         min_samples_split=min_samples_split, random_state=random_state, criterion=criterion, max_features=max_features, alpha=alpha, n_samples=n_samples)
        self.weights_ = None
    
    def fit(self, X, y):
        X = np.array(X, dtype=np.float32)
        super().fit(X, y)

        # Initialize weights and selection counters for each leaf
        self.weights_ = []
        self.selection_counts_ = []  
        
        for i in range(self.tree_.node_count):
            if self.tree_.children_left[i] == -1:  # Leaf node
                leaf_y = y[self.tree_.apply(X) == i]
                
                # Class distribution and Gini impurity
                class_counts = np.bincount(leaf_y, minlength=len(self.classes_))
                probs = class_counts / np.sum(class_counts)
                gini = self.gini_impurity(leaf_y)
                
                # Initialize selection counter (all start at 0)
                self.selection_counts_.append(np.zeros_like(probs))
                
                # Store proportions and Gini for use during prediction
                self.weights_.append((probs, class_counts, gini))
            else:
                self.weights_.append(None)
                self.selection_counts_.append(None)
        return self


    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            X = X.values

        X = np.array(X, dtype=np.float32)
        leaf_indices = self.tree_.apply(X)

        predictions = []
        for idx in leaf_indices:
            if self.weights_[idx] is not None:
                probs, _, gini = self.weights_[idx]
                counts = self.selection_counts_[idx]

                if gini < 0.7:  # Low impurity, predict majority class directly
                    chosen_class = np.argmax(probs)
                else:
                    # High impurity, rotate through classes based on selection counts
                    adjusted_probs = probs / (counts + 1)
                    chosen_class = np.argmax(adjusted_probs)
                    
                    # Update selection count for the chosen class
                    self.selection_counts_[idx][chosen_class] += 1

                predictions.append(chosen_class)
            else:
                predictions.append(0)
        
        return np.array(predictions)



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