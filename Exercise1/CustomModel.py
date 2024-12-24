from sklearn.base import BaseEstimator, ClassifierMixin

class CustomModel(BaseEstimator, ClassifierMixin):
    def __sklearn_tags__(self):
        # Define custom sklearn tags
        return {
            'non_deterministic': True,
            'X_types': ['2darray'],
            'requires_fit': True,
            'multioutput': False,
            'preserves_dtype': [float],
            'allow_nan': False,
        }
