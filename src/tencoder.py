from sklearn.base import BaseEstimator, TransformerMixin
import category_encoders as ce

class TargetEncoderWrapper(BaseEstimator, TransformerMixin):
    def __init__(self, cols=None):
        self.cols = cols
        self.encoder = None

    def fit(self, X, y=None):
        self.encoder = ce.TargetEncoder(cols=self.cols)
        self.encoder.fit(X, y)
        return self

    def transform(self, X):
        return self.encoder.transform(X)
