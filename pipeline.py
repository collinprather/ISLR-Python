'''
Import necessary Libraries
'''
import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import OneHotEncoder, StandardScaler

###############################################################
###############################################################
'''
Define Custom Transformer classes to automate data preprocessing
'''
from sklearn.base import BaseEstimator, TransformerMixin

class TypeSelector(BaseEstimator, TransformerMixin):
    '''
    Selects all columns of the indicated type (usually numerical or categorical).
    
    Returns: pd.DataFrame of the columns of the indicated type
    '''
    def __init__(self, dtype):
        self.dtype = dtype
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.select_dtypes(include=[self.dtype])
    
class StringIndexer(BaseEstimator, TransformerMixin):
    '''
    Given a pd.DataFrame of categorical variables, this class encodes them as numerical
    
    Returns pd.DataFrame of encoded categories.
    '''
    def __init__(self):
        self = self
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        assert isinstance(X, pd.DataFrame)
        return X.apply(lambda s: s.cat.codes.replace({-1: len(s.cat.categories)}))


###############################################################
###############################################################
'''
Convert categorical columns into the 'category' datatype for appropriate processing
'''
for col in hitters[categorical_col_index]:
    hitters[col] = hitters[col].astype('category')

###############################################################
###############################################################
'''
Data Pre-processing Pipeline.
'''
numericals = Pipeline([
    ('selector', TypeSelector(np.number)),
            ('scaler', StandardScaler()),
            ('impute', Imputer())
])
categoricals= Pipeline([
    ('selector', TypeSelector('category')),
            ('encoder', StringIndexer()),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])
feat_union = FeatureUnion([('num_pipe', numericals), ('cat_pipe', categoricals)])
steps = [('preproc', feat_union)]
pipeline = Pipeline(steps)
###############################################################
###############################################################