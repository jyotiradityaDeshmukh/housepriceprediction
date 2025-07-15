from sklearn.base import BaseEstimator, TransformerMixin

class FeatureEngineer(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass
    
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        df = X.copy()
        
        # Fill missing values
        none_cols = ['Alley', 'MasVnrType', 'BsmtQual', 'BsmtCond', 'BsmtExposure',
                     'BsmtFinType1', 'BsmtFinType2', 'FireplaceQu', 'GarageType',
                     'GarageFinish', 'GarageQual', 'GarageCond', 'PoolQC', 'Fence', 'MiscFeature']
        df[none_cols] = df[none_cols].fillna('None')

        zero_cols = ['MasVnrArea', 'GarageYrBlt','LotFrontage']
        df[zero_cols] = df[zero_cols].fillna(0)
        df['Electrical'] = df['Electrical'].fillna(df['Electrical'].mode()[0])

        # Feature engineering
        df['TotalSF'] = df['TotalBsmtSF'] + df['1stFlrSF'] + df['2ndFlrSF']
        df['TotalBath'] = (df['BsmtFullBath'] + df['FullBath'] + 
                           0.5 * (df['BsmtHalfBath'] + df['HalfBath']))
        df['HouseAge'] = df['YrSold'] - df['YearBuilt']
        df['RemodAge'] = df['YrSold'] - df['YearRemodAdd']
        df['GarageAge'] = df['YrSold'] - df['GarageYrBlt']
        df['GarageAge'] = df['GarageAge'].replace(0, df['GarageAge'].median())
        df['IsRemodeled'] = (df['YearBuilt'] != df['YearRemodAdd']).astype(int)
        df['OverallScore'] = df['OverallQual'] * df['OverallCond']
        df['TotalPorchSF'] = (df['OpenPorchSF'] + df['EnclosedPorch'] +
                              df['3SsnPorch'] + df['ScreenPorch'])
        df['HasPool'] = (df['PoolArea'] > 0).astype(int)
        df['Has2ndFloor'] = (df['2ndFlrSF'] > 0).astype(int)
        df['HasGarage'] = (df['GarageArea'] > 0).astype(int)
        df['HasBsmt'] = (df['TotalBsmtSF'] > 0).astype(int)
        df['HasFireplace'] = (df['Fireplaces'] > 0).astype(int)

        # Drop columns
        cols_to_drop = [
            'YearBuilt', 'YearRemodAdd', 'GarageYrBlt',
            'BsmtFullBath', 'BsmtHalfBath', 'FullBath', 'HalfBath',
            'OpenPorchSF', 'EnclosedPorch', '3SsnPorch', 'ScreenPorch',
            'TotalBsmtSF', '1stFlrSF', '2ndFlrSF',
            'PoolArea', 'GarageArea', 'GarageCars', 'Fireplaces'
        ]
        optional_drop = ['PoolQC', 'MiscFeature', 'Alley', 'Fence']
        df.drop(columns=cols_to_drop + optional_drop, inplace=True, errors='ignore')
        
        # Drop low-correlation features
        cdata = df.select_dtypes(include=['float64', 'int64'])
        if 'SalePrice' in cdata:
            corr = cdata.corr()['SalePrice'].sort_values(ascending=False)
            weak_corr_features = corr[(corr > -0.2) & (corr < 0.2)].index.tolist()
            weak_corr_features = [feat for feat in weak_corr_features if feat != 'SalePrice']
            df.drop(columns=weak_corr_features, inplace=True, errors='ignore')
        
        return df
