import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
import category_encoders as ce
from feature_engineering import FeatureEngineer

test_data = pd.read_csv('data/test.csv')
pipeline = joblib.load('./models/full_model_pipeline.pkl')

# Predict (NO NEED to manually apply feature engineering or encoding!)
predictions = pipeline.predict(test_data)
pred = pd.DataFrame(predictions, columns=['predictions'])
pred.insert(0, 'Id', pred.index+1461)
pred.rename(columns={'predictions': 'SalePrice'}, inplace=True)
print(pred.head())
print(pred.shape)
pred.to_csv('data/submission.csv', index=False)