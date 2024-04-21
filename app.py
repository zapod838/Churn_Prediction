import mlflow
import pandas as pd
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, FeatureUnion
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, FunctionTransformer, OneHotEncoder, OrdinalEncoder, Normalizer
from sklearn.base import BaseEstimator, TransformerMixin

# Set the full path to the model within MLflow's artifacts directory
model_path ="mlruns/0/1c2d058efc314794a71debd91871de44/artifacts/churn_prediction_pipeline"


# Load the model
loaded_model = mlflow.pyfunc.load_model(model_path)

#importing data
train_df = pd.read_csv("train.csv")
test_df = pd.read_csv("test.csv")

# Drop 'Churn' and 'CustomerID' columns safely
train_df = train_df.drop(columns=['Churn', 'CustomerID'], errors='ignore')
test_df = test_df.drop(columns=['CustomerID'], errors='ignore')

# Custom transformer for encoding categorical columns
class CustomEncoder(BaseEstimator, TransformerMixin):
    def __init__(self, categorical_columns, subscription_type_ordering):
        self.categorical_columns = categorical_columns
        self.subscription_type_ordering = subscription_type_ordering
        self.onehot_encoder = OneHotEncoder(sparse_output=False)
        self.encoder = OrdinalEncoder(categories=[subscription_type_ordering])

    def fit(self, X, y=None):
        # Fit one-hot encoder
        self.onehot_encoder.fit(X[self.categorical_columns])
        # Fit ordinal encoder
        self.encoder.fit(X[['SubscriptionType']])
        return self

    def transform(self, X):
        # One-hot encode the categorical columns
        encoded_categorical = self.onehot_encoder.transform(X[self.categorical_columns])
        # Convert the categorical encoded data into a DataFrame with column names
        encoded_categorical_df = pd.DataFrame(
            encoded_categorical,
            columns=self.onehot_encoder.get_feature_names_out(),
            index=X.index
        )
        # Encode 'SubscriptionType' using OrdinalEncoder
        subscription_type_encoded = self.encoder.transform(X[['SubscriptionType']])
        # Convert the encoded 'SubscriptionType' into a DataFrame
        subscription_type_encoded_df = pd.DataFrame(
            subscription_type_encoded,
            columns=['SubscriptionTypeEncoded'],
            index=X.index
        )
        # Drop the original categorical columns and concatenate the one-hot encoded columns
        X_encoded  = X.drop(self.categorical_columns + ['SubscriptionType'], axis=1, errors='ignore')
        # Check if 'Churn' column exists, if so, pass it through
        if 'Churn' in X.columns:
            X_encoded['Churn'] = X['Churn']
        X_encoded  = pd.concat([X_encoded , encoded_categorical_df, subscription_type_encoded_df], axis=1)
        return X_encoded

class ToDataFrame(BaseEstimator, TransformerMixin):
    def __init__(self, column_names):
        self.column_names = column_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.column_names)


class FeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_names):
        self.feature_names = feature_names

    def fit(self, X, y=None):
        # No fitting necessary for feature selection
        if isinstance(X, pd.DataFrame):
            # If X is a DataFrame, store the column indices that correspond to the feature_names
            self.feature_indices_ = [X.columns.get_loc(name) for name in self.feature_names]
        else:
            # Assuming the order of columns in the NumPy array matches the order of feature names
            # Feature indices will simply be a range if X is not a DataFrame at fit time
            self.feature_indices_ = list(range(len(self.feature_names)))
        return self

    def transform(self, X, y=None):
        # Handle both DataFrame and NumPy array
        if isinstance(X, pd.DataFrame):
            return X[self.feature_names]
        elif isinstance(X, np.ndarray):
            # If X is an array, use the feature indices that were stored during fit
            return X[:, self.feature_indices_]
        else:
            raise TypeError("Input must be a pandas DataFrame or a numpy ndarray.")
        
# Define the order of the categories
subscription_type_ordering = ['Basic', 'Standard', 'Premium']

# List of categorical columns to be one-hot encoded
categorical_columns = ['PaymentMethod', 'PaperlessBilling', 'ContentType', 'MultiDeviceAccess', 'DeviceRegistered','GenrePreference', 'Gender', 'ParentalControl', 'SubtitlesEnabled']

# List of numerical columns to be normalized
numerical_columns = ['AccountAge', 'MonthlyCharges', 'TotalCharges', 'ViewingHoursPerWeek', 'AverageViewingDuration' ,'ContentDownloadsPerMonth', 'UserRating', 'SupportTicketsPerMonth','WatchlistSize']

pca_features = ['MonthlyCharges', 'SupportTicketsPerMonth', 'UserRating', 'WatchlistSize', 'AccountAge', 'TotalCharges', 'ViewingHoursPerWeek', 'ContentDownloadsPerMonth', 'AverageViewingDuration']

# Using FunctionTransformer to create a no-op (no operation) pipeline step
encoded_features = ['SubscriptionTypeEncoded','PaymentMethod_Bank transfer','PaymentMethod_Credit card', 'PaymentMethod_Electronic check','PaymentMethod_Mailed check','PaperlessBilling_No', 'PaperlessBilling_Yes','ContentType_Both','ContentType_Movies', 
'ContentType_TV Shows','MultiDeviceAccess_No','MultiDeviceAccess_Yes', 'DeviceRegistered_Computer','DeviceRegistered_Mobile', 'DeviceRegistered_TV', 'DeviceRegistered_Tablet','GenrePreference_Action','GenrePreference_Comedy', 'GenrePreference_Drama', 'GenrePreference_Fantasy','GenrePreference_Sci-Fi', 'Gender_Female','Gender_Male','ParentalControl_No', 'ParentalControl_Yes','SubtitlesEnabled_No','SubtitlesEnabled_Yes']    

output_column_names = ['AccountAge', 'MonthlyCharges', 'TotalCharges', 'ViewingHoursPerWeek', 'AverageViewingDuration', 'ContentDownloadsPerMonth', 'UserRating', 'SupportTicketsPerMonth', 'WatchlistSize', 'PaymentMethod_Bank transfer', 'PaymentMethod_Credit card', 'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check', 'PaperlessBilling_No', 'PaperlessBilling_Yes', 'ContentType_Both', 'ContentType_Movies', 'ContentType_TV Shows', 'MultiDeviceAccess_No', 'MultiDeviceAccess_Yes', 'DeviceRegistered_Computer', 'DeviceRegistered_Mobile', 'DeviceRegistered_TV', 'DeviceRegistered_Tablet', 'GenrePreference_Action', 'GenrePreference_Comedy', 'GenrePreference_Drama', 'GenrePreference_Fantasy', 'GenrePreference_Sci-Fi', 'Gender_Female', 'Gender_Male', 'ParentalControl_No', 'ParentalControl_Yes', 'SubtitlesEnabled_No','SubtitlesEnabled_Yes','SubscriptionTypeEncoded'] 

# Main function to build and return the pipeline
def build_pipeline():    
    # Define the pipeline
    categorical_transformer = Pipeline(steps=[('encoder', CustomEncoder(categorical_columns, subscription_type_ordering))])
    # Define the pipeline for numerical columns
    numerical_transformer = Pipeline(steps=[('normalizer', Normalizer())])
    # Combine categorical and numerical transformers
    preprocessor = ColumnTransformer(transformers=[('num', numerical_transformer, numerical_columns),
                                                   ('cat', categorical_transformer, categorical_columns + ['SubscriptionType'])])
    
    pca_pipeline = Pipeline(steps=[('feature_selector', FeatureSelector(feature_names=pca_features)),
                                   ('scaler', StandardScaler()), ('pca', PCA(n_components=9)) ])

    # No-op pipeline for encoded features
    pass_through_pipeline = Pipeline([('identity', FunctionTransformer())])

    combined_features_pipeline = ColumnTransformer(transformers=[('pca_features', pca_pipeline, pca_features),
                                                                 ('encoded_features', pass_through_pipeline, encoded_features)], remainder='drop')
    # Testing pipeline (without resampling)
    full_pipeline = Pipeline([('preprocessor', preprocessor),('to_df', ToDataFrame(column_names=output_column_names)),
                              ('combine_features', combined_features_pipeline)]) 
        
    return full_pipeline

pipeline = build_pipeline()

from sklearn import set_config
set_config(display='diagram')
# Fit the full pipeline to the training data
pipeline.fit(train_df) 

def predict_churn(df):
    # Preprocess data
    processed_df = pipeline.transform(df)
     
    preprocessed_df = pd.DataFrame(processed_df, columns=output_column_names)
    
    # Make the prediction
    predictions = loaded_model.predict(preprocessed_df)
    return predictions

st.title('Churn Prediction App')

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type="csv")
if uploaded_file is not None:
    # Load the data
    data = pd.read_csv(uploaded_file)
    
    # Check if the data has the expected number of columns (20)
    if data.shape[1] == 20:
        # Make predictions
        data['Churn'] = predict_churn(data)
        
        # Convert DataFrame to CSV
        csv = data.to_csv(index=False).encode('utf-8')
        
        # Download link
        st.download_button(
            label="Download CSV with Churn Predictions",
            data=csv,
            file_name='churn_predictions.csv',
            mime='text/csv',
        )
    else:
        st.error("The uploaded file does not have the expected number of columns (20).")