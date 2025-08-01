import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    # Extracts hour, day, month, year from a datetime column and drops the original column.
    def __init__(self, date_column='TransactionStartTime'):
        self.date_column = date_column

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        X_copy = X.copy()
        if self.date_column in X_copy.columns:
            X_copy[self.date_column] = pd.to_datetime(X_copy[self.date_column], errors='coerce')
            X_copy['TransactionHour'] = X_copy[self.date_column].dt.hour
            X_copy['TransactionDay'] = X_copy[self.date_column].dt.day
            X_copy['TransactionMonth'] = X_copy[self.date_column].dt.month
            X_copy['TransactionYear'] = X_copy[self.date_column].dt.year
            X_copy = X_copy.drop(columns=[self.date_column])
        return X_copy

class ColumnDropper(BaseEstimator, TransformerMixin):
    # Drops specified columns.
    def __init__(self, columns_to_drop):
        self.columns_to_drop = columns_to_drop

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X.drop(columns=self.columns_to_drop, errors='ignore')

class OutlierCapper(BaseEstimator, TransformerMixin):
    # Caps outliers in numerical columns using percentiles.
    def __init__(self, columns=None, upper_percentile=0.99, lower_percentile=0.01):
        self.columns = columns
        self.upper_percentile = upper_percentile
        self.lower_percentile = lower_percentile
        self.upper_bounds = {}
        self.lower_bounds = {}

    def fit(self, X, y=None):
        if self.columns is None:
            self.columns = X.select_dtypes(include=np.number).columns.tolist()
        for col in self.columns:
            if col in X.columns:
                self.upper_bounds[col] = X[col].quantile(self.upper_percentile)
                self.lower_bounds[col] = X[col].quantile(self.lower_percentile)
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.columns:
            if col in X_copy.columns and col in self.upper_bounds and col in self.lower_bounds:
                X_copy[col] = np.where(X_copy[col] > self.upper_bounds[col], self.upper_bounds[col], X_copy[col])
                X_copy[col] = np.where(X_copy[col] < self.lower_bounds[col], self.lower_bounds[col], X_copy[col])
        return X_copy

class CustomerAggregator(BaseEstimator, TransformerMixin):
    # Aggregates data to customer-level, computes aggregates and merges static features.

    def __init__(self, customer_id_col='CustomerId', amount_col='Amount', fraud_col='FraudResult'):
        self.customer_id_col = customer_id_col
        self.amount_col = amount_col
        self.fraud_col = fraud_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        customer_df = X.groupby(self.customer_id_col).agg(
            total_transaction_amount=(self.amount_col, 'sum'),
            average_transaction_amount=(self.amount_col, 'mean'),
            transaction_count=(self.amount_col, 'count'),
            std_transaction_amount=(self.amount_col, 'std'),
            customer_fraudulent=(self.fraud_col, 'max')
        ).reset_index()
        customer_df['std_transaction_amount'] = customer_df['std_transaction_amount'].fillna(0)
        static_customer_features = [
            'CountryCode', 'CurrencyCode', 'ProviderId', 'ProductId',
            'ProductCategory', 'ChannelId', 'PricingStrategy',
            'TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear'
        ]
        static_customer_features = [col for col in static_customer_features if col in X.columns]
        first_transaction_features = X.drop_duplicates(subset=[self.customer_id_col]).set_index(self.customer_id_col)[static_customer_features]
        customer_df = customer_df.merge(first_transaction_features, on=self.customer_id_col, how='left')
        return customer_df

class CustomWOETransformer(BaseEstimator, TransformerMixin):
    # Custom Weight of Evidence (WoE) transformer for categorical and numerical features.
    def __init__(self, features=None, target_column='customer_fraudulent', numerical_bins=10, woe_cap=20):
        self.features = features
        self.target_column = target_column
        self.numerical_bins = numerical_bins
        self.woe_cap = woe_cap
        self.woe_maps = {}
        self.iv_values = {}
        self.feature_is_numeric = {}
        self.numerical_bin_edges = {}

    def fit(self, X, y=None):
        if self.features is None:
            self.features = X.select_dtypes(include=['object', 'category', 'int64', 'float64']).columns.tolist()
            if self.target_column in self.features:
                self.features.remove(self.target_column)
        if y is None:
            raise ValueError("Target variable 'y' must be provided for WOE transformer fitting.")
        X = X.reset_index(drop=True)
        y = pd.Series(y).reset_index(drop=True)
        for col in self.features:
            if col not in X.columns:
                print(f"Warning: Column '{col}' not found in X for WOE calculation. Skipping.")
                continue
            temp_df = pd.DataFrame({col: X[col], 'target': y})
            temp_df['target'] = temp_df['target'].astype(int)
            if pd.api.types.is_numeric_dtype(temp_df[col]):
                self.feature_is_numeric[col] = True
                try:
                    temp_df_no_nan = temp_df.dropna(subset=[col])
                    temp_df_no_nan['binned_col'], self.numerical_bin_edges[col] = pd.qcut(
                        temp_df_no_nan[col], q=self.numerical_bins, labels=False, duplicates='drop', retbins=True
                    )
                    temp_df = temp_df.merge(
                        temp_df_no_nan[['binned_col']], left_index=True, right_index=True, how='left'
                    )
                except Exception as e:
                    print(f"Warning: pd.qcut failed for numerical column '{col}' ({e}). Falling back to pd.cut.")
                    try:
                        n_unique = temp_df[col].nunique()
                        effective_bins = min(self.numerical_bins, n_unique)
                        if effective_bins == 0:
                            temp_df['binned_col'] = np.nan
                        else:
                            temp_df['binned_col'], self.numerical_bin_edges[col] = pd.cut(
                                temp_df[col], bins=effective_bins, labels=False, include_lowest=True, duplicates='drop', retbins=True
                            )
                    except Exception as e_cut:
                        print(f"Error: pd.cut also failed for numerical column '{col}' ({e_cut}). Column will be skipped for WOE.")
                        self.feature_is_numeric[col] = False
                        continue
                group_col = 'binned_col'
            else:
                self.feature_is_numeric[col] = False
                group_col = col
            grouped = temp_df.groupby(group_col)['target'].agg(
                total_count='count',
                bad_count=lambda x: (x == 1).sum(),
                good_count=lambda x: (x == 0).sum()
            ).reset_index()
            total_bad = grouped['bad_count'].sum()
            total_good = grouped['good_count'].sum()
            epsilon = 1e-6
            grouped['bad_rate'] = grouped['bad_count'] / (total_bad + epsilon)
            grouped['good_rate'] = grouped['good_count'] / (total_good + epsilon)
            grouped['woe'] = np.log((grouped['bad_rate'] + epsilon) / (grouped['good_rate'] + epsilon))
            grouped['woe'] = np.clip(grouped['woe'], -self.woe_cap, self.woe_cap)
            grouped['iv_contribution'] = (grouped['bad_rate'] - grouped['good_rate']) * grouped['woe']
            self.iv_values[col] = grouped['iv_contribution'].sum()
            self.woe_maps[col] = grouped.set_index(group_col)['woe'].to_dict()
        return self

    def transform(self, X):
        X_copy = X.copy()
        for col in self.features:
            if col not in X_copy.columns:
                continue
            if col in self.woe_maps:
                if self.feature_is_numeric.get(col, False):
                    if col not in self.numerical_bin_edges:
                        X_copy[col] = 0
                        continue
                    binned_col_series = pd.cut(
                        X_copy[col], bins=self.numerical_bin_edges[col], labels=False,
                        include_lowest=True, right=True, duplicates='drop'
                    )
                    X_copy[col] = binned_col_series.map(self.woe_maps[col])
                else:
                    X_copy[col] = X_copy[col].map(self.woe_maps[col])
                X_copy[col] = X_copy[col].fillna(0)
            else:
                if pd.api.types.is_numeric_dtype(X_copy[col]):
                    X_copy[col] = X_copy[col].fillna(0)
                else:
                    X_copy[col] = 0
        return X_copy

def process_data(df_raw: pd.DataFrame, target_column: str = 'FraudResult') -> pd.DataFrame:
    # Runs the data processing pipeline and returns a model-ready DataFrame.
    if df_raw is None:
        print("Raw DataFrame is None. Cannot process data.")
        return None
    print("Step 1: Initial Transaction-Level Preprocessing...")
    transaction_level_cols_to_drop = [
        'Value', 'TransactionId', 'BatchId', 'AccountId', 'SubscriptionId'
    ]
    pre_agg_pipeline = Pipeline([
        ('datetime_extractor', DateTimeFeatureExtractor(date_column='TransactionStartTime')),
        ('column_dropper_initial', ColumnDropper(columns_to_drop=transaction_level_cols_to_drop))
    ])
    df_preprocessed = pre_agg_pipeline.fit_transform(df_raw.copy())
    print("Initial preprocessing complete. Shape:", df_preprocessed.shape)
    print("\nStep 2: Aggregating data to Customer-Level...")

    customer_aggregator = CustomerAggregator(fraud_col=target_column)
    df_customer_level = customer_aggregator.fit_transform(df_preprocessed)
    print("Customer-level aggregation complete. Shape:", df_customer_level.shape)

    y_customer = df_customer_level['customer_fraudulent']
    X_customer = df_customer_level.drop(columns=['customer_fraudulent', 'CustomerId'])
    print("\nStep 3: Handling Outliers on Numerical Features...")


    numerical_features_after_agg = X_customer.select_dtypes(include=np.number).columns.tolist()
    cappable_numerical_features = [
        col for col in numerical_features_after_agg
        if col not in ['TransactionHour', 'TransactionDay', 'TransactionMonth', 'TransactionYear']
    ]


    outlier_capper = OutlierCapper(columns=cappable_numerical_features)
    X_customer_capped = outlier_capper.fit_transform(X_customer)
    print("Outlier capping complete.")
    print("\nStep 4: Applying WOE Transformation to Categorical Features...")

    categorical_features_for_woe = X_customer_capped.select_dtypes(include='object').columns.tolist()

    for col in ['CountryCode', 'PricingStrategy']:
        if col in X_customer_capped.columns and col not in categorical_features_for_woe:
            categorical_features_for_woe.append(col)

    if XVERSE_WOE_AVAILABLE:
        woe_transformer_xverse = WOE(col_names=categorical_features_for_woe, df=X_customer_capped, target_column=y_customer.name)
        woe_transformer_xverse.fit()
        X_customer_woe = woe_transformer_xverse.transform(X_customer_capped)
        print("Information Value (IV) for xverse WoE transformed features:")


        for col, iv_df in woe_transformer_xverse.iv_values.items():
            if 'IV' in iv_df.columns:
                print(f"  {col}: {iv_df['IV'].iloc[0]:.4f}")
            else:
                 print(f"  {col}: IV calculation not directly available or in a different format from xverse output.")
   
    else:
        woe_transformer_custom = CustomWOETransformer(features=categorical_features_for_woe, target_column='customer_fraudulent')
        woe_transformer_custom.fit(X_customer_capped, y_customer)
        X_customer_woe = woe_transformer_custom.transform(X_customer_capped)
        print("Information Value (IV) for Custom WoE transformed features:")
       
        for col, iv in woe_transformer_custom.iv_values.items():
            print(f"  {col}: {iv:.4f}")

    print("WOE transformation complete.")
    print("\nStep 5: Scaling all Numerical Features...")

    X_customer_woe_numeric = X_customer_woe.select_dtypes(include=np.number)

    if not np.isfinite(X_customer_woe_numeric).all().all():
        print("Warning: Non-finite values detected after WOE transformation. Attempting to clean before scaling.")
        X_customer_woe_numeric = X_customer_woe_numeric.replace([np.inf, -np.inf], np.nan)
        X_customer_woe_numeric = X_customer_woe_numeric.fillna(0)

    final_features_for_scaling = X_customer_woe_numeric.columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_customer_woe_numeric[final_features_for_scaling])
    X_processed = pd.DataFrame(X_scaled, columns=final_features_for_scaling, index=X_customer_woe.index)
    X_processed['customer_fraudulent'] = y_customer.values
    print("Scaling complete. Data is now model-ready.")

    return X_processed

if __name__ == '__main__':
    data = {
        'TransactionId': ['T1', 'T2', 'T3', 'T4', 'T5', 'T6', 'T7', 'T8', 'T9', 'T10'],
        'BatchId': ['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7', 'B8', 'B9', 'B10'],
        'AccountId': ['A1', 'A1', 'A2', 'A2', 'A3', 'A3', 'A4', 'A4', 'A5', 'A5'],
        'SubscriptionId': ['S1', 'S1', 'S2', 'S2', 'S3', 'S3', 'S4', 'S4', 'S5', 'S5'],
        'CustomerId': ['C1', 'C1', 'C2', 'C2', 'C3', 'C3', 'C4', 'C4', 'C5', 'C5'],
        'CurrencyCode': ['USD', 'USD', 'EUR', 'USD', 'EUR', 'GBP', 'USD', 'EUR', 'USD', 'GBP'],
        'CountryCode': [1, 1, 2, 1, 2, 3, 1, 2, 1, 3],
        'ProviderId': ['P1', 'P1', 'P2', 'P1', 'P2', 'P3', 'P1', 'P2', 'P1', 'P3'],
        'ProductId': ['ProdA', 'ProdA', 'ProdB', 'ProdA', 'ProdB', 'ProdC', 'ProdA', 'ProdB', 'ProdA', 'ProdC'],
        'ProductCategory': ['CatX', 'CatX', 'CatY', 'CatX', 'CatY', 'CatZ', 'CatX', 'CatY', 'CatX', 'CatZ'],
        'ChannelId': ['Ch1', 'Ch1', 'Ch2', 'Ch1', 'Ch2', 'Ch3', 'Ch1', 'Ch2', 'Ch1', 'Ch3'],
        'Amount': [100.5, 200.0, 50.0, 15000.0, 75.0, 300.0, 120.0, 60.0, 180.0, 350.0],
        'Value': [100.5, 200.0, 50.0, 15000.0, 75.0, 300.0, 120.0, 60.0, 180.0, 350.0],
        'TransactionStartTime': [
            '2023-01-01 10:00:00', '2023-01-01 11:30:00', '2023-01-02 14:00:00',
            '2023-01-03 09:00:00', '2023-01-03 16:00:00', '2023-01-04 10:00:00',
            '2023-01-05 12:00:00', '2023-01-05 13:00:00', '2023-01-06 17:00:00', '2023-01-06 18:00:00'
        ],
        'PricingStrategy': [1, 1, 2, 1, 2, 3, 1, 2, 1, 3],
        'FraudResult': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    }

    dummy_df = pd.DataFrame(data)
    print("--- Running data processing with dummy data ---")

    processed_df = process_data(dummy_df.copy(), target_column='FraudResult')
    
    if processed_df is not None:
        print("\n--- Processed Data (first 5 rows) ---")
        print(processed_df.head())
        print("\n--- Processed Data Info ---")
        processed_df.info()
        print("\n--- Processed Data Describe ---")
        print(processed_df.describe())
        print("\nUnique values in customer_fraudulent (target):", processed_df['customer_fraudulent'].unique())