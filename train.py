import pandas as pd
import numpy as np
import joblib
from catboost import CatBoostClassifier
from sklearn.model_selection import train_test_split

def load_and_prepare_data():

    # Load all dataframes
    contract = pd.read_csv('contract.csv', na_values=['', ' ', 'N/A', 'null', 'None', '?'], keep_default_na=True)
    personal = pd.read_csv('personal.csv', na_values=['', ' ', 'N/A', 'null', 'None', '?'], keep_default_na=True)
    internet = pd.read_csv('internet.csv', na_values=['', ' ', 'N/A', 'null', 'None', '?'], keep_default_na=True)
    phone = pd.read_csv('phone.csv', na_values=['', ' ', 'N/A', 'null', 'None', '?'], keep_default_na=True)

    # Clean contract df
    contract['BeginDate'] = pd.to_datetime(contract['BeginDate'], format='%Y-%m-%d', errors='coerce')
    contract['EndDate'] = pd.to_datetime(contract['EndDate'], format='%Y-%m-%d 00:00:00', errors='coerce')

    # Churned column
    contract['Churned'] = contract['EndDate'].notna().astype(int)

    # Tenure column
    snapshot_date = contract['EndDate'].max()

    temp_end = contract['EndDate'].fillna(snapshot_date)

    contract['TenureMonths'] = (
        (temp_end.dt.year - contract['BeginDate'].dt.year) * 12 +
        (temp_end.dt.month - contract['BeginDate'].dt.month)
    )

    # Fill TotalCharges nulls
    contract['TotalCharges'] = contract['TotalCharges'].fillna(contract['MonthlyCharges'])

    # Clean personal df
    personal.rename(columns={'gender':'Gender'}, inplace=True)

    # Merging datasets
    df_merged = contract.merge(personal, on='customerID', how='left').merge(internet, on='customerID', how='left').merge(phone, on='customerID', how='left')

    # HasInternet column
    df_merged['HasInternet'] = df_merged['InternetService'].notna().astype(int)

    # Fill internet nulls
    internet_features = internet.columns.to_numpy()
    internet_features = np.delete(internet_features, 0)

    internet_cols = internet_features.tolist()
    df_merged[internet_cols] = df_merged[internet_cols].fillna('No')

    # HasPhone column
    df_merged['HasPhone'] = df_merged['MultipleLines'].notna().astype(int)

    # Fill phone nulls
    df_merged['MultipleLines'] = df_merged['MultipleLines'].fillna('No')

    # Dropping unnecesary fields
    fields_to_drop = ['customerID', 'BeginDate', 'EndDate']
    df_final = df_merged.drop(fields_to_drop, axis=1)

    return df_final

def train():
    df = load_and_prepare_data()

    # Set object type to category in non-encoded df
    categorical_features = df.columns[df.dtypes=='object']

    for col in categorical_features:
        df[col] = df[col].astype("category")

    #Split
    X = df.drop('Churned', axis=1)
    y = df['Churned']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    # Let the model know which columns are categorical 
    cat_features = X_train.select_dtypes(include=['category']).columns.tolist()
    extra_cats = ['SeniorCitizen', 'HasInternet', 'HasPhone']
    cat_features.extend([col for col in extra_cats if col in X_train.columns])

    best_model = CatBoostClassifier(auto_class_weights='Balanced',
                               depth=4,
                               learning_rate=0.03,
                               l2_leaf_reg=5, 
                               random_state=42, 
                               verbose=0)

    best_model.fit(X_train, y_train, cat_features=cat_features)

    joblib.dump(best_model, "churn_model.pkl")
    joblib.dump(X.columns.tolist(), "features_names.pkl")
    joblib.dump(cat_features, "cat_features.pkl")

if __name__ == "__main__":
    train()