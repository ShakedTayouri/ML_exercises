import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

path_to_datasets = r"C:\Users\Noa\Documents\Studies\ML_exercises\Exercise1\datasets\for_regression"

# Load the dataset
data = pd.read_csv(f'{path_to_datasets}\\AdultIncome_dataset.csv', delimiter=',')

# Handle missing values (replace '?' with NaN and then impute)
data.replace('?', np.nan, inplace=True)  # Replace '?' with np.nan
imputer = SimpleImputer(strategy='most_frequent')  # Using most frequent value to impute
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Encode categorical columns using label encoding (for binary or ordinal variables like income, gender)
label_encoder = LabelEncoder()
data_imputed['income'] = label_encoder.fit_transform(data_imputed['income'])  # Convert income to 0 and 1
data_imputed['gender'] = label_encoder.fit_transform(data_imputed['gender'])  # Male=0, Female=1

# One-hot encode nominal categorical columns like workclass, education, etc.
data_encoded = pd.get_dummies(data_imputed, columns=['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'native-country'], drop_first=True)

# Separate features and target
X = data_encoded.drop(columns=['income'])  # Features
y = data_encoded['income']  # Target

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize numerical features (optional but often helpful)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


data_encoded = data_encoded.astype({col: 'int' for col in data_encoded.select_dtypes('bool').columns})


# Save the processed dataset
data_encoded.to_csv(r"C:\Users\Noa\Documents\Studies\ML_exercises\Exercise1\datasets\for_regression\AdultIncome_dataset_encoding.csv", index=False)
