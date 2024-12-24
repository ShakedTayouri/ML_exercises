import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

path_to_datasets = r"C:\Users\Noa\Documents\Studies\ML_exercises\Exercise1\datasets\for_regression"

# Load the dataset
data = pd.read_csv(f'{path_to_datasets}\\CaliforniaHousingPrices_dataset.csv', delimiter=',')

# Separate numerical and categorical columns
numerical_columns = data.select_dtypes(include=['float64', 'int64']).columns
categorical_columns = data.select_dtypes(include=['object']).columns

# Impute numerical columns with the mean strategy
numerical_imputer = SimpleImputer(strategy='mean')
data[numerical_columns] = numerical_imputer.fit_transform(data[numerical_columns])

# Impute categorical columns with the most frequent strategy
categorical_imputer = SimpleImputer(strategy='most_frequent')
data[categorical_columns] = categorical_imputer.fit_transform(data[categorical_columns])

# One-hot encode categorical columns (ocean_proximity)
data_encoded = pd.get_dummies(data, columns=['ocean_proximity'], drop_first=True)

# Separate features (X) and target (y)
X = data_encoded.drop(columns=['median_house_value'])  # Features
y = data_encoded['median_house_value']  # Target

data_encoded = data_encoded.astype({col: 'int' for col in data_encoded.select_dtypes('bool').columns})

# Save the processed dataset
data_encoded.to_csv(r"C:\Users\Noa\Documents\Studies\ML_exercises\Exercise1\datasets\for_regression\CaliforniaHousingPrices_dataset_encoding.csv", index=False)
