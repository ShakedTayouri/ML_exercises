import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

path_to_datasets = r"C:\Users\Noa\Documents\Studies\ML_exercises\Exercise1\datasets\for_regression"

# Load the dataset
data = pd.read_csv(f'{path_to_datasets}\\MergedStudentPerformance_dataset.csv', delimiter=',')

data.columns = data.columns.str.strip()

# Identify categorical columns (you can adjust this list as needed)
categorical_columns = ['school', 'sex', 'address', 'famsize', 'Pstatus', 'Mjob', 'Fjob', 'reason', 'guardian', 
                       'traveltime', 'studytime', 'failures', 'schoolsup', 'famsup', 'paid', 'activities', 'nursery', 
                       'higher', 'internet', 'romantic', 'famrel', 'freetime', 'goout', 'Dalc', 'Walc', 'health', 'absences']

# Handling missing values (replace '?' with NaN and then impute)
data.replace('?', pd.NA, inplace=True)
imputer = SimpleImputer(strategy='most_frequent')  # Using most frequent value to impute missing values
data_imputed = pd.DataFrame(imputer.fit_transform(data), columns=data.columns)

# Label Encoding for ordinal variables (like 'sex' and others)
label_encoder = LabelEncoder()
data_imputed['sex'] = label_encoder.fit_transform(data_imputed['sex'])  # For example, Male=0, Female=1

# One-Hot Encoding for nominal variables
data_encoded = pd.get_dummies(data_imputed, columns=categorical_columns, drop_first=True)

# Separate features and target
X = data_encoded.drop(columns=['G3'])  # Features (excluding the target column 'G3')
y = data_encoded['G3']  # Target

data_encoded = data_encoded.astype({col: 'int' for col in data_encoded.select_dtypes('bool').columns})

# Save the processed dataset
data_encoded.to_csv(r"C:\Users\Noa\Documents\Studies\ML_exercises\Exercise1\datasets\for_regression\MergedStudentPerformance_dataset_encoding.csv", index=False)
