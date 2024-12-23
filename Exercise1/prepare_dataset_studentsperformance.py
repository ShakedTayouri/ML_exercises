import pandas as pd
from sklearn.preprocessing import LabelEncoder


path_to_datasets = r"C:\Users\Noa\Documents\Studies\ML_exercises\Exercise1\datasets\for_regression"

# Load the dataset
data = pd.read_csv(f'{path_to_datasets}\\StudentsPerformance_dataset.csv', delimiter=',')

# 1. Manual Mapping for Ordinal Features
education_order = ["some high school", "high school", "some college",
                   "associate's degree", "bachelor's degree", "master's degree"]

lunch_order = ["free/reduced", "standard"]
prep_course_order = ["none", "completed"]

# Apply mapping
data['parental level of education'] = data['parental level of education'].apply(lambda x: education_order.index(x))
data['lunch'] = data['lunch'].apply(lambda x: lunch_order.index(x))
data['test preparation course'] = data['test preparation course'].apply(lambda x: prep_course_order.index(x))

# 2. One-Hot Encoding for Nominal Features
data = pd.get_dummies(data, columns=['gender', 'race/ethnicity'], drop_first=True)
data = data.astype(int)

print(data.head())

data = data.astype({col: 'int' for col in data.select_dtypes('bool').columns})

data.to_csv(r"C:\Users\Noa\Documents\Studies\ML_exercises\Exercise1\datasets\for_regression\StudentsPerformance_dataset_encoding.csv", index=False)