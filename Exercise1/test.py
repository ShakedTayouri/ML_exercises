import pandas as pd

data = pd.read_csv(r'C:\Users\Noa\Documents\Studies\ML_exercises\Exercise1\datasets\for_regression\WineQuality_dataset.csv')
print(data.head())
print(data.columns)
print(data['quality'].head())

if 'quality' in data.columns:
    X, y = data.drop(columns=['quality']), data['quality']
else:
    print("Column 'quality' not found.")
