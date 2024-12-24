import pandas as pd

# Load both datasets
mat = pd.read_csv(r'C:\Users\Noa\Documents\Studies\ML_exercises\Exercise1\datasets\before_merging\student-mat.csv', delimiter=';')
por = pd.read_csv(r'C:\Users\Noa\Documents\Studies\ML_exercises\Exercise1\datasets\before_merging\student-por.csv', delimiter=';')

# Merge them, ensuring to keep unique records
merged_data = pd.concat([mat, por]).drop_duplicates().reset_index(drop=True)

print(merged_data.shape)  # Should output (1044, 30 or 33 depending on encoding)

merged_data.to_csv(r"C:\Users\Noa\Documents\Studies\ML_exercises\Exercise1\datasets\for_regression\merged_student_data.csv", index=False)
