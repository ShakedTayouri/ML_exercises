import pandas as pd
import xlrd
print(xlrd.__version__)  # Should return 2.0.1 or higher


url = "https://archive.ics.uci.edu/ml/machine-learning-databases/concrete/compressive/Concrete_Data.xls"
concrete_df = pd.read_excel(url)
concrete_df.to_csv('concrete_strength.csv', index=False)\

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00242/ENB2012_data.xlsx"
energy_df = pd.read_excel(url)
energy_df.to_csv('energy_efficiency.csv', index=False)
