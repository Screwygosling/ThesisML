import pandas as pd

df = pd.read_excel(r"C:\Users\Reuther Felias\MLprojects\Thesis\crime_threat_ai\data\raw\crime_data.csv")

print(df.head())
print(df.columns)
print(len(df))

print(df.info())
print(df.isnull().sum())




