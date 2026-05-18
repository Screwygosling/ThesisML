import pandas as pd


df = pd.read_csv(r"C:\Users\Reuther Felias\MLprojects\Thesis\crime_threat_ai\data\raw\crime_data.csv")

print(f"csv file shape: {df.shape}")
print(df.head())

#remove empty columns
df = df.dropna(axis=1, how="all")

print(f"csv columns: {len(df.columns)}")

df = df[[
    "dateCommited",
    "timeCommited",
    "barangay",
    "municipal",
    "street"
    "lat"
    "lng"
    "offense"
    "offenseType"
    "typeofPlace"
    "victimeCount"
    "suspectCount"
    "victimKilled"
    "victimInjured"
]]

