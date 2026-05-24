"""
retrain.py — drop this in your project root and run it locally:
    python retrain.py

It reads  data/processed/processed_crime_data.csv
Writes    models/crime_penalty_model.pkl   (replaces existing)
          data/processed/training_crime_data.csv

Changes from original:
  - crime_penalty now scaled 0-100 with real spread
  - areaCrimeCount per barangay aggregated and injected
  - severity weights kept from original notebook
  - same Random Forest, same features — drop-in replacement
"""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import pickle
import os

# ── paths ─────────────────────────────────────────────────────────────────────
BASE = r"C:\Users\Reuther Felias\MLprojects\thesis\crime_threat_ai"
INPUT_CSV = os.path.join(BASE, 'data', 'processed', 'processed_crime_data.csv')
OUT_CSV   = os.path.join(BASE, 'data', 'processed', 'training_crime_data.csv')
MODEL_OUT = os.path.join(BASE, 'models', 'crime_penalty_model.pkl')

# ── load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv(INPUT_CSV)
print(f"Loaded {len(df)} rows, columns: {list(df.columns)}")

# ── severity map (same as original) ──────────────────────────────────────────
severity_map = {
    'MURDER':               5,
    'RAPE':                 5,
    'SHOOTING':             5,
    'ROBBERY':              4,
    'STABBING':             4,
    'ACTS OF LASCIVIOUSNESS': 4,
    'CARNAPPING':           3,
    'MOTORNAPPING':         3,
    'PHYSICAL INJURIES':    3,
    'MAULING':              3,
    'THEFT':                1,
}
df['crime_severity'] = df['offense'].str.strip().str.upper().map(severity_map).fillna(2)

# ── areaCrimeCount per barangay ──────────────────────────────────────────────
# Already present in processed_crime_data.csv — recompute only if missing
if 'areaCrimeCount' not in df.columns:
    barangay_counts = df.groupby('barangay')['offense'].count().rename('areaCrimeCount')
    df = df.merge(barangay_counts, on='barangay', how='left')

# ── FIXED crime_penalty formula — 0 to 100 scale with real spread ─────────────
#
# Original formula clipped everything to 0-1 which collapsed variance.
# New formula:
#   penalty = w_severity  * severity        (0-5  → weight 8)   max 40
#           + w_frequency * log(areaCrime)  (log scale)          max ~30
#           + w_victims   * victimCount     (capped at 10)       max 20
#           + w_time      * time_risk       (night/weekend)      max 10
#                                                          TOTAL max ~100
#
# Each component is independently meaningful and matches real crime risk factors.

# time risk: night (22-05) and weekends are higher risk
hour_risk = df['hour'].apply(lambda h: 1.0 if (h >= 22 or h <= 5) else
                              0.5 if (h >= 18 or h <= 8) else 0.2)
dow_risk  = df['day_of_week'].apply(lambda d: 1.0 if d >= 5 else 0.5)
time_risk = (hour_risk + dow_risk).clip(0, 2) / 2  # normalise 0-1

victim_count = df['victimCount'].clip(0, 10) if 'victimCount' in df.columns else pd.Series([1]*len(df))

# Use exact column names from your processed CSV:
# victimKilled_binary, victimInjured_encoded, victimCount
killed   = df['victimKilled_binary']   if 'victimKilled_binary'   in df.columns else 0
injured  = df['victimInjured_encoded'] if 'victimInjured_encoded' in df.columns else 0

df['crime_penalty'] = (
    8.0  * df['crime_severity']                          +  # severity (0-40)
    30.0 * np.log1p(df['areaCrimeCount']) /
           np.log1p(df['areaCrimeCount'].max())          +  # frequency log-scaled (0-30)
    2.0  * victim_count                                  +  # victimCount capped at 10 (0-20)
    5.0  * killed                                        +  # killed binary (0 or 5)
    3.0  * injured                                       +  # injured encoded
    10.0 * time_risk                                        # time risk (0-10)
).clip(0, 100)

print(f"\ncrime_penalty stats (NEW 0-100 scale):")
print(df['crime_penalty'].describe())
print(f"Unique values: {df['crime_penalty'].nunique()}")

# ── features (identical to original — no changes to main.py needed) ───────────
features = [
    'lat', 'lng', 'month', 'day_of_week', 'hour',
    'areaCrimeCount', 'barangay_encoded', 'municipal_encoded',
    'victimCount', 'crime_severity'
]

# Drop rows with missing feature values
df_train = df.dropna(subset=features + ['crime_penalty'])
print(f"\nTraining rows after dropna: {len(df_train)}")

X = df_train[features]
y = df_train['crime_penalty']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── train (same RF config as original) ────────────────────────────────────────
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

y_pred      = model.predict(X_test)
train_r2    = r2_score(y_train, model.predict(X_train))
test_r2     = r2_score(y_test, y_pred)
mse         = mean_squared_error(y_test, y_pred)

print(f"\nModel performance:")
print(f"  MSE:        {mse:.4f}")
print(f"  Train R²:   {train_r2:.4f}")
print(f"  Test  R²:   {test_r2:.4f}")
print(f"  Pred range: {y_pred.min():.2f} – {y_pred.max():.2f}")

if train_r2 - test_r2 > 0.1:
    print("  ⚠️  Possible overfitting")
else:
    print("  ✅ Model generalises well")

# Feature importance
import pandas as pd
fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_})
fi = fi.sort_values('Importance', ascending=False)
print(f"\nTop features:\n{fi.to_string(index=False)}")

# ── save ──────────────────────────────────────────────────────────────────────
df_train.to_csv(OUT_CSV, index=False)
print(f"\nSaved training data → {OUT_CSV}")

os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
with open(MODEL_OUT, 'wb') as f:
    pickle.dump(model, f)
print(f"Saved model        → {MODEL_OUT}")
print("\n✅ Done — push models/crime_penalty_model.pkl to GitHub")