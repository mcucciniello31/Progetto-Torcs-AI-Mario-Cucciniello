import os
import pickle
import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split

BASE_DIR = "/Users/macucc/Downloads/AI-AutonomeGuide"
MODELS_DIR = os.path.join(BASE_DIR, "models")

print("1/3 Caricamento dei dati e dei file di configurazione...")
df = pd.read_csv(os.path.join(MODELS_DIR, "dataset_clean.csv"))

with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "rb") as f:
    features = pickle.load(f)
with open(os.path.join(MODELS_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

X_raw = df[features].values
y = df[["target_steer", "target_accel", "target_brake"]].values

print("2/3 Preparazione e partizionamento train/test...")
X_train_raw, X_test_raw, y_train, y_test = train_test_split(X_raw, y, test_size=0.20, random_state=42)

# Normalizza
X_train = scaler.transform(X_train_raw)
X_test = scaler.transform(X_test_raw)

print("3/3 Analisi delle prestazioni per K da 1 a 10...")
print("-" * 75)
print(f"{'K':>2} | {'R² Sterzata':>12} | {'R² Accel':>12} | {'R² Frenata':>12} | {'R² Medio':>12}")
print("-" * 75)

best_k = None
best_score = -1

for k in range(1, 11):
    model = KNeighborsRegressor(
        n_neighbors=k,
        weights="distance",
        algorithm="ball_tree",
        metric="euclidean",
        n_jobs=-1  # Usa tutte le CPU disponibili per velocizzare il test
    )
    model.fit(X_train, y_train)
    
    # Calcola R² per ciascun target
    y_pred = model.predict(X_test)
    r2_scores = []
    for i in range(3):
        ss_res = np.sum((y_test[:, i] - y_pred[:, i]) ** 2)
        ss_tot = np.sum((y_test[:, i] - np.mean(y_test[:, i])) ** 2)
        r2 = 1.0 - (ss_res / ss_tot)
        r2_scores.append(r2)
        
    mean_r2 = np.mean(r2_scores)
    print(f"{k:>2} | {r2_scores[0]:12.4f} | {r2_scores[1]:12.4f} | {r2_scores[2]:12.4f} | {mean_r2:12.4f}")
    
    if mean_r2 > best_score:
        best_score = mean_r2
        best_k = k

print("-" * 75)
print(f"Risultato: Il miglior K trovato è K = {best_k} (con un R² medio di {best_score:.4f})")
