#!/usr/bin/env python3
"""
AI-AutonomeGuide - addestramento_knn.py

Script che legge il dataset pulito e addestra un KNeighborsRegressor per predire
sterzo (steer), acceleratore (accel) e freno (brake) a partire dai sensori dell'auto.

Dipende dai file:
  - models/dataset_clean.csv
  - models/scaler.pkl
  - models/feature_names.pkl

E genera i seguenti elementi:
  - models/knn_model.pkl           -> modello del KNN addestrato
  - plots/train_predictions.png    -> scatter predizione / valore reale
  - plots/train_residuals.png      -> istogramma degli errori di predizion
"""

import os
import sys
import pickle
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

TARGET_COLS = ["target_steer", "target_accel", "target_brake"]

def load_data():
    """Caricamento del dataset, dello scaler e delle feature."""
    clean_path = os.path.join(MODELS_DIR, "dataset_clean.csv")
    scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
    feature_path = os.path.join(MODELS_DIR, "feature_names.pkl")

    for path in [clean_path, scaler_path, feature_path]:
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"File non trovato: {path}\n"
                "Assicurati di aver lanciato lo script preparazione_dataset.py."
            )

    df = pd.read_csv(clean_path)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(feature_path, "rb") as f:
        features = pickle.load(f)

    return df, scaler, features

def plot_evaluation(y_test, y_pred):
    """Generazione dei grafici (e il relativo salvatggio nella cartella plots/."""
    # Scatter: valore reale in comparazione col valore predetto
    plt.figure(figsize=(18, 5))
    names = ["Sterzo (Steer)", "Acceleratore (Accel)", "Freno (Brake)"]
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.scatter(y_test[:, i], y_pred[:, i], alpha=0.3, color="dodgerblue", edgecolors="none")
        min_val = min(y_test[:, i].min(), y_pred[:, i].min())
        max_val = max(y_test[:, i].max(), y_pred[:, i].max())
        plt.plot([min_val, max_val], [min_val, max_val], color="crimson", linestyle="--", lw=2)
        plt.title(f"{names[i]}: Reale / Predetto")
        plt.xlabel("Valore Reale (Manuale)")
        plt.ylabel("Valore Predetto (KNN)")
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "train_predictions.png"))
    plt.close()

    # Istogrammi dei valori residui (cioè degli "errori")
    plt.figure(figsize=(18, 5))
    residuals = y_test - y_pred
    
    for i in range(3):
        plt.subplot(1, 3, i+1)
        plt.hist(residuals[:, i], bins=30, color="crimson", alpha=0.7, edgecolor="black")
        plt.axvline(0, color="black", linestyle="--", lw=1.5)
        plt.title(f"Residui di {names[i]}")
        plt.xlabel("Errore (Valore Reale - Valore Predetto)")
        plt.ylabel("Frequenza")
        
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "train_residuals.png"))
    plt.close()
    print(f"Grafici salvati in: {PLOTS_DIR}")

def main():
    parser = argparse.ArgumentParser(description="Addestramento del modello KNN per la guida autonoma.")
    parser.add_argument("--k", type=int, default=3, help="Numero di vicini per il KNN (3)")
    parser.add_argument("--weights", type=str, default="distance", choices=["uniform", "distance"], help="Ponderazione dei vicini (default: distance)")
    args = parser.parse_args()

    try:
        # Carica i dati
        print("Caricamento dataset e scaler...")
        df, scaler, features = load_data()

        # Estrai X (normalizzato) e y (valori reali delle azioni)
        X = scaler.transform(df[features].values)
        y = df[TARGET_COLS].values

        # Split in Train e Test set
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.20, random_state=42
        )

        print(f"\nAddestramento del regressore KNN con k={args.k} ({args.weights} weights)...")
        # Inizializza e addestra il modello KNN (usando ball_tree per calcoli rapidi in real-time)
        knn = KNeighborsRegressor(
            n_neighbors=args.k,
            weights=args.weights,
            algorithm="ball_tree",
            metric="euclidean"
        )
        knn.fit(X_train, y_train)

        # Valuta sul test set
        print("Valutazione del modello sul test set...")
        y_pred = knn.predict(X_test)
        
        # Limita i valori predetto ai range ammessi per sicurezza
        y_pred[:, 0] = np.clip(y_pred[:, 0], -1.0, 1.0) # steer
        y_pred[:, 1] = np.clip(y_pred[:, 1], 0.0, 1.0)  # accel
        y_pred[:, 2] = np.clip(y_pred[:, 2], 0.0, 1.0)  # brake

        # Calcola le metriche
        for i, col in enumerate(TARGET_COLS):
            r2 = r2_score(y_test[:, i], y_pred[:, i])
            mae = mean_absolute_error(y_test[:, i], y_pred[:, i])
            rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
            print(f"\nMetriche per {col}:")
            print(f"  R2 Score : {r2:7.4f}")
            print(f"  MAE      : {mae:7.4f}")
            print(f"  RMSE     : {rmse:7.4f}")

        # Genera i grafici di valutazione
        plot_evaluation(y_test, y_pred)

        # Salva il modello addestrato
        model_output_path = os.path.join(MODELS_DIR, "knn_model.pkl")
        with open(model_output_path, "wb") as f:
            pickle.dump(knn, f)

        print("\n" + "="*50)
        print(" ADDESTRAMENTO COMPLETATO CON SUCCESSO!")
        print(f" Modello KNN salvato in: {model_output_path}")
        print("="*50)

    except Exception as e:
        print(f"\nERRORE durante il training: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
