#!/usr/bin/env python3
"""
AI-AutonomeGuide - preparazione_dataset.py
========================================
Unisce tutti i CSV dei giri presenti in dataset_laps/, esegue la pulizia dei dati,
seleziona i giri migliori e genera i file per l'addestramento.

Output generati:
  - models/dataset_merged.csv      -> tutti i dati uniti
  - models/dataset_clean.csv       -> dati puliti e filtrati
  - models/scaler.pkl              -> StandardScaler addestrato
  - models/feature_names.pkl       -> elenco delle feature usate
  - plots/eda_distributions.png    -> distribuzione dei target
  - plots/eda_correlations.png     -> heatmap di correlazione
  - plots/eda_track_positions.png  -> visualizzazione traiettorie
"""

import os
import sys
import glob
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Backend non interattivo per macOS/server
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# --- CONFIGURAZIONE ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_laps")
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

# Feature di input per il modello KNN
FEATURE_COLS = [
    "angle",
    "trackPos",
    "speedX",
    "speedY",
    "wsv_avg",
    "track_0",  "track_1",  "track_2",  "track_3",  "track_4",
    "track_5",  "track_6",  "track_7",  "track_8",  "track_9",
    "track_10", "track_11", "track_12", "track_13", "track_14",
    "track_15", "track_16", "track_17", "track_18",
]

# Target da predire (le azioni del pilota)
TARGET_COLS = ["target_steer", "target_accel", "target_brake"]

def load_all_laps(folder: str) -> pd.DataFrame:
    """Carica tutti i CSV di giro presenti nella cartella e li unisce."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"Nessun file *.csv trovato in: {folder}\n"
            "Guida prima manualmente l'auto avviando manual_control_keyboard.py."
        )

    frames = []
    print("Caricamento dei giri:")
    for fp in files:
        df = pd.read_csv(fp)
        df["_source_file"] = os.path.basename(fp)   # traccia il file di origine
        
        # Se manca distFromStart (perché è stato rimosso o rinominato), lo ricrea integrando la velocità sul tempo
        if "timestamp" in df.columns and "speedX" in df.columns:
            dt = df["timestamp"].diff().fillna(0.02)
            df["distFromStart"] = (df["speedX"] / 3.6 * dt).cumsum()
        else:
            # Fallback approssimativo basato sull'indice della riga
            df["distFromStart"] = np.arange(len(df)) * 0.5
            
        frames.append(df)
        print(f"  [{os.path.basename(fp)}] -> {len(df):>5} righe")

    merged = pd.concat(frames, ignore_index=True)
    
    # Calcola la media della velocità angolare delle ruote (wheelSpinVel)
    wsv_cols = ["wheelSpinVel_0", "wheelSpinVel_1", "wheelSpinVel_2", "wheelSpinVel_3"]
    if all(c in merged.columns for c in wsv_cols):
        merged["wsv_avg"] = merged[wsv_cols].mean(axis=1)
    else:
        merged["wsv_avg"] = 0.0

    print(f"\nTotale righe grezze unite: {len(merged)}")
    return merged

def clean_and_filter_data(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra i dati eliminando sbandate, fuori pista gravi o istanti di fermo."""
    print("\nFiltro e pulizia dei dati...")
    
    # 1. Rimuove righe dove l'auto è ferma (velocità < 5 km/h)
    df_clean = df[df["speedX"] > 5.0].copy()
    
    # 2. Rimuove righe con fuori pista gravi (trackPos > 1.3)
    df_clean = df_clean[df_clean["trackPos"].abs() <= 1.3]
    
    # 3. Rimuove righe dove non c'è accelerazione o lo sterzo è nullo (es. fase di reset)
    df_clean = df_clean.dropna(subset=FEATURE_COLS + TARGET_COLS)
    
    print(f"Righe dopo la pulizia: {len(df_clean)} (scartate {len(df) - len(df_clean)} righe)")
    return df_clean

def generate_eda_plots(df: pd.DataFrame):
    """Genera grafici esplorativi e li salva nella cartella plots/."""
    print("\nGenerazione dei grafici EDA...")
    
    # 1. Distribuzione dei Target
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(TARGET_COLS):
        plt.subplot(1, 3, i+1)
        sns.histplot(df[col], kde=True, bins=30, color="teal")
        plt.title(f"Distribuzione di {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eda_distributions.png"))
    plt.close()
    
    # 2. Heatmap di correlazione
    plt.figure(figsize=(12, 10))
    corr_cols = ["angle", "trackPos", "speedX", "wsv_avg"] + TARGET_COLS
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matrice di Correlazione (Feature Principali & Target)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eda_correlations.png"))
    plt.close()

    # 3. Visualizzazione traiettoria (trackPos vs distFromStart)
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="distFromStart", y="trackPos", hue="_source_file", legend=False, alpha=0.6)
    plt.axhline(1.0, color="red", linestyle="--", label="Bordo Sinistro")
    plt.axhline(-1.0, color="red", linestyle="--", label="Bordo Destro")
    plt.title("Traiettoria (Posizione Laterale vs Distanza dal Giro)")
    plt.xlabel("Distanza di Giro (m)")
    plt.ylabel("Posizione Laterale (trackPos)")
    plt.savefig(os.path.join(PLOTS_DIR, "eda_track_positions.png"))
    plt.close()
    print(f"Grafici salvati con successo in: {PLOTS_DIR}")

def main():
    try:
        # Carica tutti i file CSV
        df = load_all_laps(DATASET_DIR)
        
        # Salva dataset unito grezzo
        df.to_csv(os.path.join(MODELS_DIR, "dataset_merged.csv"), index=False)
        
        # Esegui pulizia
        df_clean = clean_and_filter_data(df)
        
        # Genera i grafici prima dello scaling per avere valori reali
        generate_eda_plots(df_clean)
        
        # Esegui normalizzazione delle feature
        print("\nNormalizzazione delle feature (StandardScaler)...")
        scaler = StandardScaler()
        scaler.fit(df_clean[FEATURE_COLS].values)
        
        # Salva dataset pulito
        df_clean.to_csv(os.path.join(MODELS_DIR, "dataset_clean.csv"), index=False)
        
        # Salva lo scaler e la lista delle feature
        with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "wb") as f:
            pickle.dump(FEATURE_COLS, f)
            
        print("\n" + "="*50)
        print(" PREPARAZIONE DEI DATI COMPLETATA CON SUCCESSO!")
        print(f" Scaler salvato in: {os.path.join(MODELS_DIR, 'scaler.pkl')}")
        print(f" Dataset pulito in: {os.path.join(MODELS_DIR, 'dataset_clean.csv')}")
        print("="*50)
        
    except Exception as e:
        print(f"\nERRORE durante la preparazione dei dati: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
