#!/usr/bin/env python3
"""
AI-AutonomeGuide - preparazione_dataset.py

Unisce tutti i CSV dei giri presenti in dataset_laps/, esegue la pulizia dei dati,
seleziona i giri migliori e genera i file per l'addestramento.

Output generati:
  - models/dataset_merged.csv      -> tutti i dati uniti
  - models/dataset_clean.csv       -> dati puliti e filtrati
  - models/scaler.pkl              -> StandardScaler addestrato
  - models/feature_names.pkl       -> elenco delle feature utilizzate
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
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# CONFIGURAZIONE
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

# Target da predire (le azioni del pilota durante la corsa)
TARGET_COLS = ["target_steer", "target_accel", "target_brake"]

def load_all_laps(folder: str) -> pd.DataFrame:
    """Carica tutti i CSV presenti nella cartella e li unisce."""
    files = sorted(glob.glob(os.path.join(folder, "*.csv")))
    if not files:
        raise FileNotFoundError(
            f"Nessun file *.csv trovato in: {folder}\n"
            "ATTENZIONE: bisogna prima guidare manualmente l'auto avviando controllo_manuale_tastiera.py o controllo_manuale_dualshockPS4.py"
        )

    frames = []
    print("Caricamento dei giri:")
    for fp in files:
        df = pd.read_csv(fp)
        df["_source_file"] = os.path.basename(fp)   # traccia il file di origine
        
        # Se manca distFromStart, ricrearlo 
        if "timestamp" in df.columns and "speedX" in df.columns:
            dt = df["timestamp"].diff().fillna(0.02)
            df["distFromStart"] = (df["speedX"] / 3.6 * dt).cumsum()
        else:
            df["distFromStart"] = np.arange(len(df)) * 0.5
            
        frames.append(df)
        print(f"  [{os.path.basename(fp)}] -> {len(df):>5} righe")

    merged = pd.concat(frames, ignore_index=True)
    
    # Calcolo della media della velocità angolare delle ruote 
    wsv_cols = ["wheelSpinVel_0", "wheelSpinVel_1", "wheelSpinVel_2", "wheelSpinVel_3"]
    if all(c in merged.columns for c in wsv_cols):
        merged["wsv_avg"] = merged[wsv_cols].mean(axis=1)
    else:
        merged["wsv_avg"] = 0.0

    print(f"\nIl totale delle righe unite (prima del filtraggio) è: {len(merged)}")
    return merged

def data_filter(df: pd.DataFrame) -> pd.DataFrame:
    """Filtra i dati eliminando fuori pista, problemi allo sterzo o momenti di stallo (acc e fren insieme)"""
    print("\nFiltraggio e pulizia dei dati...")
    
    #Rimozione righe dove l'auto è ferma, approssimativamente con velocità < 5 km/h
    df_clean = df[df["speedX"] > 5.0].copy()
    
    # Rimozione righe con fuori pista, approssimativamente con trackPos > 1.5
    df_clean = df_clean[df_clean["trackPos"].abs() <= 1.5]
    
    # Rimozione righe dove c'è stallo 
    df_clean = df_clean.dropna(subset=FEATURE_COLS + TARGET_COLS)
    
    print(f"Righe dopo il filtraggio: {len(df_clean)} (eliminate {len(df) - len(df_clean)} righe)")
    return df_clean

def generate_eda_plots(df: pd.DataFrame):
    """Generazione grafici di progetto."""
    # Target
    plt.figure(figsize=(15, 5))
    for i, col in enumerate(TARGET_COLS):
        plt.subplot(1, 3, i+1)
        sns.histplot(df[col], kde=True, bins=30, color="dodgerblue")
        plt.title(f"Distribuzione di {col}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eda_distributions.png"))
    plt.close()
    
    # Heatmap di correlazione
    plt.figure(figsize=(12, 10))
    corr_cols = ["angle", "trackPos", "speedX", "wsv_avg"] + TARGET_COLS
    corr = df[corr_cols].corr()
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5)
    plt.title("Matrice di Correlazione (Feature Principali & Target)")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, "eda_correlations.png"))
    plt.close()

    # Visualizzazione delle traiettorie
    plt.figure(figsize=(12, 6))
    sns.lineplot(data=df, x="distFromStart", y="trackPos", hue="_source_file", legend=False, alpha=0.6)
    plt.axhline(1.0, color="crimson", linestyle="--", label="Bordo Sinistro")
    plt.axhline(-1.0, color="crimson", linestyle="--", label="Bordo Destro")
    plt.title("Traiettoria (Posizione / Distanza dal Giro)")
    plt.xlabel("Distanza di Giro (in metri)")
    plt.ylabel("Posizione Laterale (trackPos)")
    plt.savefig(os.path.join(PLOTS_DIR, "eda_track_positions.png"))
    plt.close()
    print(f"Grafici salvati in: {PLOTS_DIR}")

def main():
    try:
        # Caricamento di tutti i file dei giri di test
        df = load_all_laps(DATASET_DIR)
        
        # Salvataggio dataset grezzo
        df.to_csv(os.path.join(MODELS_DIR, "dataset_merged.csv"), index=False)
        
        df_clean = data_filter(df)
        
        # Grafico dello scaling 
        generate_eda_plots(df_clean)
        
        # Normalizzazione delle 24 feature
        print("\nNormalizzazione delle feature...")
        scaler = StandardScaler()
        scaler.fit(df_clean[FEATURE_COLS].values)
        
        # Salvataggio dataset pulito
        df_clean.to_csv(os.path.join(MODELS_DIR, "dataset_clean.csv"), index=False)
        
        # Salvataggio dello scaler con la lista delle feature
        with open(os.path.join(MODELS_DIR, "scaler.pkl"), "wb") as f:
            pickle.dump(scaler, f)
        with open(os.path.join(MODELS_DIR, "feature_names.pkl"), "wb") as f:
            pickle.dump(FEATURE_COLS, f)
            
        print(" Dati preparati con successo!!!")
        print(f" Scaler salvato in: {os.path.join(MODELS_DIR, 'scaler.pkl')}")
        print(f" Dataset pulito salvato in: {os.path.join(MODELS_DIR, 'dataset_clean.csv')}")
        
    except Exception as e:
        print(f"\Errore {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
