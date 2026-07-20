"""
STEP 2 - Training e Valutazione del Modello KNN
================================================
Legge il dataset pulito prodotto da step1_prepare_data.py,
addestra un KNeighborsRegressor separato per steer/accel/brake
(o multi-output), valuta le performance e salva il modello.

Dipende da:
  - models/dataset_clean.csv
  - models/scaler.pkl
  - models/feature_names.pkl

Output generati:
  - models/knn_model.pkl           → modello KNN addestrato
  - plots/train_predictions.png    → scatter pred vs reale
  - plots/train_residuals.png      → residui per target

Uso:
  python step2_train_knn.py
  python step2_train_knn.py --k 10          # scegli k manualmente
  python step2_train_knn.py --eval-only     # rivaluta modello esistente
"""

import os
import sys
import pickle
import argparse
import json

# Forza stdout UTF-8 su Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GroupShuffleSplit, GroupKFold
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# ─────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
PLOTS_DIR  = os.path.join(BASE_DIR, "plots")

TARGET_COLS = ["target_steer", "target_accel", "target_brake"]
TARGET_RANGES = {
    "target_steer": (-1.0,  1.0),
    "target_accel": ( 0.0,  1.0),
    "target_brake": ( 0.0,  1.0),
}

# Iperparametri default
DEFAULT_K         = 4            # numero vicini (default se non cercato)
DEFAULT_WEIGHTS   = "distance"   # "uniform" oppure "distance"
DEFAULT_ALGO      = "ball_tree"  # piu' veloce di "brute" su dataset medi
DEFAULT_METRIC    = "euclidean"
TEST_SIZE         = 0.20         # 20% per test
RANDOM_STATE      = 42

# ─────────────────────────────────────────────
# UTILITÀ
# ─────────────────────────────────────────────
def load_artifacts(dataset_filename="dataset_clean.csv"):
    """Carica dataset pulito, scaler e lista feature."""
    clean_path   = os.path.join(MODELS_DIR, dataset_filename)
    scaler_path  = os.path.join(MODELS_DIR, "scaler.pkl")
    feature_path = os.path.join(MODELS_DIR, "feature_names.pkl")

    for p in [clean_path, scaler_path, feature_path]:
        if not os.path.exists(p):
            raise FileNotFoundError(
                f"File non trovato: {p}\n"
                "Assicurati di aver generato il dataset corrispondente."
            )

    df = pd.read_csv(clean_path)
    with open(scaler_path,  "rb") as f: scaler  = pickle.load(f)
    with open(feature_path, "rb") as f: features = pickle.load(f)

    return df, scaler, features


def prepare_xy(df: pd.DataFrame, features: list, scaler):
    """Estrae X (normalizzato) e y (raw) dal DataFrame."""
    X_raw = df[features].values
    y     = df[TARGET_COLS].values
    X     = scaler.transform(X_raw)
    return X, y





# ─────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────
def train(X_train, y_train, k: int) -> KNeighborsRegressor:
    """Addestra un KNN multi-output sui dati puri senza compressione."""
    model = KNeighborsRegressor(
        n_neighbors=k,
        weights=DEFAULT_WEIGHTS,
        algorithm=DEFAULT_ALGO,
        metric=DEFAULT_METRIC,
        n_jobs=1   # n_jobs=1 e' piu' veloce per query singole su Windows
    )
    model.fit(X_train, y_train)
    return model


# ─────────────────────────────────────────────
# VALUTAZIONE
# ─────────────────────────────────────────────
def evaluate(model: KNeighborsRegressor, X_test, y_test) -> dict:
    """Calcola metriche per ogni target."""
    y_pred = model.predict(X_test)
    results = {}

    target_names_mapping = {
        "target_steer": "target_sterzata",
        "target_accel": "target_accelerazione",
        "target_brake": "target_frenata"
    }

    for i, col in enumerate(TARGET_COLS):
        mae  = mean_absolute_error(y_test[:, i], y_pred[:, i])
        rmse = np.sqrt(mean_squared_error(y_test[:, i], y_pred[:, i]))
        r2   = r2_score(y_test[:, i], y_pred[:, i])
        results[col] = {"mae": mae, "rmse": rmse, "r2": r2,
                        "y_true": y_test[:, i], "y_pred": y_pred[:, i]}
        
        printed_name = target_names_mapping.get(col, col)
        print(f"  {printed_name}:")
        print(f"    MAE (Errore Assoluto Medio) = {mae:.4f}")
        print(f"    RMSE (Radice dell'Errore Quadratico Medio) = {rmse:.4f}")
        print(f"    R² (Coefficiente di Determinazione) = {r2:.4f}")

    return results


# ─────────────────────────────────────────────
# PLOT RISULTATI
# ─────────────────────────────────────────────
def plot_predictions(results: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("KNN – Previsione vs Reale (Test Set)", fontsize=13, fontweight="bold")

    colors = ["#3b82f6", "#22c55e", "#ef4444"]

    for ax, (col, res), color in zip(axes, results.items(), colors):
        lo, hi = TARGET_RANGES[col]
        ax.scatter(res["y_true"], res["y_pred"],
                   alpha=0.15, s=4, color=color, rasterized=True)
        ax.plot([lo, hi], [lo, hi], "k--", linewidth=1, label="perfetto")
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
        ax.set_xlabel("Valore Reale")
        ax.set_ylabel("Previsione KNN")
        ax.set_title(f"{col}\nMAE={res['mae']:.4f}  R²={res['r2']:.4f}")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "train_predictions.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"\n  Grafico salvato: {out}")


def plot_residuals(results: dict):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle("KNN – Distribuzione Residui (Test Set)", fontsize=13, fontweight="bold")

    colors = ["#3b82f6", "#22c55e", "#ef4444"]

    for ax, (col, res), color in zip(axes, results.items(), colors):
        residuals = res["y_pred"] - res["y_true"]
        ax.hist(residuals, bins=60, color=color, alpha=0.8, edgecolor="none")
        ax.axvline(0, color="black", linewidth=1.2, linestyle="--")
        ax.axvline(residuals.mean(), color="red", linewidth=1,
                   linestyle=":", label=f"media={residuals.mean():.4f}")
        ax.set_title(col)
        ax.set_xlabel("Residuo (pred − reale)")
        ax.set_ylabel("Frequenza")
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    out = os.path.join(PLOTS_DIR, "train_residuals.png")
    plt.savefig(out, dpi=120)
    plt.close()
    print(f"  Grafico salvato: {out}")


# ─────────────────────────────────────────────
# SALVATAGGIO MODELLO
# ─────────────────────────────────────────────
def save_model(model: KNeighborsRegressor):
    model_path = os.path.join(MODELS_DIR, "knn_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    print(f"\n  Modello KNN salvato: {model_path}")


def load_model() -> KNeighborsRegressor:
    model_path = os.path.join(MODELS_DIR, "knn_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Modello non trovato: {model_path}")
    with open(model_path, "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="Training KNN per Imitation Learning su TORCS")
    parser.add_argument("--dataset",   type=str,  default="dataset_clean.csv", help="Nome file del dataset in models/ (default: dataset_clean.csv)")
    args = parser.parse_args()

    # 1. Carica (foto3: Caricamento dataset)
    print("1/4 Caricamento dataset")
    df, scaler, features = load_artifacts(dataset_filename=args.dataset)

    # Prepara X, y
    X, y = prepare_xy(df, features, scaler)

    # 2. Split train/test (80/20 split) (foto4: Split per prevenire perdita di dati)
    print("\n2/4 Split per prevenire perdita di dati")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_STATE
    )

    # 3. Training (foto5: Addestramento KNN)
    print("\n3/4 Addestramento KNN")
    k = DEFAULT_K
    model = train(X_train, y_train, k)
    save_model(model)

    # 4. Valutazione
    print("\n4/4 Valutazione sul test set...")
    results = evaluate(model, X_test, y_test)

    # Grafici
    print("\n  Generazione grafici...")
    plot_predictions(results)
    plot_residuals(results)

    print("\n2°STEP COMPLETATO")


if __name__ == "__main__":
    main()
