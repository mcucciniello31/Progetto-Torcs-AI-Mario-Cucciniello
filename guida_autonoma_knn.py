"""
STEP 3 - Agente KNN in TORCS (Behavior Cloning)
================================================
Carica il modello KNN addestrato e lo usa per guidare in TORCS
in tempo reale via protocollo SCR (UDP).

La struttura del loop è identica a torcs_jm_par_modulare.py,
con la funzione di guida sostituita dall'inferenza KNN.

La gestione del GEAR rimane automatica (logica rule-based):
i dati di training non sono sufficientemente precisi per imparare
le marce, mentre la logica automatica è già robusta.

Dipende da:
  - models/knn_model.pkl
  - models/scaler.pkl
  - models/feature_names.pkl

Uso:
  1. Avvia TORCS con un circuito in modalità Practice/Race
  2. python step3_knn_drive.py
  
  Opzioni:
  --host localhost    → host TORCS (default: localhost)
  --port 3001         → porta TORCS (default: 3001)
  --steps 100000      → max step prima di fermarsi
  --verbose           → stampa telemetria ad ogni step
"""

import os
import sys
import socket
import time
import pickle
import argparse

# Forza stdout UTF-8 su Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import numpy as np

# ─────────────────────────────────────────────
# CONFIGURAZIONE
# ─────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

DATA_SIZE = 2 ** 17

# Feature di input – DEVE corrispondere a step1_prepare_data.py
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

# Soglie per marce automatiche (km/h) – stessa logica del bot originale
GEAR_SPEEDS = [0, 45, 90, 145, 200, 250]

# Soglia fuori pista per eventuale fallback
TRACK_LIMIT = 1.05


# ─────────────────────────────────────────────
# CLASSI PROTOCOLLO SCR (identiche agli altri script)
# ─────────────────────────────────────────────
class ServerState:
    def __init__(self):
        self.d = {}

    def parse_server_str(self, s: str):
        s = s.strip()[:-1]
        for item in s.strip().lstrip("(").rstrip(")").split(")("):
            parts = item.split(" ")
            self.d[parts[0]] = self._parse_value(parts[1:])

    @staticmethod
    def _parse_value(tokens):
        if not tokens:
            return tokens
        if len(tokens) == 1:
            try:
                return float(tokens[0])
            except ValueError:
                return tokens[0]
        result = []
        for t in tokens:
            try:
                result.append(float(t))
            except ValueError:
                result.append(t)
        return result


class DriverAction:
    def __init__(self):
        self.d = {
            "accel":  0.0,
            "brake":  0.0,
            "clutch": 0.0,
            "gear":   1,
            "steer":  0.0,
            "focus":  [-90, -45, 0, 45, 90],
            "meta":   0,
        }

    def __repr__(self):
        out = ""
        for k, v in self.d.items():
            out += "(" + k + " "
            if isinstance(v, list):
                out += " ".join(str(x) for x in v)
            else:
                out += "%.3f" % v
            out += ")"
        return out


# ─────────────────────────────────────────────
# CARICAMENTO MODELLO
# ─────────────────────────────────────────────
class KNNAgent:
    """
    Avvolge il KNN sklearn per inferenza real-time.
    Gestisce la normalizzazione internamente.
    """

    def __init__(self):
        model_path   = os.path.join(MODELS_DIR, "knn_model.pkl")
        scaler_path  = os.path.join(MODELS_DIR, "scaler.pkl")
        feature_path = os.path.join(MODELS_DIR, "feature_names.pkl")

        for p in [model_path, scaler_path, feature_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"File non trovato: {p}\n"
                    "Esegui prima step1 e step2."
                )

        with open(model_path,   "rb") as f: self.model    = pickle.load(f)
        with open(scaler_path,  "rb") as f: self.scaler   = pickle.load(f)
        with open(feature_path, "rb") as f: self.features = pickle.load(f)

        print(f"  Modello KNN caricato ({self.model.n_neighbors} vicini)")
        print(f"  Feature: {len(self.features)}")

        # Warmup: forza la costruzione dell'indice ball_tree alla partenza
        # (evita latenza elevata al primo step reale)
        _dummy = np.zeros((1, len(self.features)))
        self.model.predict(_dummy)
        print("  Indice ball_tree: pronto.")

    def predict(self, state: dict) -> dict:
        """
        Riceve il dizionario stato TORCS (S.d) e restituisce
        {'steer', 'accel', 'brake'} come float clippati.
        """
        # Appiattiamo le liste (es. wheelSpinVel -> wheelSpinVel_0, _1, ecc.) come nel CSV
        flat_state = {}
        for k, v in state.items():
            if isinstance(v, list):
                for i, val in enumerate(v):
                    flat_state[f"{k}_{i}"] = val
            else:
                flat_state[k] = v

        if "wheelSpinVel" in state and len(state["wheelSpinVel"]) == 4:
            flat_state["wsv_avg"] = sum(state["wheelSpinVel"]) / 4.0
        else:
            flat_state["wsv_avg"] = 0.0

        # Estrai feature nell'ordine corretto, con fallback a 0.0
        x = np.array([[flat_state.get(f, 0.0) for f in self.features]])
        x = self.scaler.transform(x)
        pred = self.model.predict(x)[0]   # [steer, accel, brake]

        return {
            "steer":  float(np.clip(pred[0], -1.0,  1.0)),
            "accel":  float(np.clip(pred[1],  0.0,  1.0)),
            "brake":  float(np.clip(pred[2],  0.0,  1.0)),
        }


# ─────────────────────────────────────────────
# LOGICHE AUSILIARIE
# ─────────────────────────────────────────────
def auto_gear(speed_kmh: float, current_gear: int, steer: float) -> int:
    """
    Cambio marce automatico rule-based.
    In curva stretta (|steer| > 0.4) mantiene la marcia attuale
    per evitare scalate indesiderate.
    """
    if abs(steer) > 0.4:
        return current_gear
    gear = 1
    for i, th in enumerate(GEAR_SPEEDS):
        if speed_kmh > th:
            gear = i + 1
    return min(gear, 6)


def print_telemetry(step: int, state: dict, action: dict, source: str):
    """Stampa riga di telemetria formattata."""
    spd  = state.get("speedX", 0)
    tpos = state.get("trackPos", 0)
    ang  = state.get("angle", 0)
    gear = state.get("gear", 1)
    print(
        f"  step={step:>5}   spd={spd:>6.1f} km/h   "
        f"pos={tpos:>+.3f}   ang={ang:>+.3f}   "
        f"gear={gear:.0f}   "
        f"st={action['steer']:>+.3f}   acc={action['accel']:.3f}   brk={action['brake']:.3f}"
    )


# ─────────────────────────────────────────────
# CONNESSIONE UDP
# ─────────────────────────────────────────────
def connect(so: socket.socket, host: str, port: int):
    """Invia init e aspetta identificazione dal server TORCS."""
    init_angles = "-45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45"
    initmsg = f"SCR(init {init_angles})"

    print(f"  Connessione a {host}:{port}...")
    while True:
        so.sendto(initmsg.encode(), (host, port))
        try:
            data, _ = so.recvfrom(DATA_SIZE)
            if "***identified***" in data.decode():
                print("  >>> CONNESSO A TORCS.")
                break
        except socket.timeout:
            print("  (in attesa di TORCS...)")


# ─────────────────────────────────────────────
# LOOP PRINCIPALE
# ─────────────────────────────────────────────
def drive_loop(agent: KNNAgent, host: str, port: int,
               max_steps: int, verbose: bool):
    """Loop di controllo real-time via UDP."""

    so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    so.settimeout(1.0)

    connect(so, host, port)

    S = ServerState()
    R = DriverAction()

    step         = 0
    knn_cnt      = 0

    try:
        while step < max_steps:
            # ── Ricezione dati dal server ──────────────────
            try:
                data, _ = so.recvfrom(DATA_SIZE)
                sockstr  = data.decode()
            except socket.timeout:
                # Server non risponde: manda comunque l'ultima azione
                so.sendto(repr(R).encode(), (host, port))
                continue

            # Gestione messaggi speciali
            if "***shutdown***" in sockstr:
                print("\n  >>> GARA TERMINATA dal server.")
                break
            if "***restart***" in sockstr:
                print("\n  >>> RESTART ricevuto. Riconnessione...")
                connect(so, host, port)
                step = 0
                continue

            # ── Parse stato ───────────────────────────────
            S.parse_server_str(sockstr)
            state = S.d

            # ── Inferenza KNN ─────────────────────────────
            action = agent.predict(state)
            source = "KNN"
            knn_cnt += 1

            # ── Aiuti alla guida (come nel manual control) ────────────────
            speed = state.get("speedX", 0)
            steer = action["steer"]
            accel = action["accel"]
            brake = action["brake"]


            # Frenata preventiva progressiva (Safety Fallback per alte velocità)
            track_list = state.get("track", [200.0]*19)
            track_front = track_list[9] if len(track_list) > 9 else 200.0
            
            # Soglia dinamica in base alla velocità (spazio di frenata fisico proporzionale alla velocità)
            safe_threshold = max(35.0, speed * 0.4)
            critical_threshold = max(20.0, speed * 0.25)

            # Se andiamo veloci (>100 km/h), a ruote dritte (steer < 0.15) e la curva si avvicina
            # Rilevamento geometrico: track_front < safe_threshold E più corto dei sensori adiacenti (segno che la pista curva), o in assoluto critico (<critical_threshold)
            is_curve_ahead = False
            if len(track_list) > 10:
                is_curve_ahead = (track_front < safe_threshold and (track_front < track_list[8] or track_front < track_list[10])) or (track_front < critical_threshold)

            if speed > 100.0 and is_curve_ahead and abs(steer) < 0.15:
                accel = 0.0
                if track_front < critical_threshold:
                    brake = max(brake, 0.8)  # staccata forte vicino alla curva
                else:
                    # Rallentamento progressivo: da 0.2 a 0.7 man mano che ci avviciniamo
                    progress = (safe_threshold - track_front) / (safe_threshold - critical_threshold)
                    brake = max(brake, 0.2 + progress * 0.5)

            # Ripartizione e rilascio in curva per evitare testacoda (EBD)
            if brake > 0.05:
                accel = 0.0
                if abs(steer) >= 0.15:
                    brake = min(0.3, brake)  # limita il freno se stiamo sterzando

            wheel_vel = state.get('wheelSpinVel', [0,0,0,0])
            if len(wheel_vel) == 4:
                # Controllo di trazione (riduce accel se le ruote dietro slittano più di quelle davanti)
                if (wheel_vel[2]+wheel_vel[3]) - (wheel_vel[0]+wheel_vel[1]) > 15:
                    accel *= 0.5
                # ABS base (riduce freno se a bassa velocità c'è rischio di blocco ruote anteriori)
                if brake > 0.1 and speed > 15 and (wheel_vel[0]+wheel_vel[1])/2.0 < 5:
                    brake *= 0.1

            # Ripartitore frenata in curva (sostituito dalle logiche sopra)
            # if brake > 0.1 and abs(steer) > 0.15: 
            #     brake *= (1.0 - abs(steer)*0.8)

            action["accel"] = accel
            action["brake"] = brake

            # ── Cambio marce automatico ────────────────────
            current_gear = int(state.get("gear", 1))
            gear = auto_gear(speed, current_gear, action["steer"])

            # ── Costruzione risposta ──────────────────────
            R.d["steer"] = action["steer"]
            R.d["accel"] = action["accel"]
            R.d["brake"] = action["brake"]
            R.d["gear"]  = gear
            R.d["meta"]  = 0

            # ── Invio risposta ────────────────────────────
            so.sendto(repr(R).encode(), (host, port))

            # ── Log ───────────────────────────────────────
            if verbose or step % 100 == 0:
                print_telemetry(step, state, action, source)

            step += 1

    except KeyboardInterrupt:
        print("\n\n  Interruzione utente (Ctrl+C).")

    finally:
        so.close()
        if knn_cnt > 0:
            print(f"\n── Riepilogo sessione ─────────────────────────────")
            print(f"  Step totali : {knn_cnt}")
            print(f"  Controllo KNN      : {knn_cnt:>5}  (100.0%)")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="TORCS KNN Agent – Imitation Learning"
    )
    parser.add_argument("--host",     default="localhost", help="Host TORCS")
    parser.add_argument("--port",     type=int, default=3001, help="Porta TORCS")
    parser.add_argument("--steps",    type=int, default=100000, help="Max step")
    parser.add_argument("--verbose",  action="store_true",
                        help="Stampa telemetria ad ogni step")
    args = parser.parse_args()

    # Carica agente
    print("1/2 Caricamento modello KNN...")
    agent = KNNAgent()

    print("\n2/2 Avvio loop di guida...")

    drive_loop(
        agent=agent,
        host=args.host,
        port=args.port,
        max_steps=args.steps,
        verbose=args.verbose,
    )

    print("\n3°STEP COMPLETATO")


if __name__ == "__main__":
    main()
