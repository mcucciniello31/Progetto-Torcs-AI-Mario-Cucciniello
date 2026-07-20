"""
Agent KNN in TORCS (Behavior Cloning) - guida_autonoma_knn.py

Script che carica il modello KNN addestrato e lo utilizza per guidare in TORCS
in tempo reale tramite protocollo SCR (UDP).

Dipende dai file:
  - models/knn_model.pkl
  - models/scaler.pkl
  - models/feature_names.pkl
"""
#Configurazione
import os
import sys
import socket
import time
import pickle
import argparse
import numpy as np

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_SIZE = 2 ** 17
TRACK_LIMIT = 1.5
# Feature di input che deve corrispondere a preparazione_dataset.py
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

#Dato che anche qui le marce restano automatiche, definiamo la logica di cambio
GEAR_SPEEDS = [0, 45, 90, 145, 200, 250]

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

class KNNAgent:
"""Gestione del modello KNN in TORCS"""

    def __init__(self):
        model_path   = os.path.join(MODELS_DIR, "knn_model.pkl")
        scaler_path  = os.path.join(MODELS_DIR, "scaler.pkl")
        feature_path = os.path.join(MODELS_DIR, "feature_names.pkl")

        for p in [model_path, scaler_path, feature_path]:
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"Errore! - File non trovato: {p}\n"
                    "Eseguire prima preparazione_dataset.py e addestramento_knn.py"
                )

        with open(model_path,   "rb") as f: self.model    = pickle.load(f)
        with open(scaler_path,  "rb") as f: self.scaler   = pickle.load(f)
        with open(feature_path, "rb") as f: self.features = pickle.load(f)

        print(f"  Modello KNN caricato ({self.model.n_neighbors} vicini)")
        print(f"  Feature presenti: {len(self.features)}")

    def predict(self, state: dict) -> dict:
        """
        Ricezione dello stato TORCS (S.d) e restituzione dei 3 valori
        """
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

        # Estrazione feature in ordine
        x = np.array([[flat_state.get(f, 0.0) for f in self.features]])
        x = self.scaler.transform(x)
        pred = self.model.predict(x)[0] 

        return {
            "steer":  float(np.clip(pred[0], -1.0,  1.0)),
            "accel":  float(np.clip(pred[1],  0.0,  1.0)),
            "brake":  float(np.clip(pred[2],  0.0,  1.0)),
        }
#Connessione a TORCS tramite UDP
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

#logica cambio automatico (identica alla guida manuale)
def auto_gear(speed_kmh: float, current_gear: int, steer: float) -> int:
    if abs(steer) > 0.4:
        return current_gear
    gear = 1
    for i, th in enumerate(GEAR_SPEEDS):
        if speed_kmh > th:
            gear = i + 1
    return min(gear, 6)
  
#stampa dei dati in tempo reale (ogni secondo), separati da spazi
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

#Logica di guida del modello
def drive_loop(agent: KNNAgent, host: str, port: int,
               max_steps: int, verbose: bool):
    """La guida viene controllata via UDP."""

    so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    so.settimeout(1.0)

    connect(so, host, port)

    S = ServerState()
    R = DriverAction()

    step         = 0
    knn_cnt      = 0

    try:
        while step < max_steps:
            # Dati in tempo reale proveniente dal server
            try:
                data, _ = so.recvfrom(DATA_SIZE)
                sockstr  = data.decode()
            except socket.timeout:
                # Logica di errore di connessione
                so.sendto(repr(R).encode(), (host, port))
                continue
              
            S.parse_server_str(sockstr)
            state = S.d

            # Inferenza KNN: processo con cui l'algoritmo decide 
            #come guidare l'auto, circa 50 volte ogni secondo
            action = agent.predict(state)
            source = "KNN"
            knn_cnt += 1

            # ---Serie di accortenza inserite per facilitare la guida al modello---
          
            speed = state.get("speedX", 0)
            steer = np.clip(action["steer"] * 1.2, -1.0, 1.0)
            accel = action["accel"]
            brake = np.clip(action["brake"] * 2.5, 0.0, 1.0)

            # Se andiamo veloci sul dritto e l'ingresso della chicane è vicino, 
            #forziamo il freno per non andare lunghi
            track_list = state.get("track", [200.0]*19)
            track_front = track_list[9] if len(track_list) > 9 else 200.0
            if speed > 70.0 and abs(steer) < 0.05 and track_front < 20.0:
                brake = max(brake, 0.7)
                accel = 0.0
              
             # Cambio marce automatico
            current_gear = int(state.get("gear", 1))
            gear = auto_gear(speed, current_gear, action["steer"])

            wheel_vel = state.get('wheelSpinVel', [0,0,0,0])
            if len(wheel_vel) == 4:
                # Controllo di trazione (riduce accel se le ruote dietro slittano più di quelle davanti)
                if (wheel_vel[2]+wheel_vel[3]) - (wheel_vel[0]+wheel_vel[1]) > 15:
                    accel *= 0.5
                # Logica di ABS - Anti-lock Braking System, che impedisce il bloccaggio delle ruote durante 
                #le frenate brusche (riduce il freno se a bassa velocità c'è il rischio di bloccare le ruote anteriori)
                if brake > 0.1 and speed > 15 and (wheel_vel[0]+wheel_vel[1])/2.0 < 5:
                    brake *= 0.1

            # Ripartitore di frenata in curva (riduce freno quando si sterza bruscamente)
            if brake > 0.1 and abs(steer) > 0.15: 
                brake *= (1.0 - abs(steer)*0.8)

            action["accel"] = accel
            action["brake"] = brake
            action["steer"] = steer

            # Risposta del modello -> invio -> messaggio di conferma/errore
            R.d["steer"] = action["steer"]
            R.d["accel"] = action["accel"]
            R.d["brake"] = action["brake"]
            R.d["gear"]  = gear
            R.d["meta"]  = 0
            so.sendto(repr(R).encode(), (host, port))
            if verbose or step % 100 == 0:
                print_telemetry(step, state, action, source)

            step += 1

    except KeyboardInterrupt:
        print("\n\n  Per interrompere la corsa, premere Ctrl+C.")

    finally:
        so.close()
        if knn_cnt > 0:
            print(f"\nRiepilogo dati ")
            print(f"  Percentuale di controllo del modello KNN: {knn_cnt:>5}  (100.0%)")
          
def main():
    parser = argparse.ArgumentParser(
        description="Agent KNN in Torcs – Behaviour Cloning"
    )
    parser.add_argument("--host",     default="localhost", help="Host TORCS")
    parser.add_argument("--port",     type=int, default=3001, help="Porta TORCS")
    parser.add_argument("--verbose",  action="store_true",
    help="Stampa dati telemetrici ad ogni step")
    args = parser.parse_args()

    # Carica agente -> avvio script di guida autonoma
    print("1/2 Caricamento modello KNN...")
    agent = KNNAgent()
    print("\n2/2 Avvio script di guida autonoma...")

    drive_loop(
        agent=agent,
        host=args.host,
        port=args.port,
        verbose=args.verbose,
    )

    print("\nGuida Completata!!!")


if __name__ == "__main__":
    main()
