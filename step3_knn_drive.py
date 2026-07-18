#!/usr/bin/env python3
"""
AI-AutonomeGuide - step3_knn_drive.py
======================================
Carica lo Scaler e il modello KNN addestrati per guidare l'auto in TORCS
in tempo reale usando il protocollo UDP SCR.

Dipende da:
  - models/knn_model.pkl
  - models/scaler.pkl
  - models/feature_names.pkl

Esecuzione:
  conda activate torcs-env
  python step3_knn_drive.py
"""

import socket
import sys
import os
import time
import pickle
import numpy as np

# --- CONFIGURAZIONE ---
HOST = 'localhost'
PORT = 3001
SID = 'SCR'
DATA_SIZE = 2**17

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Soglie per le marce automatiche (in km/h)
GEAR_SPEEDS = [0, 45, 90, 145, 200, 250]
TRACK_LIMIT = 1.1  # Limite oltre il quale si applica il comportamento di recovery

class ServerState:
    def __init__(self):
        self.d = dict()

    def parse_server_str(self, server_string):
        servstr = server_string.strip()[:-1]
        sslisted = servstr.strip().lstrip('(').rstrip(')').split(')(')
        for i in sslisted:
            w = i.split(' ')
            self.d[w[0]] = self.destringify(w[1:])

    def destringify(self, s):
        if not s: return s
        if type(s) is str:
            try: return float(s)
            except ValueError: return s
        elif type(s) is list:
            if len(s) < 2: return self.destringify(s[0])
            else: return [self.destringify(i) for i in s]


class DriverAction:
    def __init__(self):
        self.d = {
            'accel': 0.0,
            'brake': 0.0,
            'clutch': 0.0,
            'gear': 1,
            'steer': 0.0,
            'focus': [-90, -45, 0, 45, 90],
            'meta': 0
        }

    def __repr__(self):
        out = str()
        for k in self.d:
            out += '(' + k + ' '
            v = self.d[k]
            if not isinstance(v, list): out += '%.3f' % v
            else: out += ' '.join([str(x) for x in v])
            out += ')'
        return out


class KNNAgent:
    def __init__(self):
        model_path = os.path.join(MODELS_DIR, "knn_model.pkl")
        scaler_path = os.path.join(MODELS_DIR, "scaler.pkl")
        feature_path = os.path.join(MODELS_DIR, "feature_names.pkl")

        for path in [model_path, scaler_path, feature_path]:
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"File non trovato: {path}\n"
                    "Assicurati di aver completato step1_prepare_data.py e step2_train_knn.py."
                )

        with open(model_path, "rb") as f:
            self.model = pickle.load(f)
        with open(scaler_path, "rb") as f:
            self.scaler = pickle.load(f)
        with open(feature_path, "rb") as f:
            self.features = pickle.load(f)

        print(f"Modello KNN caricato correttamente ({len(self.features)} feature di input).")

    def act(self, S_d):
        """Esegue l'inferenza KNN per predire le azioni di guida."""
        # Costruisci il vettore delle feature in tempo reale
        # Ricrea wsv_avg
        wsv = S_d.get('wheelSpinVel', [0.0]*4)
        wsv_avg = sum(wsv) / 4.0

        track = S_d.get('track', [0.0]*19)
        
        # Mapping esatto del dizionario dei sensori nelle feature previste dallo scaler
        feature_dict = {
            "angle": S_d.get('angle', 0.0),
            "trackPos": S_d.get('trackPos', 0.0),
            "speedX": S_d.get('speedX', 0.0),
            "speedY": S_d.get('speedY', 0.0),
            "wsv_avg": wsv_avg
        }
        for i in range(19):
            feature_dict[f"track_{i}"] = track[i]

        # Costruisci l'array numpy ordinato come richiesto dal modello
        x_raw = np.array([feature_dict[f] for f in self.features]).reshape(1, -1)
        
        # Normalizza le feature
        x_scaled = self.scaler.transform(x_raw)
        
        # Esegui la predizione
        prediction = self.model.predict(x_scaled)[0]
        
        # Estrai e normalizza/taglia i target nei range corretti
        steer = np.clip(prediction[0], -1.0, 1.0)
        accel = np.clip(prediction[1], 0.0, 1.0)
        brake = np.clip(prediction[2], 0.0, 1.0)

        # Evita combinazioni assurde (es. frenare e accelerare contemporaneamente al massimo)
        if brake > 0.1:
            accel = max(0.0, accel - brake)

        return steer, accel, brake


def main():
    print("=" * 60)
    print(" AI-AutonomeGuide - AGENTE DI GUIDA AUTONOMA KNN")
    print("=" * 60)

    try:
        agent = KNNAgent()
    except Exception as e:
        print(f"Errore di inizializzazione dell'agente: {e}")
        return

    # Inizializzazione socket UDP
    so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    so.settimeout(2.0)
    initmsg = f"{SID}(init -45 -19 -12 -7 -4 -2.5 -1.2 -0.6 0 0.6 1.2 2.5 4 7 12 19 45)"

    try:
        so.sendto(initmsg.encode(), (HOST, PORT))
    except Exception as e:
        print(f"Errore di connessione a TORCS: {e}")
        return

    S = ServerState()
    R = DriverAction()

    print("\nConnessione a TORCS in corso... (Avvia la corsa Practice/Race su TORCS)")

    try:
        while True:
            try:
                buf, _ = so.recvfrom(DATA_SIZE)
            except socket.timeout:
                print(".", end="", flush=True)
                continue

            if not buf:
                continue

            server_str = buf.decode()
            if "***identified***" in server_str:
                print("\n>>> Connessione stabilita con successo!")
                continue

            S.parse_server_str(server_str)

            # Leggi parametri chiave per logica rule-based ausiliaria
            speed_x = S.d.get('speedX', 0.0)
            track_pos = S.d.get('trackPos', 0.0)
            angle = S.d.get('angle', 0.0)

            # 1. LOGICA DI RECOVERY (Se l'auto esce fuori pista, il KNN fallisce)
            if abs(track_pos) > TRACK_LIMIT:
                # Comportamento di emergenza deterministico per rientrare
                print(f"\r[EMERGENZA/RECOVERY] Auto fuori pista (trackPos: {track_pos:.2f}). Rientro automatico...", end="")
                # Sterza forte verso il centro
                steer = -np.sign(track_pos) * 0.4
                accel = 0.25
                brake = 0.0
            else:
                # 2. GUIDA AUTONOMA KNN
                steer, accel, brake = agent.act(S.d)
                
                # Applica una correzione di sicurezza se lo sterzo del KNN è instabile sul dritto
                if abs(track_pos) < 0.1 and abs(angle) < 0.02:
                    steer = 0.0  # mantieni dritto

            # 3. CAMBIO MARCIA AUTOMATICO (Molto più robusto del cambio appreso)
            gear = 1
            for i, speed in enumerate(GEAR_SPEEDS):
                if speed_x > speed:
                    gear = i + 1
            gear = min(gear, 6)

            # Invia comandi a TORCS
            R.d['steer'] = steer
            R.d['accel'] = accel
            R.d['brake'] = brake
            R.d['gear'] = gear
            R.d['meta'] = 0

            # Stampa feedback ogni secondo di simulazione
            cur_time = S.d.get('curLapTime', 0.0)
            if int(cur_time) % 5 == 0 and cur_time - int(cur_time) < 0.05:
                print(f"\rGuida Autonoma... Velocità: {int(speed_x)} km/h | Sterzo: {steer:5.2f} | Gas: {accel:4.2f} | Freno: {brake:4.2f}", end="")

            so.sendto(str(R).encode(), (HOST, PORT))

    except KeyboardInterrupt:
        print("\nInterruzione da tastiera. Spegnimento agente.")
    finally:
        so.close()

if __name__ == "__main__":
    main()
