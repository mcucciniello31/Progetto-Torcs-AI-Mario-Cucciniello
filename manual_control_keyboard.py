#!/usr/bin/env python3
"""
AI-AutonomeGuide - manual_control_keyboard.py
=============================================
Consente il controllo manuale del veicolo in TORCS tramite tastiera del Mac
(tasti Freccia e W/S/R/Spazio) e salva i dati telemetrici di ciascun giro
nella cartella dataset_laps/ per il Behavioral Cloning.

Esecuzione:
  conda activate torcs-env
  python manual_control_keyboard.py
"""

import socket
import sys
import os
import time
import csv
import argparse
import numpy as np
from pynput import keyboard

# --- CONFIGURAZIONE ---
HOST = 'localhost'
PORT = 3001
SID = 'SCR'
DATA_SIZE = 2**17
TRACK_LIMIT = 1.3  # Limite per considerare l'auto fuori pista

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_laps")
os.makedirs(DATASET_DIR, exist_ok=True)

# Intestazioni delle colonne per il file CSV (coerenti con FEATURE_COLS + TARGET_COLS)
HEADERS = [
    "timestamp", "angle", "trackPos", "speedX", "speedY", "speedZ", "rpm",
    "wheelSpinVel_0", "wheelSpinVel_1", "wheelSpinVel_2", "wheelSpinVel_3",
    "track_0", "track_1", "track_2", "track_3", "track_4",
    "track_5", "track_6", "track_7", "track_8", "track_9",
    "track_10", "track_11", "track_12", "track_13", "track_14",
    "track_15", "track_16", "track_17", "track_18",
    "target_steer", "target_accel", "target_brake"
]

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


class KeyboardController:
    def __init__(self):
        self.keys = set()
        self.steer = 0.0
        self.accel = 0.0
        self.brake = 0.0
        self.gear = 1
        self.meta = 0
        self.recording = True  # Attivo di default per semplificare

    def press(self, key):
        try:
            k = key.char if hasattr(key, 'char') else key.name
            if k not in self.keys:
                self.keys.add(k)
            
            # Gestione cambio marcia manuale
            if k == 'w':
                self.gear = min(self.gear + 1, 6)
            elif k == 's':
                self.gear = max(self.gear - 1, -1)
            elif k == 'r':
                self.meta = 1
            elif k == 'space':
                self.recording = not self.recording
                status = "ATTIVA" if self.recording else "DISATTIVATA"
                print(f"\n>>> Registrazione {status}")
        except Exception:
            pass

    def release(self, key):
        try:
            k = key.char if hasattr(key, 'char') else key.name
            self.keys.discard(k)
        except Exception:
            pass

    def update(self, sensors):
        current_speed = sensors.get('speedX', 0.0)

        # 1. Calcolo target sterzo
        target_steer = 0.0
        if 'left' in self.keys:
            target_steer = 1.0
        elif 'right' in self.keys:
            target_steer = -1.0

        # Smoothing sterzo (coefficiente 0.15 per sterzata progressiva)
        self.steer += 0.15 * (target_steer - self.steer)

        # Riduzione dinamica dello sterzo a velocità elevata (previene sbandate)
        if current_speed > 50.0:
            self.steer *= max(0.3, 1.0 - (current_speed - 50.0) / 250.0)

        # Stabilizzazione automatica (aiuta a riallineare l'auto sul dritto)
        angle = sensors.get('angle', 0.0)
        stabilization = 0.25 * angle
        final_steer = self.steer + stabilization
        final_steer = max(-1.0, min(1.0, final_steer))

        # 2. Calcolo acceleratore
        target_accel = 0.0
        if 'up' in self.keys:
            target_accel = 1.0
        self.accel += 0.20 * (target_accel - self.accel)
        self.accel = max(0.0, min(1.0, self.accel))

        # 3. Calcolo freno
        target_brake = 0.0
        if 'down' in self.keys:
            target_brake = 1.0
        self.brake += 0.25 * (target_brake - self.brake)
        self.brake = max(0.0, min(1.0, self.brake))

        meta_out = self.meta
        self.meta = 0  # Reset

        return final_steer, self.accel, self.brake, self.gear, meta_out


def save_to_disk(buffer, headers, lap_number, lap_time, is_recovery):
    if not buffer: return
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    minutes = int(lap_time // 60)
    seconds = int(lap_time % 60)
    milliseconds = int((lap_time % 1) * 1000)
    time_str = f"{minutes:02d}-{seconds:02d}-{milliseconds:03d}"
    prefix = "recovery_lap" if is_recovery else "lap"
    filename = f"{prefix}_{lap_number:03d}_time_{time_str}_{timestamp}.csv"
    filepath = os.path.join(DATASET_DIR, filename)
    with open(filepath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(buffer)
    print(f"\n>>> [GIRO COMPLETATO & SALVATO] {prefix.upper()}: {lap_number} | Tempo: {time_str}")


def main():
    parser = argparse.ArgumentParser(description="AI-AutonomeGuide - Manual Control Keyboard Recorder")
    parser.add_argument("--recovery", action="store_true", help="Salva i giri registrati come recovery_lap_ invece di lap_")
    args = parser.parse_args()
    is_recovery = args.recovery

    print("=" * 60)
    print(" AI-AutonomeGuide - GUIDA MANUALE E REGISTRAZIONE DATASET")
    if is_recovery:
        print(" (MODALITÀ REGISTRAZIONE RECOVERY LAPS: recovery_lap_*.csv)")
    else:
        print(" (MODALITÀ REGISTRAZIONE NORMAL LAPS: lap_*.csv)")
    print("=" * 60)
    print("Controlli:")
    print("  - Frecce Direzionali : Sterzo / Gas / Freno")
    print("  - Tasti W / S        : Cambio Marcia Manuale")
    print("  - Tasto R            : Riavvio Gara (Restart)")
    print("  - Barra Spaziatrice  : Attiva/Disattiva Registrazione")
    print("-" * 60)

    # Inizializza controller tastiera
    ctrl = KeyboardController()
    listener = keyboard.Listener(on_press=ctrl.press, on_release=ctrl.release)
    listener.start()

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

    # Variabili di stato del giro
    lap_number = 1
    prev_lap_time = 0.0
    prev_damage = 0.0
    buffer = []
    start_time = None

    print("\nIn attesa di connessione da TORCS... (Avvia la gara Practice/Race su TORCS)")

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
                print("\n>>> Connessione stabilita con TORCS!")
                continue

            S.parse_server_str(server_str)

            # Leggi sensori critici
            cur_lap_time = S.d.get('curLapTime', 0.0)
            track_pos = S.d.get('trackPos', 0.0)
            damage = S.d.get('damage', 0.0)
            speed_x = S.d.get('speedX', 0.0)

            # Rilevamento completamento giro (reset curLapTime dopo aver corso)
            if cur_lap_time < prev_lap_time and prev_lap_time > 10.0:
                # Salva il giro
                save_to_disk(buffer, HEADERS, lap_number, prev_lap_time, is_recovery)
                lap_number += 1
                buffer = []
                prev_lap_time = 0.0
                prev_damage = damage
                # Forza il riavvio per fermare l'auto sulla griglia e avere partenze pulite
                R.d['meta'] = 1
                so.sendto(str(R).encode(), (HOST, PORT))
                time.sleep(0.5)
                continue

            # Aggiorna controlli dal tastierino
            steer, accel, brake, gear, meta = ctrl.update(S.d)

            # Se l'utente preme 'R' per riavviare manualmente, svuota il buffer del giro
            if meta == 1:
                print(f"\n>>> [ANNULLATO] Gara riavviata manualmente. Giro scartato.")
                buffer = []
                prev_lap_time = 0.0
                prev_damage = damage

            # Costruisci risposta per TORCS
            R.d['steer'] = steer
            R.d['accel'] = accel
            R.d['brake'] = brake
            R.d['gear'] = gear
            R.d['meta'] = meta

            # Se la registrazione è attiva e la macchina si muove, salva lo stato nel buffer
            if ctrl.recording and cur_lap_time > 0.1 and speed_x > 2.0:
                # Estrai vettore track
                track_sens = S.d.get('track', [0.0]*19)
                wsv = S.d.get('wheelSpinVel', [0.0]*4)
                
                row = [
                    cur_lap_time,
                    S.d.get('angle', 0.0),
                    track_pos,
                    speed_x,
                    S.d.get('speedY', 0.0),
                    S.d.get('speedZ', 0.0),
                    S.d.get('rpm', 0.0),
                    wsv[0], wsv[1], wsv[2], wsv[3]
                ]
                # Aggiungi le 19 distanze track
                row.extend(track_sens)
                # Aggiungi i target che il modello dovrà predire (l'azione dell'utente)
                row.extend([steer, accel, brake])
                
                buffer.append(row)
                
                # Feedback a schermo ogni secondo di simulazione
                if int(cur_lap_time) % 5 == 0 and cur_lap_time - int(cur_lap_time) < 0.05:
                    print(f"\rRegistrando... Giro: {lap_number} | Tempo: {int(cur_lap_time)}s | Velocità: {int(speed_x)} km/h", end="")

            prev_lap_time = cur_lap_time
            so.sendto(str(R).encode(), (HOST, PORT))

    except KeyboardInterrupt:
        print("\nInterruzione da tastiera. Chiusura del client.")
    finally:
        listener.stop()
        so.close()

if __name__ == "__main__":
    main()
