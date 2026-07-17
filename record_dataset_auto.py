#!/usr/bin/env python3
"""
AI-AutonomeGuide - record_dataset_auto.py
=========================================
Genera AUTOMATICAMENTE il dataset di addestramento facendo guidare l'auto
dal controllore euristico (esperto). Registra i sensori e i comandi ottimali
in file CSV all'interno di dataset_laps/.

Risolve la difficoltà della guida manuale da tastiera raccogliendo dati
di guida perfetti e puliti in autonomia.

Esecuzione:
  conda activate torcs-env
  python record_dataset_auto.py --laps 10
"""

import socket
import sys
import os
import time
import csv
import math
import argparse

# --- CONFIGURAZIONE ---
HOST = 'localhost'
PORT = 3001
SID = 'SCR'
DATA_SIZE = 2**17

# Parametri del controllore euristico
TARGET_SPEED = 90.0  # Velocità di sicurezza per traiettorie stabili
STEER_GAIN = 15.0
CENTERING_GAIN = 0.20
BRAKE_THRESHOLD = 0.90
GEAR_SPEEDS = [0, 45, 90, 145, 200, 250]
ENABLE_TRACTION_CONTROL = True

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_laps")
os.makedirs(DATASET_DIR, exist_ok=True)

# Intestazioni coerenti con gli altri step
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
            'meta': 0,
            'steer': 0.0,
            'focus': [-90, -45, 0, 45, 90]
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

# --- MODULI DEL PILOTA EURISTICO ---
def calculate_steering(S):
    angle = S.get('angle', 0.0)
    track_pos = S.get('trackPos', 0.0)
    steer = (angle * STEER_GAIN / math.pi) - (track_pos * CENTERING_GAIN)
    return max(-1.0, min(1.0, steer))

def calculate_throttle(S, steer, current_accel):
    speed_x = S.get('speedX', 0.0)
    # Se siamo al di sotto della velocità target (aggiustata in curva dallo sterzo), acceleriamo
    if speed_x < (TARGET_SPEED - abs(steer) * 2.5):
        accel = min(1.0, current_accel + 0.3)
    else:
        accel = max(0.0, current_accel - 0.2)
        
    # Boost in partenza
    if speed_x < 10.0:
        accel += 0.5
    return max(0.0, min(1.0, accel))

def apply_brakes(S):
    angle = S.get('angle', 0.0)
    if abs(angle) > BRAKE_THRESHOLD:
        return 0.3
    return 0.0

def shift_gears(S):
    speed_x = S.get('speedX', 0.0)
    gear = 1
    for i, speed in enumerate(GEAR_SPEEDS):
        if speed_x > speed:
            gear = i + 1
    return min(gear, 6)

def traction_control(S, accel):
    if ENABLE_TRACTION_CONTROL:
        wsv = S.get('wheelSpinVel', [0.0]*4)
        # Slittamento ruote motrici posteriori
        if ((wsv[2] + wsv[3]) - (wsv[0] + wsv[1])) > 2.0:
            accel -= 0.15
    return max(0.0, accel)


def save_to_disk(buffer, headers, lap_number, lap_time):
    if not buffer: return
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    minutes = int(lap_time // 60)
    seconds = int(lap_time % 60)
    milliseconds = int((lap_time % 1) * 1000)
    time_str = f"{minutes:02d}-{seconds:02d}-{milliseconds:03d}"
    filename = f"lap_{lap_number:03d}_time_{time_str}_{timestamp}.csv"
    filepath = os.path.join(DATASET_DIR, filename)
    with open(filepath, "w", newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(buffer)
    print(f"\n>>> [GIRO SALVATO AUTOMATICAMENTE] Giro: {lap_number} | Tempo: {time_str}")


def main():
    parser = argparse.ArgumentParser(description="Registratore automatico di dataset via bot euristico.")
    parser.add_argument("--laps", type=int, default=10, help="Numero di giri da registrare prima di fermarsi (default: 10)")
    args = parser.parse_args()

    print("=" * 60)
    print(" AI-AutonomeGuide - RECORD DATASET AUTOMATICO (BOT ESPERTO)")
    print("=" * 60)
    print(f"Obiettivo: Registrare {args.laps} giri di guida perfetti.")
    print("-" * 60)

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

    # Stato della registrazione
    lap_number = 1
    prev_lap_time = 0.0
    prev_damage = 0.0
    buffer = []
    
    print("\nIn attesa di TORCS... (Avvia la corsa Practice/Race su TORCS)")

    try:
        while lap_number <= args.laps:
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

            # Leggi sensori
            cur_lap_time = S.d.get('curLapTime', 0.0)
            track_pos = S.d.get('trackPos', 0.0)
            damage = S.d.get('damage', 0.0)
            speed_x = S.d.get('speedX', 0.0)

            # 1. Rileva fine giro per salvare il CSV
            if cur_lap_time < prev_lap_time and prev_lap_time > 10.0:
                save_to_disk(buffer, HEADERS, lap_number, prev_lap_time)
                lap_number += 1
                buffer = []
                prev_lap_time = 0.0
                prev_damage = damage
                
                if lap_number > args.laps:
                    print(f"\n>>> Completati con successo tutti i {args.laps} giri richiesti!")
                    break
                    
                # Riavvia la simulazione per fermare la macchina sulla linea e ripartire puliti
                R.d['meta'] = 1
                so.sendto(str(R).encode(), (HOST, PORT))
                time.sleep(0.5)
                continue

            # 2. Rileva sbandata critica / fuori pista per riavviare (se succede)
            if abs(track_pos) > 1.3 or damage > (prev_damage + 100.0):
                print(f"\n>>> [ANNULLATO] Sbandata o urto rilevato. Giro scartato per sicurezza.")
                buffer = []
                R.d['meta'] = 1
                so.sendto(str(R).encode(), (HOST, PORT))
                prev_damage = damage
                prev_lap_time = 0.0
                time.sleep(0.5)
                continue

            # 3. Logica di guida euristica (l'esperto calcola le azioni)
            steer = calculate_steering(S.d)
            brake = apply_brakes(S.d)
            accel = calculate_throttle(S.d, steer, R.d['accel'])
            accel = traction_control(S.d, accel)
            gear = shift_gears(S.d)

            # Assegna le azioni
            R.d['steer'] = steer
            R.d['accel'] = accel
            R.d['brake'] = brake
            R.d['gear'] = gear
            R.d['meta'] = 0

            # 4. Registra i campioni nel buffer
            if cur_lap_time > 0.1 and speed_x > 2.0:
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
                row.extend(track_sens)
                # target sono i comandi ideali del bot euristico
                row.extend([steer, accel, brake])
                
                buffer.append(row)

                if int(cur_lap_time) % 5 == 0 and cur_lap_time - int(cur_lap_time) < 0.05:
                    print(f"\rRegistrazione Automatica... Giro {lap_number}/{args.laps} | Tempo: {int(cur_lap_time)}s | Velocità: {int(speed_x)} km/h", end="")

            prev_lap_time = cur_lap_time
            so.sendto(str(R).encode(), (HOST, PORT))

    except KeyboardInterrupt:
        print("\nInterruzione manuale del programma.")
    finally:
        so.close()

if __name__ == "__main__":
    main()
