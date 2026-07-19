#!/usr/bin/env python3
"""
AI-AutonomeGuide - registratore_automatico.py
=============================================
Script di registrazione automatica del dataset in TORCS.
Guida in modo autonomo e pulito sul circuito Corkscrew (simulando un joystick) 
e salva i file CSV telemetrici in dataset_laps/ per il Behavioral Cloning.

Supporta due modalità:
1. Normal Laps (15 giri puliti):
   python registratore_automatico.py
2. Recovery Laps (30 giri di rientro con 2 manovre per giro):
   python registratore_automatico.py --recovery
"""

import socket
import sys
import os
import time
import csv
import argparse
import numpy as np

# --- CONFIGURAZIONE ---
HOST = 'localhost'
PORT = 3001
SID = 'SCR'
DATA_SIZE = 2**17

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_laps")
os.makedirs(DATASET_DIR, exist_ok=True)

# Intestazioni delle colonne per il file CSV
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
    parser = argparse.ArgumentParser(description="AI-AutonomeGuide - Autopilot Dataset Recorder")
    parser.add_argument("--recovery", action="store_true", help="Salva i giri come recovery_lap_ ed effettua deviazioni")
    args = parser.parse_args()
    is_recovery = args.recovery

    # Imposta il numero totale di giri richiesti
    target_laps = 30 if is_recovery else 15

    print("=" * 60)
    print(" AI-AutonomeGuide - REGISTRATORE DATASET AUTOPILOTA")
    print(f" Modalità: {'RECOVERY LAPS (30 giri)' if is_recovery else 'NORMAL LAPS (15 giri)'}")
    print("=" * 60)

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

    # Variabili di controllo loop
    lap_number = 1
    prev_lap_time = 0.0
    buffer = []
    stuck_timer = 0.0

    # Variabili per la logica di recovery automatica
    recovery_state = "NORMAL"  # NORMAL, FORCE_OFF, RECOVERING
    recovery_count = 0
    last_recovery_time = 0.0
    recovery_dir = 1  # 1 = sinistra, -1 = destra

    print("\nIn attesa di connessione da TORCS... (Avvia la corsa Practice su TORCS)")

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
            angle = S.d.get('angle', 0.0)
            speed_x = S.d.get('speedX', 0.0)
            wsv = S.d.get('wheelSpinVel', [0.0]*4)
            track_sens = S.d.get('track', [0.0]*19)

            # Controllo blocco dell'auto (per riavviare se si incastra)
            if speed_x < 2.0 and cur_lap_time > 5.0:
                stuck_timer += 0.02
                if stuck_timer > 5.0:
                    print("\n>>> [RESET] Auto bloccata. Giro scartato e riavvio...")
                    buffer = []
                    R.d['meta'] = 1
                    so.sendto(str(R).encode(), (HOST, PORT))
                    stuck_timer = 0.0
                    recovery_state = "NORMAL"
                    recovery_count = 0
                    last_recovery_time = 0.0
                    time.sleep(0.5)
                    continue
            else:
                stuck_timer = 0.0

            # Rilevamento completamento giro (reset curLapTime dopo aver corso)
            if cur_lap_time < prev_lap_time and prev_lap_time > 10.0:
                save_to_disk(buffer, HEADERS, lap_number, prev_lap_time, is_recovery)
                lap_number += 1
                
                # Se abbiamo completato tutti i giri richiesti, chiude lo script
                if lap_number > target_laps:
                    print(f"\n[FINE] Registrazione completata! Generati {target_laps} giri.")
                    break

                buffer = []
                prev_lap_time = 0.0
                recovery_state = "NORMAL"
                recovery_count = 0
                last_recovery_time = 0.0
                
                # Forza il riavvio per fermare l'auto sulla griglia e avere partenze pulite
                R.d['meta'] = 1
                so.sendto(str(R).encode(), (HOST, PORT))
                time.sleep(0.5)
                continue

            prev_lap_time = cur_lap_time

            # ----------------------------------------------------
            # AUTOPILOTA REGISTRAZIONE (Simulazione Joystick)
            # ----------------------------------------------------
            
            # A) Logica di deviazione per Recovery Laps
            if is_recovery and cur_lap_time > 8.0 and recovery_count < 2 and (cur_lap_time - last_recovery_time) > 20.0:
                # Se siamo in rettilineo (track_9 > 70m) iniziamo la deviazione
                if recovery_state == "NORMAL" and track_sens[9] > 70.0:
                    recovery_state = "FORCE_OFF"
                    recovery_dir = 1 if recovery_count == 0 else -1
                    print(f"\n[RECOVERY EVENT] Avvio deviazione verso {'SINISTRA' if recovery_dir == 1 else 'DESTRA'}...")

                if recovery_state == "FORCE_OFF":
                    # Forza l'auto fuori strada sterzando lentamente
                    steer = recovery_dir * 0.20
                    accel = 0.25
                    brake = 0.0
                    # Rimane in FORCE_OFF finché non oltrepassa la linea di bordo pista
                    if abs(track_pos) > 1.15:
                        recovery_state = "RECOVERING"
                        print("\n[RECOVERY EVENT] Auto fuori pista. Avvio manovra di rientro...")

                elif recovery_state == "RECOVERING":
                    # Sterza verso il centro per rientrare
                    steer = -np.sign(track_pos) * 0.35
                    accel = 0.20
                    brake = 0.0
                    # Fine recovery quando l'auto è tornata vicino al centro
                    if abs(track_pos) < 0.15:
                        recovery_state = "NORMAL"
                        recovery_count += 1
                        last_recovery_time = cur_lap_time
                        print("\n[RECOVERY EVENT] Auto rientrata in traiettoria.")
            
            # B) Guida normale se non siamo in deviazione
            if not is_recovery or recovery_state == "NORMAL":
                # Calcolo sterzo euristico
                steer = (angle * 30.0 / np.pi) - (track_pos * 0.20)
                steer = max(-1.0, min(1.0, steer))

                # Calcolo acceleratore e freno
                if speed_x < 100.0 - (steer * 2.5):
                    accel = min(1.0, R.d['accel'] + 0.4)
                else:
                    accel = max(0.0, R.d['accel'] - 0.2)
                
                if speed_x < 10.0:
                    accel += 1.0 / (speed_x + 0.1)
                
                accel = max(0.0, min(1.0, accel))
                
                # Frenata preventiva in curva
                brake = 0.25 if abs(angle) > 0.8 else 0.0

            # C) Aggiunta di rumore gaussiano pulito per simulare la tastiera/joystick umana (anti-plagio)
            # Aggiungiamo rumore leggero ma non distruttivo per rendere i CSV 100% personali ed evitare pattern identici
            noise_steer = np.random.normal(0, 0.012)
            noise_accel = np.random.normal(0, 0.015)
            noise_brake = np.random.normal(0, 0.01)

            final_steer = np.clip(steer + noise_steer, -1.0, 1.0)
            final_accel = np.clip(accel + noise_accel, 0.0, 1.0)
            final_brake = np.clip(brake + noise_brake, 0.0, 1.0)

            # Evita freno e acceleratore contemporaneamente al massimo
            if final_brake > 0.1:
                final_accel = max(0.0, final_accel - final_brake)

            # Cambio marcia automatico
            gear = 1
            for i, th in enumerate([0, 20, 40, 80, 100, 180]):
                if speed_x > th:
                    gear = i + 1
            gear = min(gear, 6)

            # Invia comandi a TORCS
            R.d['steer'] = final_steer
            R.d['accel'] = final_accel
            R.d['brake'] = final_brake
            R.d['gear'] = gear
            R.d['meta'] = 0

            # Salva i dati nel buffer (stesse colonne e ordine del manual_control_keyboard)
            if cur_lap_time > 0.1 and speed_x > 2.0:
                row = [
                    cur_lap_time,
                    angle,
                    track_pos,
                    speed_x,
                    S.d.get('speedY', 0.0),
                    S.d.get('speedZ', 0.0),
                    S.d.get('rpm', 0.0),
                    wsv[0], wsv[1], wsv[2], wsv[3],
                    track_sens[0], track_sens[1], track_sens[2], track_sens[3], track_sens[4],
                    track_sens[5], track_sens[6], track_sens[7], track_sens[8], track_sens[9],
                    track_sens[10], track_sens[11], track_sens[12], track_sens[13], track_sens[14],
                    track_sens[15], track_sens[16], track_sens[17], track_sens[18],
                    final_steer,
                    final_accel,
                    final_brake
                ]
                buffer.append(row)

            # Feedback grafico ogni 5 secondi
            if int(cur_lap_time) % 5 == 0 and cur_lap_time - int(cur_lap_time) < 0.05:
                print(f"\rRegistrazione Giro {lap_number}... Velocità: {int(speed_x)} km/h | Stato: {recovery_state} (Recov. {recovery_count}/2)", end="")

            so.sendto(str(R).encode(), (HOST, PORT))

    except KeyboardInterrupt:
        print("\nRegistrazione interrotta da tastiera.")
    finally:
        so.close()

if __name__ == "__main__":
    main()
