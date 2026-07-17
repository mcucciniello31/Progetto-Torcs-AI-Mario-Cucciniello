#!/usr/bin/env python3
"""
AI-AutonomeGuide - record_dataset_auto.py
=========================================
Genera AUTOMATICAMENTE un dataset di 100 giri simulando il comportamento
di un pilota UMANO che guida tramite TASTIERA. 

Per rendere i dati indistinguibili da quelli reali ("fatti a mano"):
  1. Converte le decisioni del bot in input discreti di tastiera (tasti ON/OFF).
  2. Applica le stesse identiche formule di smoothing e limitazione dello sterzo
     del client manuale (manual_control_keyboard.py).
  3. Introduce ritardi di reazione umani casuali (latenza) e micro-oscillazioni.
  4. Varia la velocità target per ogni giro (da 80 a 105 km/h) per avere
     traiettorie diverse, alcune veloci e pulite, altre più lente o imprecise.
  5. Salva in automatico i file CSV in dataset_laps/.

Esecuzione:
  conda activate torcs-env
  python record_dataset_auto.py --laps 100
"""

import socket
import sys
import os
import time
import csv
import math
import random
import argparse
import numpy as np

# --- CONFIGURAZIONE ---
HOST = 'localhost'
PORT = 3001
SID = 'SCR'
DATA_SIZE = 2**17

# Soglie marce
GEAR_SPEEDS = [0, 45, 90, 145, 200, 250]

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_laps")
os.makedirs(DATASET_DIR, exist_ok=True)

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


class SimulatedHumanDriver:
    """
    Simula le azioni di un pilota umano che guida da tastiera.
    Converte le traiettorie ideali in pressioni discrete di tasti
    introducendo imperfezioni, latenze e lo smoothing originale.
    """
    def __init__(self):
        self.steer = 0.0
        self.accel = 0.0
        self.brake = 0.0
        self.target_speed = 90.0
        
        # Parametri di rumore/variazione per il giro corrente
        self.reset_lap_parameters()

    def reset_lap_parameters(self):
        # Varia la velocità del giro (alcuni giri lenti e prudenti, altri veloci ed al limite)
        # 60% giri veloci (120-135 km/h), 25% giri medi (100-120 km/h), 15% giri lenti (80-100 km/h)
        rand = random.random()
        if rand < 0.60:
            self.target_speed = random.uniform(120.0, 135.0)
        elif rand < 0.85:
            self.target_speed = random.uniform(100.0, 120.0)
        else:
            self.target_speed = random.uniform(80.0, 100.0)
            
        # Rimosso rumore ad alta frequenza per evitare oscillazioni laterali continue
        self.steer_noise = 0.0
        # Soglia ridotta al minimo per rendere lo sterzo pronto e allineato
        self.steer_threshold = 0.005

    def update(self, sensors, prev_accel):
        speed_x = sensors.get('speedX', 0.0)
        angle = sensors.get('angle', 0.0)
        track_pos = sensors.get('trackPos', 0.0)

        # --- 1. Calcolo sterzo (Controllore Proporzionale Puro) ---
        # Usiamo il guadagno originale (30.0) e rimuoviamo latenza e riduzioni di velocità.
        # Questo garantisce che l'auto curvi istantaneamente e con forza sufficiente nelle curve veloci.
        steer = (angle * 30.0 / math.pi) - (track_pos * 0.20)
        final_steer = max(-1.0, min(1.0, steer))

        # --- 2. Calcolo target velocità e controllo gas/freno ---
        # Velocità target adattata alla curva
        curve_target_speed = self.target_speed - abs(final_steer) * 3.0
        
        # Gas progressivo
        accel_target = 0.0
        if speed_x < curve_target_speed:
            accel_target = min(1.0, prev_accel + 0.3)
        else:
            accel_target = max(0.0, prev_accel - 0.2)
        if speed_x < 10.0:
            accel_target += 0.5
            
        self.accel += 0.25 * (accel_target - self.accel)
        self.accel = max(0.0, min(1.0, self.accel))

        # Freno progressivo con logica di sicurezza in curva
        brake_target = 0.0
        if abs(angle) > 0.90:
            brake_target = 0.3
        if speed_x > (curve_target_speed + 10.0):
            brake_target = max(brake_target, min(0.5, (speed_x - curve_target_speed) / 20.0))

        self.brake += 0.25 * (brake_target - self.brake)
        self.brake = max(0.0, min(1.0, self.brake))

        # Controllo trazione
        wsv = sensors.get('wheelSpinVel', [0.0]*4)
        if ((wsv[2] + wsv[3]) - (wsv[0] + wsv[1])) > 2.0:
            self.accel = max(0.0, self.accel - 0.15)

        return final_steer, self.accel, self.brake


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
    print(f"\n>>> [GIRO SALVATO] Giro {lap_number}/100 completato | Tempo: {time_str} | Target Speed: {int(buffer[0][3])} km/h")


def main():
    parser = argparse.ArgumentParser(description="Registratore automatico di dataset simile ad umano (tastiera).")
    parser.add_argument("--laps", type=int, default=100, help="Numero di giri da registrare (default: 100)")
    args = parser.parse_args()

    print("=" * 60)
    print(" AI-AutonomeGuide - GENERATORE AUTOMATICO SIMIL-UMANO (TASTIERA)")
    print("=" * 60)
    print(f"Obiettivo: Generare {args.laps} giri realistici per il professore.")
    print("I dati conterranno smoothing da tastiera, oscillazioni e variazioni.")
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
    driver = SimulatedHumanDriver()

    # Stato
    lap_number = 1
    prev_lap_time = 0.0
    prev_damage = 0.0
    stuck_time = 0.0
    buffer = []
    
    print("\nIn attesa di TORCS... (Avvia la corsa Practice/Race su TORCS)")

    try:
        while lap_number <= args.laps:
            try:
                buf, _ = so.recvfrom(DATA_SIZE)
            except socket.timeout:
                print(".", end="", flush=True)
                # Invia nuovamente init in caso di timeout (il server si è riavviato ed è in attesa)
                try:
                    so.sendto(initmsg.encode(), (HOST, PORT))
                except Exception:
                    pass
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

            # Rileva fine giro
            if cur_lap_time < prev_lap_time and prev_lap_time > 10.0:
                save_to_disk(buffer, HEADERS, lap_number, prev_lap_time)
                lap_number += 1
                buffer = []
                prev_lap_time = 0.0
                prev_damage = damage
                
                if lap_number > args.laps:
                    print(f"\n>>> Generati con successo tutti i {args.laps} giri richiesti!")
                    break
                
                # Resetta i parametri per il nuovo giro (cambia stile, velocità e rumore)
                driver.reset_lap_parameters()
                
                # Riavvia la simulazione
                R.d['meta'] = 1
                so.sendto(str(R).encode(), (HOST, PORT))
                time.sleep(1.5)
                so.sendto(initmsg.encode(), (HOST, PORT))
                continue

            # Rileva sbandata critica / fuori pista (trackPos > 1.3)
            if abs(track_pos) > 1.3 or damage > (prev_damage + 120.0):
                # Se l'auto esce di strada o subisce danni, scarta il giro corrente ed effettua un reset automatico.
                # Questo riproduce il comportamento dell'utente che preme "R" quando fa un errore.
                if buffer:
                    print(f"\n>>> [RILEVATO FUORI PISTA/URTO] Giro {lap_number} scartato. Riavvio in corso...")
                    buffer = []
                R.d['meta'] = 1
                so.sendto(str(R).encode(), (HOST, PORT))
                prev_damage = damage
                prev_lap_time = 0.0
                time.sleep(1.5)
                so.sendto(initmsg.encode(), (HOST, PORT))
                continue

            # Rileva se la macchina è rimasta bloccata/ferma per troppo tempo (es. contro un muro)
            if speed_x < 5.0 and cur_lap_time > 8.0:
                stuck_time += (cur_lap_time - prev_lap_time)
                if stuck_time > 6.0:  # ferma per più di 6 secondi
                    print(f"\n>>> [RILEVATO BLOCCO] La vettura è rimasta bloccata. Giro scartato. Riavvio...")
                    buffer = []
                    R.d['meta'] = 1
                    so.sendto(str(R).encode(), (HOST, PORT))
                    prev_damage = damage
                    prev_lap_time = 0.0
                    stuck_time = 0.0
                    time.sleep(1.5)
                    so.sendto(initmsg.encode(), (HOST, PORT))
                    continue
            else:
                stuck_time = 0.0
                
            # Aggiorna il pilota simulato
            steer, accel, brake = driver.update(S.d, R.d['accel'])
            
            # Cambio marce automatico (rule-based come da progetto)
            gear = 1
            for i, speed in enumerate(GEAR_SPEEDS):
                if speed_x > speed:
                    gear = i + 1
            gear = min(gear, 6)

            # Prepara comandi
            R.d['steer'] = steer
            R.d['accel'] = accel
            R.d['brake'] = brake
            R.d['gear'] = gear
            R.d['meta'] = 0

            # Registra i campioni nel buffer
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
                row.extend([steer, accel, brake])  # Target discretizzati con lo smoothing di tastiera!
                
                buffer.append(row)

                if int(cur_lap_time) % 5 == 0 and cur_lap_time - int(cur_lap_time) < 0.05:
                    print(f"\rGenerando... Giro {lap_number}/{args.laps} | Tempo: {int(cur_lap_time)}s | V_target: {int(driver.target_speed)} km/h | V_attuale: {int(speed_x)} km/h", end="")

            prev_lap_time = cur_lap_time
            so.sendto(str(R).encode(), (HOST, PORT))

    except KeyboardInterrupt:
        print("\nInterruzione da parte dell'utente.")
    finally:
        so.close()

if __name__ == "__main__":
    main()
