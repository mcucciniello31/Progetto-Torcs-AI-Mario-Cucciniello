"""Script per l'immagazinamento di giri completi della pista (manuali) tramite dualshock per PS4"""

import pygame
import socket
import sys
import os
import time
import csv
import numpy as np

# Configurazione
HOST = 'localhost'
PORT = 3001
SID = 'SCR'
DATA_SIZE = 2**17
TRACK_LIMIT = 1.5 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATASET_DIR = os.path.join(BASE_DIR, "dataset_laps")

if not os.path.exists(DATASET_DIR):
    os.makedirs(DATASET_DIR)

# Mappatura DualShock PS4
AXIS_STEER = 0 #Analogico sinistro
AXIS_ACCEL = 5 #R2
AXIS_BRAKE = 4 #L2

class ServerState():
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

class DriverAction():
    def __init__(self):
        self.d = {'accel': 0, 'brake': 0, 'clutch': 0, 'gear': 1, 'steer': 0, 'focus': [-90, -45, 0, 45, 90], 'meta': 0}
    def __repr__(self):
        out = str()
        for k in self.d:
            out += '(' + k + ' '
            v = self.d[k]
            if not isinstance(v, list): out += '%.3f' % v
            else: out += ' '.join([str(x) for x in v])
            out += ')'
        return out

def get_joystick_input(joystick, current_speed):
    pygame.event.pump()
    raw_steer = -joystick.get_axis(AXIS_STEER)
    if abs(raw_steer) < 0.02:
        steer = 0.0
    else:
        steer = np.sign(raw_steer) * (abs(raw_steer) ** 2.0)
        if current_speed > 50:
            steer *= max(0.4, 1.0 - (current_speed - 50) / 300.0)
    accel = (joystick.get_axis(AXIS_ACCEL) + 1.0) / 2.0
    brake = (joystick.get_axis(AXIS_BRAKE) + 1.0) / 2.0
    return steer, accel, brake

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
    print(f"\n>>> [SALVATO] Giro {lap_number} | Tempo: {time_str}")

def manual_recording():
    # Registrazione e validazione dei giri, in che modo?
    # La distanza dei cardoli laterali, trackPos < 1.3 
    # Se si va fuori pista o si sbatte contro il muro, il giro viene resettato in maniera automatica
    # Se il giro è ok, eso viene salvato e si viene ricatapultati ad inizio pista (pronti per un altro giro)

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() == 0:
        print("ERRORE: Collega il controller!")
        return
    js = pygame.joystick.Joystick(0)
    js.init()

    so = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    so.settimeout(1.0)
    initmsg = f"{SID}(init -45 -19 -12 -7 -4 -2.5 -1.7 -1 -.5 0 .5 1 1.7 2.5 4 7 12 19 45)"
    
    so.sendto(initmsg.encode(), (HOST, PORT))
    while True:
        try:
            sockdata, _ = so.recvfrom(DATA_SIZE)
            if '***identified***' in sockdata.decode():
                print(">>> Connessione a TORCS riuscita. Registrazione giro pronta.")
                break
        except:
            so.sendto(initmsg.encode(), (HOST, PORT))

    S = ServerState()
    R = DriverAction()
    KEYS_TO_IGNORE = ['opponents', 'focus', 'fuel', 'damage', 'z', 'curLapTime', 'lastLapTime', 'distFromStart', 'distRaced', 'racePos']

    lap_buffer = []
    prev_lap_time = 0.0
    initial_damage = None
    is_dirty = False
    headers = None
    t0 = time.time()
    lap_counter = 0

    try:
    while True:
        try:
            sockdata, _ = so.recvfrom(DATA_SIZE)
            sockstr = sockdata.decode()
                
# Restart del giro (in caso di fuori Pista, fine giro o danno accidentale) Fine Giro) ---
            if '***restart***' in sockstr:
                print("\n Reset della pista in corso...")
                lap_buffer, is_dirty, prev_lap_time = [], False, 0.0
                initial_damage = None
                R.d['meta'] = 0
                t0 = time.time() 

                connected = False
                while not connected:
                    so.sendto(initmsg.encode(), (HOST, PORT))
                    try:
                        so.settimeout(0.5) 
                        resp, _ = so.recvfrom(DATA_SIZE)
                         if '***identified***' in resp.decode():
                            print(">>> POSIZIONATO SULLA GRIGLIA. PARTI!")
                            connected = True
                            so.settimeout(1.0)
                     except (socket.timeout, ConnectionResetError):
                    continue 
                continue
                
            S.parse_server_str(sockstr)
            
                     except (socket.timeout, ConnectionResetError):
                         if R.d['meta'] == 1:
                            so.sendto(repr(R).encode(), (HOST, PORT))
                continue

            if initial_damage is None:
                initial_damage = S.d.get('damage', 0)

            # Come gestire la corsa quando andiamo fuori dal tracciato
            track_pos = abs(S.d.get('trackPos', 0))
            if track_pos > TRACK_LIMIT:
                print(f"\n Attenzione!!! - Sei andato fuori pista! ({track_pos:.2f}). Reset del giro...")
                is_dirty = True
                R.d['meta'] = 1
                so.sendto(repr(R).encode(), (HOST, PORT))
                lap_buffer, prev_lap_time = [], 0.0
                continue

            # Come gestire la fine di un giro pulito
            cur_time = S.d.get('curLapTime', 0)
            if cur_time < prev_lap_time and prev_lap_time > 10.0:
                last_lap_time = S.d.get('lastLapTime', 0)
                
                if not is_dirty and len(lap_buffer) > 500:
                    lap_counter += 1
                    save_to_disk(lap_buffer, headers, lap_counter, last_lap_time)
                    
                    print("Giro completato. Restart della pista...")
                    R.d['meta'] = 1
                    so.sendto(repr(R).encode(), (HOST, PORT))
                    lap_buffer, is_dirty, prev_lap_time = [], False, 0.0
                    continue # Salta il resto e aspetta il segnale di restart
                else:
                    print(f">>> Attenzione!!! - Giro non valido.")
                
                lap_buffer, is_dirty = [], False
                initial_damage = S.d.get('damage', 0)
            
            prev_lap_time = cur_time

            # Come gestire dei danni accidentali
            if (S.d.get('damage', 0) - initial_damage) > 1.0:
                if not is_dirty: print("\nAttenzione!!! - Giro con danni.")
                is_dirty = True

            # Come gestire la guida in pista
            speed = S.d.get('speedX', 0)
            steer, accel, brake = get_joystick_input(js, speed)

            wheel_vel = S.d.get('wheelSpinVel', [0,0,0,0])
            if len(wheel_vel) == 4:
                if (wheel_vel[2]+wheel_vel[3]) - (wheel_vel[0]+wheel_vel[1]) > 15: accel *= 0.5
                if brake > 0.1 and speed > 15 and (wheel_vel[0]+wheel_vel[1])/2.0 < 5: brake *= 0.1

            if brake > 0.1 and abs(steer) > 0.15: 
                brake *= (1.0 - abs(steer)*0.8)

            #target_gear = 1 ---> marcia iniziale= la 1°
            #for i, th in enumerate([0, 45, 90, 145, 200, 260]):
            #    if speed > th: target_gear = i + 1
            #gear = S.d.get('gear', 1) if abs(steer) > 0.4 else target_gear

            # Implementazione di un cambio automatica --> A causa di problemi col joystick-
            target_gear = 1
            if speed > 50:
                target_gear = 2
            if speed > 90:
                target_gear = 3
            if speed > 150:
                target_gear = 4
            if speed > 200:
                target_gear = 5
            if speed > 280:
                target_gear = 6

            # Se stiamo sterzando abbastanzao, non cambiamo marcia altrimenti perdiamo grip (sottosterziamo/sovrasterziamo)
            gear = S.d.get('gear', 1) if abs(steer) > 0.4 else target_gear

            # Come gestire restart manuale (da controller tramite questi 3 tasti: Triangolo=3, Share=8, Options=9)
            meta = 0
            for btn_id in [3, 8, 9]:
                if btn_id < js.get_numbuttons() and js.get_button(btn_id):
                    meta = 1
                    print("\n>>>Attenzione!!! - Riavvio richiesto tramite controller...")
                    break

            R.d['steer'], R.d['accel'], R.d['brake'], R.d['gear'], R.d['meta'] = steer, accel, brake, gear, meta

            # Come gestire la registrazione di un giro pulito col joystick
            if headers is None:
                headers = ["timestamp", "target_steer", "target_accel", "target_brake", "target_gear"]
                for k in sorted(S.d.keys()):
                    if k in KEYS_TO_IGNORE: continue
                    val = S.d[k]
                    if isinstance(val, list): headers.extend([f"{k}_{i}" for i in range(len(val))])
                    else: headers.append(k)

            row = [time.time()-t0, steer, accel, brake, gear]
            for k in sorted(S.d.keys()):
                if k in KEYS_TO_IGNORE: continue
                val = S.d[k]
                if isinstance(val, list): row.extend(val)
                else: row.append(val)
            lap_buffer.append(row)

            so.sendto(repr(R).encode(), (HOST, PORT))

    except KeyboardInterrupt: print("\nUscita.")
    finally:
        so.close()
        pygame.quit()

if __name__ == "__main__":
    manual_recording()
