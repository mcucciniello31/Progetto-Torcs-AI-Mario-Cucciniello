import pygame
import time

def main():
    pygame.init()
    pygame.joystick.init()
    
    count = pygame.joystick.get_count()
    if count == 0:
        print("Nessun controller rilevato! Assicurati che sia connesso tramite Bluetooth o cavo USB.")
        pygame.quit()
        return

    print(f"Rilevati {count} controller:")
    for i in range(count):
        js = pygame.joystick.Joystick(i)
        js.init()
        print(f"[{i}]: {js.get_name()}")

    js = pygame.joystick.Joystick(0)
    print("\n--- TEST SENSORI JOYSTICK (Premi Ctrl+C per uscire) ---")
    print("Muovi le levette e premi i grilletti per vedere gli indici degli assi...")
    
    try:
        while True:
            pygame.event.pump()
            num_axes = js.get_numaxes()
            axes_states = []
            for a in range(num_axes):
                val = js.get_axis(a)
                axes_states.append(f"Asse {a}: {val:.2f}")
            
            # Stampa gli stati degli assi su un'unica riga sovrascrivibile
            print("\r" + " | ".join(axes_states), end="", flush=True)
            time.sleep(0.1)
    except KeyboardInterrupt:
        print("\nTest terminato.")
    finally:
        pygame.quit()

if __name__ == "__main__":
    main()
