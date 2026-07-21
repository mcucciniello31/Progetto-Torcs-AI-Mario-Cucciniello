# Progetto Guida Autonoma in TORCS (Behavioral Cloning via K-NN)

Questo repository contiene il progetto di guida autonoma sviluppato per l'insegnamento di **Artificial Intelligence: Metodi e Applicazioni**. L'obiettivo è realizzare un agente intelligente in grado di guidare autonomamente in pista all'interno del simulatore TORCS mediante tecniche di **Behavioral Cloning (Imitation Learning)**.

---

## 🎮 Software e Strumenti Utilizzati

Il progetto si basa sulla combinazione di simulatori fisici e librerie scientifiche in Python:
* **TORCS (The Open Racing Car Simulator):** Simulatore open-source di corse automobilistiche utilizzato come ambiente di test.
* **SCR (Simulated Car Racing):** Client di connessione in tempo reale che scambia telemetria e comandi con il simulatore tramite protocollo UDP.
* **Python 3:** Utilizzato per l'intera pipeline di data science e controllo con le seguenti librerie principali:
  * `scikit-learn` (per l'addestramento del regressore `KNeighborsRegressor` e la pipeline di normalizzazione)
  * `pandas` & `numpy` (per l'elaborazione dati e manipolazione dei dataset)
  * `matplotlib` & `seaborn` (per l'analisi esplorativa dei dati e la generazione dei grafici prestazionali)

---

## 🎯 Intento del Progetto

Il progetto implementa un approccio di **apprendimento per imitazione**. Registrando la telemetria durante sessioni di guida umana (sensori laser, velocità, angoli di orientamento) e le relative azioni sui pedali e sullo sterzo, viene addestrato un modello statistico **K-Nearest Neighbors (KNN)**. 

Durante la guida autonoma, il modello riceve in tempo reale lo stato dei sensori e predice istante per istante i valori ottimali di:
1. **Sterzata (Steer)**
2. **Accelerazione (Accel)**
3. **Frenata (Brake)**

### ⚙️ Ottimizzazioni Fisiche e Controlli Dinamici
Per superare i limiti naturali di "smoothing" (appiattimento della media) del KNN, sono stati integrati i seguenti aiuti alla guida nel codice client:
* **Boost Sterzo a `1.2x` e Freno a `2.5x`:** Amplificano le predizioni del KNN per garantire inserimenti in curva precisi e staccate decise.
* **Frenata Preventiva Chicane (tarata a `19 metri`):** Se l'auto procede sul dritto a velocità elevata (>70 km/h) e rileva un ostacolo imminente a meno di 19 metri, forza la frenata a 0.7 e taglia il gas per impostare l'ingresso nella chicane. Non appena l'auto inizia a sterzare, il controllo forzato si disattiva per lasciar scorrere il veicolo in curva.
* **ABS & Traction Control (TCS):** Sistemi base per prevenire il bloccaggio delle ruote anteriori e il pattinamento di quelle motrici posteriori.

---

## 📂 Struttura della Cartella `media/`

All'interno della cartella `media/` puoi trovare i file multimediali di presentazione del progetto:
* [corkscrew.png](file:///Users/macucc/Downloads/AI-AutonomeGuide/media/Cornscrew.png) - Mappa dettagliata e andamento altimetrico della celebre curva.
* [giro_migliore_Ai.png](file:///Users/macucc/Downloads/AI-AutonomeGuide/media/giro_migliore_Ai.png) - Screenshot dell'auto guidata dall'AI in percorrenza.
* [giro_migliore_AI.mov](file:///Users/macucc/Downloads/AI-AutonomeGuide/media/giro_migliore_AI.mov) - Video del giro migliore registrato in risoluzione ottimizzata, che dimostra la stabilità dell'agente KNN durante la corsa.

---

## 🛠️ Passaggi Sequenziali per l'Esecuzione

Segui i passaggi in ordine per preparare i dati, addestrare il modello e lanciare la guida autonoma sul tuo PC:

### Passo 1: Preparazione e Pulizia del Dataset
Unisce tutti i file CSV dei giri salvati in `dataset_laps/`, effettua la pulizia dei dati e bilancia i campioni per dare peso alle fasi di frenata:
```bash
python preparazione_dataset.py
```
*Questo genererà il dataset pulito in `models/dataset_clean.csv`, lo scaler e i grafici esplorativi in `plots/`.*

### Passo 2: Addestramento del Modello KNN
Addestra il regressore KNN sul dataset preparato al passo precedente, mostrando a schermo la definizione delle metriche e i relativi score di precisione sul test set:
```bash
python addestramento_knn.py
```
*Verranno generati i file del modello `knn_model.pkl` e i grafici di residui e predizioni.*

### Passo 3: Avvio della Guida Autonoma in TORCS
1. Avvia **TORCS** dal terminale o dalle applicazioni.
2. Vai su **Race ➔ Practice ➔ Configure Race** per impostare la pista e l'auto.
3. Seleziona **New Race** (il simulatore andrà in attesa del client UDP sulla porta 3001).
4. Avvia lo script client dal tuo terminale:
```bash
python guida_autonoma_knn.py
```
