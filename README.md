# Autonomous-Fault-Detection-Model
A machine learning project designed to automatically detect system faults or anomalies in real time using sensor data. This model applies supervised and unsupervised learning techniques — including Random Forest, SVM, and Autoencoders — to classify normal vs. faulty behavior.


---

## Project Overview  

Aircraft engines and other complex mechanical systems generate vast amounts of sensor data during operation. Detecting faults early is crucial for safety and cost efficiency.  
This project leverages **Autoencoders**, a neural network architecture that learns to reconstruct normal sensor patterns and flags anomalies when deviations occur.

Using NASA’s **CMAPSS Turbofan Engine Degradation Simulation Dataset**, the model automatically identifies abnormal sensor behavior that indicates potential equipment failures.  

---

## Objectives  

- Build an **unsupervised anomaly detection system** using Autoencoders.  
- Detect early signs of **engine degradation or malfunction**.  
- Analyze sensor data trends across different engine cycles.  
- Visualize reconstruction errors and flag anomalies over time.  

---

## Dataset Description  

**Source:** NASA’s **Commercial Modular Aero-Propulsion System Simulation (CMAPSS)** dataset.  
This dataset simulates degradation data for multiple jet engines operating under various conditions.

### 📁 Files Used:
- `train_FD001.txt` – Training data (sensor readings from multiple engines)  
- `test_FD001.txt` – Test data for validation  
- `RUL_FD001.txt` – Remaining useful life labels (not used for unsupervised training)

Each record represents one cycle (time step) of an engine, with **26 columns** including:
- **ID columns:** `engine_id`, `cycle`
- **Operational settings:** `setting1`, `setting2`, `setting3`
- **Sensor readings:** `sensor1` through `sensor21`

### Example:
| engine_id | cycle | setting1 | setting2 | sensor1 | sensor2 | sensor3 | ... |
|------------|--------|-----------|-----------|----------|----------|----------|----|
| 1 | 1 | -0.0007 | 0.0000 | 518.67 | 641.82 | 1589.70 | ... |

---

## Workflow  

The model follows a standard **machine learning pipeline** for unsupervised fault detection.

### 1. **Data Preprocessing**
- Loaded sensor data from `train_FD001.txt`.  
- Dropped irrelevant columns (`engine_id`, `cycle`, and settings).  
- Normalized all 21 sensor readings using **MinMaxScaler** for stability.

### 2. **Autoencoder Architecture**
- Input Layer: 21 neurons  
- Encoding Layers: progressively compress data (14 → 7 → 3)  
- Decoding Layers: reconstruct input (3 → 7 → 14 → 21)  
- Activation: ReLU (Hidden layers), Linear (Output layer)

### 3. **Model Training**
- Optimizer: `adam`  
- Loss: Mean Squared Error (MSE)  
- Epochs: 50  
- Batch size: 32  
- Early stopping based on validation loss to prevent overfitting  

Training achieved a **loss ≈ 0.0024**, indicating the model effectively learned normal patterns.

### 4. **Reconstruction Error**
After training, the Autoencoder reconstructs each sample.  
The **Mean Squared Error (MSE)** between input and output is computed:
```python
recon_error = np.mean((X - X_hat)**2, axis=1)
The reconstruction error serves as an **anomaly score**, where higher values indicate unusual sensor patterns that may signify faults.

---

## Thresholding & Anomaly Detection  

To distinguish between normal and faulty readings, a **threshold** is applied to the reconstruction errors.  
Two approaches were tested:

1. **Percentile Rule (99th percentile):**  
   Marks the top 1% of readings with the highest errors as anomalies.  

2. **Mean + 3×Standard Deviation:**  
   Defines anomalies as readings that deviate significantly (3σ) from the mean reconstruction error.

In this project, the **99th percentile rule** was selected for consistency and interpretability.

| Metric | Value |
|:--------|:------:|
| Threshold | 0.00587 |
| Anomaly Rate | 1% |
| Detection Type | Unsupervised |

---

## Visual Analysis  

### 🔹 Reconstruction Error Distribution  
A histogram visualizes how the majority of reconstruction errors cluster near zero (normal behavior),  
while a small fraction exceed the threshold—these are **potential faults**.

### 🔹 Engine 1 – Reconstruction Error over Cycles  
This plot shows how reconstruction error evolves over time for one engine.  
Spikes near the end of life correspond to **mechanical degradation** or **fault conditions**.

### 🔹 Engine 1 – Anomaly Flags  
Displays a binary fault indicator (1 = anomaly, 0 = normal) for each cycle.  
A rise in anomalies indicates the system is entering a critical failure phase.

---

## Results Summary  

- The Autoencoder successfully learned normal operational behavior of the engines.  
- Training and validation losses both converged around **0.0024**, showing stable learning.  
- Anomalies (1% of data) were detected near the end-of-life cycles, aligning with expected degradation patterns.  
- Visualization confirmed that **reconstruction errors increased sharply before faults occurred**, validating the model’s predictive potential.

---

## Key Insights  

1. **Unsupervised Learning Strength:**  
   Autoencoders can detect faults even without labeled data, making them ideal for early warning systems.

2. **Low Reconstruction Loss:**  
   Indicates that the model has captured the normal sensor patterns effectively.

3. **Explainability:**  
   Reconstruction error trends are easily interpretable by maintenance engineers.

4. **Scalability:**  
   The same framework can be extended to other subsets (`FD002–FD004`) or industrial sensors.

---

## Future Improvements  

- Extend model to **multivariate fault prediction** using stacked or convolutional autoencoders.  
- Combine Autoencoder output with **LSTM layers** to predict Remaining Useful Life (RUL).  
- Integrate results into a **real-time monitoring dashboard** for predictive maintenance.  
- Experiment with **different thresholding methods** (e.g., Isolation Forest, IQR).

---

## Repository Structure
```
fault_detection/
├── README.md
├── requirements.txt
├── .gitignore
│
├── Autoencoder_Fault_Detection_Model.ipynb
│
├── data/
│ ├── train_FD001.txt
│ ├── test_FD001.txt
│ ├── RUL_FD001.txt
│ └── readme.txt
│
├── models/
│ ├── autoencoder_fault_detection.keras
│ └── minmax_scaler.pkl
│
└── outputs/
├── anomaly_scores_FD001.csv
├── reconstruction_error_plot.png
├── engine1_anomalies.png
└── training_loss_curve.png
```
---
