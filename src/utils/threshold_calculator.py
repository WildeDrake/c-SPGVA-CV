import os
import numpy as np
import data_process as dp
import matplotlib.pyplot as plt

ROOT_DIR = "../../dataset"
WINDOW_SIZE = 52
HANDS = ["LEFT", "RIGHT"]

all_means = []
all_labels = []

subjects = sorted([s for s in os.listdir(ROOT_DIR) if s.startswith("s")])

for subj in subjects:
    subj_path = os.path.join(ROOT_DIR, subj, "all")
    for hand in HANDS:
        hand_path = os.path.join(subj_path, hand)
        if not os.path.exists(hand_path):
            continue
        for f in os.listdir(hand_path):
            if not f.endswith(".txt"):
                continue
            gesture_name = f.split("-")[0]
            file_path = os.path.join(hand_path, f)
            data = dp.txt2array(file_path)
            data = dp.preprocessing(data)
            start, end = dp.detect_muscle_activity(data)
            start, end = int(start), int(end)
            activation = data[:, start:end]

            # Ventanas deslizantes
            for i in range(0, activation.shape[1] - WINDOW_SIZE + 1):
                window = activation[:, i:i + WINDOW_SIZE].astype(np.float32)
                mean_activation = np.mean(np.abs(window))
                all_means.append(mean_activation)
                all_labels.append(gesture_name)

all_means = np.array(all_means)

# ------------------ Percentiles y sugerencias ------------------ #
percentiles = [0.1, 0.5, 1, 5, 10]  # percentiles bajos
for p in percentiles:
    val = np.percentile(all_means, p)
    print(f"Percentil {p}% de activación media: {val:.4f}")

# ------------------ Histograma ------------------ #
plt.figure(figsize=(10,6))
plt.hist(all_means, bins=200, color='skyblue', alpha=0.7)
plt.xlabel("Activación media de ventana")
plt.ylabel("Cantidad de ventanas")
plt.title("Distribución de activación media (todos los gestos)")
plt.show()
