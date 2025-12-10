import os
import torch
import numpy as np
from utils.semgdata_loader import semgdata_load

ROOT_preprocessed = "./preprocessed_dataset/cross_subject/"
SPLITS = ["Healthy", "DMD", "Neuropathy", "Parkinson", "Stroke", "Artifact", "ALS"]

for split in SPLITS:
    print(f"\n=== Split: {split} ===")
    dataset = semgdata_load(root=ROOT_preprocessed, split=split, shuffle=False)
    all_data = dataset.data.squeeze(1).numpy()

    n_samples, n_channels, window_size = all_data.shape
    print(f"Número de muestras: {n_samples}, Canales: {n_channels}, Window size: {window_size}")

    for ch in range(n_channels):
        channel_data = all_data[:, ch, :].flatten()
        print(f"Canal {ch+1}: min={channel_data.min():.2f}, max={channel_data.max():.2f}, mean={channel_data.mean():.2f}, std={channel_data.std():.2f}")

    # histograma de intensidades para ver distribución
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 5))
    plt.hist(all_data.flatten(), bins=200, color='skyblue', alpha=0.7)
    plt.title(f"Distribución de intensidades en {split}")
    plt.xlabel("Valor EMG")
    plt.ylabel("Cantidad de muestras")
    plt.show()