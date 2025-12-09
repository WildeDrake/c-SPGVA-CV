import numpy as np
import matplotlib.pyplot as plt

def load_txt_channels(path):
    """Carga un txt y devuelve data con filas = canales."""
    data = np.loadtxt(path)
    return data.T  # filas = canales

def compare_channel(path1, path2, channel=0):
    """Compara visualmente un canal entre dos señales."""
    data1 = load_txt_channels(path1)
    data2 = load_txt_channels(path2)

    # Asegurar que el canal exista
    if channel >= data1.shape[0] or channel >= data2.shape[0]:
        raise ValueError(f"El canal {channel} no existe en uno de los archivos.")

    sig1 = data1[channel]
    sig2 = data2[channel]

    # Ajustar longitud si difieren
    min_len = min(len(sig1), len(sig2))
    sig1 = sig1[:min_len]
    sig2 = sig2[:min_len]

    plt.figure(figsize=(12, 4))
    plt.plot(sig1, label="Señal 1")
    plt.plot(sig2, label="Señal 2", alpha=0.7)
    plt.title(f"Comparación Canal {channel+1}")
    plt.legend()
    plt.tight_layout()
    plt.show()


def compare_8channels(path1, path2):
    data1 = load_txt_channels(path1)
    data2 = load_txt_channels(path2)

    min_len = min(data1.shape[1], data2.shape[1])

    plt.figure(figsize=(12, 14))
    for ch in range(8):
        plt.subplot(8, 1, ch+1)
        plt.plot(data1[ch][:min_len], label="Señal 1")
        plt.plot(data2[ch][:min_len], label="Señal 2", alpha=0.7)
        plt.title(f"Canal {ch+1}")
        plt.legend()

    plt.tight_layout()
    plt.show()


# EJEMPLO
PATHOLOGIES = ["all", "DMD", "Neuropathy", "Parkinson", "Stroke", "ALS", "Artifact"]

path_healthy = "../dataset/s1/all/LEFT/Fist-1.txt"
path_dmd     = "../dataset/s1/Parkinson/LEFT/Fist-1.txt"
path_Neuropathy = "../dataset/s1/Neuropathy/LEFT/Fist-1.txt"
path_Parkinson = "../dataset/s1/Parkinson/LEFT/Fist-1.txt"
path_Stroke = "../dataset/s1/Stroke/LEFT/Fist-1.txt"
path_ALS = "../dataset/s1/ALS/LEFT/Fist-1.txt"
path_Artifact = "../dataset/s1/Artifact/LEFT/Fist-1.txt"

compare_8channels(path_healthy, path_dmd)
compare_8channels(path_healthy, path_Neuropathy)
compare_8channels(path_healthy, path_Parkinson)
compare_8channels(path_healthy, path_Stroke)
compare_8channels(path_healthy, path_ALS)
compare_8channels(path_healthy, path_Artifact)
