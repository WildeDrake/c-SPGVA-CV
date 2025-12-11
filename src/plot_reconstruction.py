import numpy as np
import matplotlib.pyplot as plt

# Nombres de tus archivos
FILE_ORIG = "./saved_model/train_base/x_original_sample.npy"
FILE_REC = "./saved_model/train_base/x_reconstruido_sample.npy"

def plot_comparison():
    try:
        # Cargar datos
        orig = np.load(FILE_ORIG, allow_pickle=True)
        rec = np.load(FILE_REC, allow_pickle=True)
        
        # Ajustar dimensiones si es necesario (asumiendo 1 muestra, 8 canales, N tiempo)
        # Si vienen en batch, tomamos la primera muestra [0]
        if orig.ndim > 2: orig = orig[0]
        if rec.ndim > 2: rec = rec[0]
        
        # Asegurar forma (8, 52)
        if orig.shape[0] != 8: orig = orig.T
        if rec.shape[0] != 8: rec = rec.T

        print(f"Graficando señales... Shape: {orig.shape}")

        # Configurar plot
        fig, axes = plt.subplots(8, 1, figsize=(10, 12), sharex=True)
        fig.suptitle('Calidad de Reconstrucción: Real (Azul) vs. SGVA (Naranja)', fontsize=16)

        for i in range(8):
            axes[i].plot(orig[i], label='Original (Input)', color='#1f77b4', linewidth=1.5, alpha=0.7)
            axes[i].plot(rec[i], label='Reconstruido (Output)', color='#ff7f0e', linewidth=1.5, alpha=0.8, linestyle='--')
            
            axes[i].set_ylabel(f'CH {i+1}', rotation=0, labelpad=20, fontweight='bold')
            axes[i].grid(True, alpha=0.3)
            
            # Solo poner leyenda en el primer canal para no ensuciar
            if i == 0:
                axes[i].legend(loc='upper right')

        axes[-1].set_xlabel('Tiempo (Muestras)')
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.savefig("grafico_reconstruccion.png", dpi=300)
        plt.show()
        print("✅ Gráfico guardado como 'grafico_reconstruccion.png'")

    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    plot_comparison()