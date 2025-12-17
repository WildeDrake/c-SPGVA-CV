import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
import pandas as pd

# Archivos (Tus rutas)
FILE_FEAT = "./saved_model/finetune_cVAE/tsne_zy_features.npy"
FILE_LABELS = "./saved_model/finetune_cVAE/tsne_y_labels.npy"

# Mapa de Gestos
GESTURE_MAP = {0: "Fist", 1: "Open", 2: "Tap", 3: "WaveIn", 4: "WaveOut"}

# Configuración
SAMPLES_TO_PLOT = 5000  # <--- TRUCO: Usamos 5k puntos aleatorios para que sea rápido

def plot_tsne():
    try:
        print("Cargando datos latentes...")
        features = np.load(FILE_FEAT, allow_pickle=True)
        labels = np.load(FILE_LABELS, allow_pickle=True)

        # Si labels viene one-hot, pasar a índice
        if labels.ndim > 1:
            labels = np.argmax(labels, axis=1)
        
        # Aplanar features si vienen extrañas
        if features.ndim > 2:
            features = features.reshape(features.shape[0], -1)

        total_samples = features.shape[0]
        print(f"Datos totales: {features.shape}")

        # --- SUBMUESTREO ESTRATÉGICO ---
        if total_samples > SAMPLES_TO_PLOT:
            print(f"Dataset muy grande ({total_samples}). Submuestrando a {SAMPLES_TO_PLOT} para visualización rápida...")
            indices = np.random.choice(total_samples, SAMPLES_TO_PLOT, replace=False)
            features = features[indices]
            labels = labels[indices]
        # -------------------------------

        print(f"Calculando t-SNE sobre {features.shape[0]} muestras...")

        # Calcular t-SNE 
        tsne = TSNE(n_components=2, random_state=42, perplexity=30, init='pca', learning_rate='auto')
        projections = tsne.fit_transform(features)

        # Crear DataFrame
        df = pd.DataFrame({
            'x': projections[:, 0],
            'y': projections[:, 1],
            'Gesto': [GESTURE_MAP.get(l, str(l)) for l in labels]
        })

        # Graficar
        plt.figure(figsize=(10, 8))
        sns.scatterplot(
            data=df, x='x', y='y', hue='Gesto', 
            palette='viridis', s=60, alpha=0.8, edgecolor='k'
        )
        
        plt.title('Espacio Latente de Contenido ($Z_y$): Agrupación de Gestos', fontsize=15)
        plt.xlabel('Dimensión t-SNE 1')
        plt.ylabel('Dimensión t-SNE 2')
        plt.grid(True, alpha=0.3)
        plt.legend(title='Gestos', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        output_file = "grafico_tsne_zy.png"
        plt.savefig(output_file, dpi=300)
        print(f"Gráfico guardado exitosamente como '{output_file}'")
        plt.show()

    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    plot_tsne()