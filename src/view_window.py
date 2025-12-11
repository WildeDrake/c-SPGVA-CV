import os
import sys
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# 1. IMPORTACIÓN ROBUSTA
# ---------------------------------------------------------
try:
    # Intentamos importar desde utils
    # Esto asume que tienes una carpeta 'utils' con un archivo 'semgdata_loader.py'
    from utils.semgdata_loader import semgdata_load, PATHOLOGY_MAP
except ImportError:
    # Si falla, intentamos agregar el directorio padre al path (útil si corres desde src/)
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
    try:
        from utils.semgdata_loader import semgdata_load, PATHOLOGY_MAP
    except ImportError as e:
        print("❌ Error Crítico de Importación:")
        print(f"   No se pudo encontrar 'utils.semgdata_loader'.")
        print(f"   Detalles: {e}")
        print("   Asegúrate de ejecutar este script desde la raíz del proyecto o que la estructura sea correcta.")
        sys.exit(1)

# ---------------------------------------------------------
# 2. CONFIGURACIÓN VISUAL
# ---------------------------------------------------------
# Mapeo de Gestos (Ajusta si tus IDs son diferentes)
GESTURE_MAP = {
    0: "Fist", 1: "Open", 2: "Pinch", 
    3: "WaveIn", 4: "WaveOut", 5: "Relax"
}

# Invertir PATHOLOGY_MAP para obtener nombres desde IDs
ID_TO_PATHOLOGY = {v: k for k, v in PATHOLOGY_MAP.items()}

def plot_window(x, y, d, c, sample_idx=0):
    """
    Grafica los 8 canales de una muestra específica.
    """
    # --- 1. Procesamiento de Datos ---
    # x viene como Tensor (1, 8, 52) o (8, 52). Lo pasamos a numpy.
    signal = x.detach().cpu().numpy()
    
    # Quitar dimensión de canal unitario si existe (1, 8, 52) -> (8, 52)
    if signal.ndim == 3:
        signal = signal.squeeze(0)
    
    # Verificar forma (8 canales x 52 tiempo)
    if signal.shape[0] != 8:
        # Si está transpuesto (52, 8), lo giramos
        if signal.shape[1] == 8:
            signal = signal.T
        else:
            print(f"⚠️ Advertencia: Forma de señal extraña: {signal.shape}")

    # --- 2. Procesamiento de Etiquetas ---
    # Decodificar Gesto (Y)
    if y.ndim > 0 and len(y) > 1: # Es One-Hot
        y_idx = torch.argmax(y).item()
    else:
        y_idx = y.item()
    gesture_name = GESTURE_MAP.get(y_idx, f"Gesto {y_idx}")

    # Decodificar Patología (C)
    if c is not None:
        if c.ndim > 0 and len(c) > 1: # Es One-Hot
            c_idx = torch.argmax(c).item()
        else:
            c_idx = c.item()
        pathology_name = ID_TO_PATHOLOGY.get(c_idx, f"Patología {c_idx}")
    else:
        pathology_name = "N/A"

    # --- 3. Graficar (Estilo Profesional) ---
    fig, axes = plt.subplots(nrows=8, ncols=1, figsize=(10, 12), sharex=True)
    
    fig.suptitle(f'Visualización de Ventana (Sample #{sample_idx})\n'
                 f'Patología: {pathology_name} | Gesto: {gesture_name}', fontsize=16)

    # Colores para estética (azul clínico)
    line_color = '#1f77b4'

    for i in range(8):
        # Graficar canal i
        axes[i].plot(signal[i, :], color=line_color, linewidth=1.2)
        
        # Etiquetas y Grid
        axes[i].set_ylabel(f'CH {i+1}', rotation=0, labelpad=20, fontweight='bold', fontsize=10)
        axes[i].grid(True, alpha=0.3, linestyle='--')
        
        # Opcional: Línea cero para referencia
        # axes[i].axhline(0, color='gray', linewidth=0.5, alpha=0.5)

    # Eje X solo abajo
    axes[-1].set_xlabel('Tiempo (Ventana de 200ms aprox)', fontsize=12)
    
    # Ajustar márgenes para que no se corten las etiquetas
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    plt.show()

# ---------------------------------------------------------
# 3. MAIN
# ---------------------------------------------------------
def main(args):
    print(f"--- Visualizador de Loader ---")
    print(f"Ruta Datos: {args.root}")
    print(f"Patología:  {args.pathology}")
    
    try:
        # Instanciar el Dataset usando la clase importada
        dataset = semgdata_load(
            root=args.root, 
            split=args.split, 
            pathology_name=args.pathology,
            shuffle=True # Shuffle para ver cosas distintas cada vez
        )
        
        loader = DataLoader(dataset, batch_size=32, shuffle=True)
        
        # Obtener un batch
        batch = next(iter(loader))
        
        # Desempaquetar (x, y, d, c)
        if len(batch) == 4:
            x_batch, y_batch, d_batch, c_batch = batch
        else:
            print(f"Formato de batch inesperado: {len(batch)} elementos.")
            return

        # Seleccionar el índice solicitado (o el 0 por defecto)
        idx = args.index
        if idx >= len(x_batch):
            print(f"⚠️ El índice {idx} es mayor que el batch size ({len(x_batch)}). Usando 0.")
            idx = 0
            
        print(f"\nGenerando gráfica para la muestra #{idx} del batch...")
        plot_window(x_batch[idx], y_batch[idx], d_batch[idx], c_batch[idx], sample_idx=idx)

    except FileNotFoundError:
        print(f"❌ Error: No se encontró el archivo .npy para '{args.pathology}' en '{args.root}/{args.split}'.")
    except Exception as e:
        print(f"❌ Error inesperado: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualizar una ventana del DataLoader sEMG")
    
    parser.add_argument('--root', type=str, default='./preprocessed_dataset', 
                        help='Ruta raíz de los datos')
    parser.add_argument('--split', type=str, default='training', 
                        help='Split a cargar (training, validation, testing)')
    parser.add_argument('--pathology', type=str, default='DMD', 
                        choices=list(PATHOLOGY_MAP.keys()),
                        help='Nombre de la patología a visualizar')
    parser.add_argument('--index', type=int, default=2, 
                        help='Índice de la muestra dentro del batch a graficar (0-31)')

    args = parser.parse_args()
    main(args)