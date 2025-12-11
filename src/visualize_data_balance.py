import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from collections import Counter
import argparse
from utils.semgdata_loader import semgdata_load, PATHOLOGY_MAP

# ---------------------------------------------------------
# 1. IMPORTAR TUS UTILIDADES
# ---------------------------------------------------------

# Configuración Visual
GESTURE_NAMES = {0: "Fist", 1: "Open", 2: "Tap", 3: "WaveIn", 4: "WaveOut", 5: "Relax"}
GESTURE_COLORS = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

# ---------------------------------------------------------
# 2. FUNCIONES DE ANÁLISIS
# ---------------------------------------------------------
def analyze_loader(loader):
    """Cuenta cuántas muestras hay de cada gesto en el dataloader."""
    y_counts = Counter()
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            # Tu dataset devuelve: x, y, d, c
            if len(batch) == 4:
                _, y, _, _ = batch
            else:
                continue

            # Manejo de One-Hot vs Índice
            if y.ndim > 1 and y.shape[1] > 1: 
                y_indices = torch.argmax(y, dim=1).cpu().numpy()
            else:
                y_indices = y.cpu().numpy()
                
            y_counts.update(y_indices)
            total += y.size(0)
            
    return y_counts, total

def plot_pathology_balance(split_name, stats, output_dir):
    """Crea un gráfico grid mostrando el balance de gestos por patología."""
    pathologies = sorted(stats.keys())
    if not pathologies: return

    n_pats = len(pathologies)
    cols = 2
    rows = (n_pats + 1) // 2  # Cálculo dinámico de filas
    
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    fig.suptitle(f"Balance de Datos - Split: {split_name.upper()}", fontsize=16, y=0.99)
    
    axes = axes.flatten() if n_pats > 1 else [axes]

    for i, pat in enumerate(pathologies):
        ax = axes[i]
        counts = stats[pat]['counts']
        total = stats[pat]['total']
        
        # Preparar datos X e Y
        g_ids = sorted(counts.keys())
        g_names = [GESTURE_NAMES.get(g, f"G{g}") for g in g_ids]
        g_values = [counts[g] for g in g_ids]
        
        # Colores consistentes
        colors = [GESTURE_COLORS[g % len(GESTURE_COLORS)] for g in g_ids]
        
        # Barras
        bars = ax.bar(g_names, g_values, color=colors, alpha=0.8)
        
        # Etiquetas de valor
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height, f'{int(height)}', 
                    ha='center', va='bottom', fontsize=9)

        ax.set_title(f"{pat} (N={total})", fontweight='bold')
        ax.grid(axis='y', linestyle='--', alpha=0.3)
        # Margen superior para que no se corte el texto
        if g_values:
            ax.set_ylim(0, max(g_values) * 1.15)

    # Apagar ejes sobrantes
    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    filename = os.path.join(output_dir, f"balance_{split_name}.png")
    plt.savefig(filename)
    plt.close()
    print(f"✅ Gráfico guardado: {filename}")

# ---------------------------------------------------------
# 3. MAIN LOOP
# ---------------------------------------------------------
def main(args):
    if not os.path.exists(args.root):
        print(f"❌ Error: No encuentro el directorio de datos: {args.root}")
        return

    splits = ["training", "validation", "testing", "cross_subject"]
    os.makedirs(args.out, exist_ok=True)
    
    print(f"--- Iniciando Auditoría Visual (Usando utils.py) ---")

    for split in splits:
        split_dir = os.path.join(args.root, split)
        if not os.path.exists(split_dir):
            continue

        print(f"\n📂 Procesando Split: {split}...")
        split_stats = {}
        
        # Iterar sobre las patologías definidas en TU utils.py
        for pat_name in PATHOLOGY_MAP.keys():
            try:
                # -------------------------------------------------
                # USAMOS TU CLASE IMPORTADA
                # -------------------------------------------------
                dataset = semgdata_load(root=args.root, split=split, pathology_name=pat_name)
                
                # Loader rápido para contar
                loader = DataLoader(dataset, batch_size=2048, num_workers=0, shuffle=False)
                
                counts, total = analyze_loader(loader)
                
                if total > 0:
                    split_stats[pat_name] = {'counts': counts, 'total': total}
                    print(f"   -> {pat_name:<12}: {total} muestras")
                
            except (FileNotFoundError, RuntimeError):
                # Es normal que no existan todas las patologías en todos los splits
                continue
            except Exception as e:
                print(f"   ⚠️ Error inesperado en {pat_name}: {e}")

        if split_stats:
            plot_pathology_balance(split, split_stats, args.out)
        else:
            print(f"   (Sin datos legibles en {split})")

    print(f"\n✨ Listo. Imágenes guardadas en: {args.out}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=str, default='./preprocessed_dataset', help="Ruta a la carpeta con training/val/etc.")
    parser.add_argument('--out', type=str, default='./dataset_plots', help="Carpeta de salida")
    args = parser.parse_args()
    
    main(args)