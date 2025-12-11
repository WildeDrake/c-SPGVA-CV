import argparse
import os
import numpy as np
import torch
from model_diva import DIVA
from utils.semgdata_loader import load_split, PATHOLOGY_MAP, C_DIM, load_multiple_splits

# -------------------------- Configuración --------------------------
ROOT_preprocessed = "./preprocessed_dataset"
TOTAL_GESTURES = 5

# Lista completa de patologías a generar (7 clases)
PATHOLOGIES_TO_GENERATE = list(PATHOLOGY_MAP.keys())
# Usamos un solo sujeto sano y sus gestos como base para la generación
BASE_SUBJECT_NAME = "Healthy" 

# -------------------------- Funciones de Utilidad --------------------------

# Nota: Asumimos que model.encode() fue adaptado para devolver las medias latentes (mu)
# en tu model_diva.py, ya que esa función no fue proporcionada directamente.
# Si el modelo no tiene model.encode(), se puede usar la función forward y tomar los zs/zy.

def extract_latents(loader, model, device):
    """
    Extrae los latentes zs (Sujeto) y zy (Gesto) y las etiquetas de Gesto (Y) de las señales sanas.
    """
    model.eval()
    all_zs, all_zy, all_y = [], [], []
    with torch.no_grad():
        for x, y, d, c in loader: 
            x, y, d, c = x.to(device), y.to(device), d.to(device), c.to(device) 
            
            # Usaremos los encoders internos para obtener los latentes medios
            # El forward de DIVA devuelve 12 cosas; necesitamos reestructurar model_diva para
            # tener un método de encode más limpio. Temporalmente, usaremos forward
            # y asumimos que los outputs son los latentes muestreados (zd_q, zy_q)
            
            # Nota: Esto es un workaround. Lo ideal es usar mu_zd, mu_zy. 
            # Los latentes muestreados aquí son zd_q, zx_q, zy_q.
            *_, qzd, pzd, zd_q, qzx, pzx, zx_q, qzy, pzy, zy_q = model.forward(d, x, y, c)

            # Usamos los latentes muestreados (zd_q, zy_q) como representación de la base
            all_zs.append(zd_q.cpu().numpy())
            all_zy.append(zy_q.cpu().numpy())
            all_y.append(y.argmax(dim=1).cpu().numpy()) # Etiqueta de Gesto (Y)
            
    return np.concatenate(all_zs, axis=0), np.concatenate(all_zy, axis=0), np.concatenate(all_y, axis=0)

def generate_signals(model, device, base_zs, base_zy, base_y, target_pathology_id, num_samples):
    """
    Genera nuevas señales condicionales y sus etiquetas de Gesto y Patología.
    """
    model.eval()
    all_x_generated = []
    all_y_labels = [] # Lista para guardar la etiqueta de Gesto
    all_c_labels = [] # Lista para guardar la etiqueta de Patología

    # Crear el vector de Condición C (el mismo para todo el lote)
    C_tensor = torch.zeros(1, C_DIM).to(device)
    C_tensor[0, target_pathology_id] = 1.0
    
    # Para la generación, elegimos muestras aleatorias de los latentes de base
    indices = np.random.choice(len(base_zs), size=num_samples, replace=True)
    
    # Convertir a tensores y mover a device
    zs_sample = torch.tensor(base_zs[indices], dtype=torch.float32).to(device)
    zy_sample = torch.tensor(base_zy[indices], dtype=torch.float32).to(device)
    
    # Obtener las etiquetas de Gesto (Y) correspondientes a los latentes muestreados
    y_labels_batch = base_y[indices] 

    # Samplear ruido (zx) de una distribución Normal estándar
    zx_dim = model.zx_dim
    zx_sample = torch.randn(num_samples, zx_dim).to(device)

    # Replicar el tensor C para todas las muestras en el lote
    C_batch = C_tensor.repeat(num_samples, 1)

    with torch.no_grad():
        # Ejecutar el Decoder
        x_gen = model.px(zs_sample, zx_sample, zy_sample, C_batch)
        
        # Guardar la señal generada
        all_x_generated.append(x_gen.cpu().numpy())

        # Guardar las etiquetas de Gesto (Y) y Patología (C)
        all_y_labels.append(y_labels_batch)
        all_c_labels.append(np.full(num_samples, target_pathology_id, dtype=np.int64))

    X_gen = np.concatenate(all_x_generated, axis=0)
    Y_labels = np.concatenate(all_y_labels, axis=0)
    C_labels = np.concatenate(all_c_labels, axis=0)

    # 🚨 Devolvemos la señal generada y las dos etiquetas
    return X_gen, Y_labels, C_labels


# -------------------------- Main Execution --------------------------
def main_generation(args):
    device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")
    
    # 1. Cargar el modelo Fine-Tuned (el mejor o el último checkpoint)
    print(f"Cargando modelo Fine-Tuned desde: {args.model_path}")
    model = DIVA(args).to(device)
    # Se añade el 'ft-mode' para que DIVA se inicialice correctamente (aunque no sea necesario en la generación)
    model.ft_mode = 'finetune' 
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    # 2. Extraer Latentes de Base (Sanos)
    base_loader = load_split(ROOT_preprocessed, "training", BASE_SUBJECT_NAME, batch_size=args.batch_size, shuffle=False)
    base_zs, base_zy, base_y = extract_latents(base_loader, model, device)
    
    # 3. Preparar directorio de salida
    os.makedirs(args.generated_dir, exist_ok=True)
    
    print(f"\nIniciando Generación Condicional de {len(PATHOLOGIES_TO_GENERATE)} Patologías...")

    # Creamos un contenedor único para todas las señales y etiquetas
    X_all, Y_all, C_all = [], [], []

    # 4. Bucle de Generación por Patología
    for pat_name, pat_id in PATHOLOGY_MAP.items():
        print(f"  → Generando señales para: {pat_name} (ID: {pat_id})")
        
        num_samples = len(base_zs) * 10
        
        generated_signals, generated_y, generated_c = generate_signals(
            model, device, base_zs, base_zy, base_y, pat_id, num_samples
        )
        
        # 🚨 GUARDAR CADA GENERACIÓN COMO UN ÚNICO ARCHIVO .NPY CON DATOS Y ETIQUETAS
        output_path = os.path.join(args.generated_dir, f"generated_{pat_name}.npy")
        np.save(
            output_path, 
            {
                "X": generated_signals, # Señales generadas
                "Y": generated_y,       # Etiqueta de Gesto (5 Clases)
                "C": generated_c        # Etiqueta de Patología (7 Clases)
            }, 
            allow_pickle=True
        )
        print(f"  → Guardado {generated_signals.shape[0]} muestras en {output_path}")

    print("\n¡Generación Condicional Completada!")


# -------------------------- 5. Argparse --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Conditional sEMG Generation (DIVA)")
    parser.add_argument('--model-path', type=str, default='./saved_model/diva_best_seed0.model',
                        help="Ruta al modelo DIVA Fine-Tuned (.model)")
    parser.add_argument('--generated-dir', type=str, default='./generated_bank',
                        help="Ruta para guardar las señales generadas (.npy)")
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    
    # Parámetros del modelo (deben coincidir con el entrenamiento)
    parser.add_argument('--c-dim', type=int, default=C_DIM)
    parser.add_argument('--d-dim', type=int, default=10)
    parser.add_argument('--x-dim', type=int, default=416)
    parser.add_argument('--y-dim', type=int, default=TOTAL_GESTURES)
    parser.add_argument('--zd-dim', type=int, default=128)
    parser.add_argument('--zx-dim', type=int, default=128)
    parser.add_argument('--zy-dim', type=int, default=128)
    parser.add_argument('--aux_loss_multiplier_y', type=float, default=3500.)
    parser.add_argument('--aux_loss_multiplier_d', type=float, default=2000.)
    parser.add_argument('--beta_d', type=float, default=1.)
    parser.add_argument('--beta_x', type=float, default=1.)
    parser.add_argument('--beta_y', type=float, default=1.)
    
    args = parser.parse_args()
    main_generation(args)