import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from classifier_eval import SimpleMLP, PATHOLOGY_MAP, train_mlp, evaluate_mlp

# Reutilizamos el modelo y el mapa de patologías

# Directorio donde se encuentran los archivos .npy originales de Fine-Tuning
ROOT_SYNTHETIC_ORIGINAL = "./preprocessed_dataset/training"
# Directorio donde se encuentran los archivos .npy GENERADOS
ROOT_GENERATED_BANK = "./generated_bank"

# -------------------------- Función de Carga de Datos de Fidelidad --------------------------
def load_fidelity_data(generated_dir, original_dir, batch_size=256, test_split=0.2):
    X_original, X_generated = [], []
    
    # 1. Cargar datos ORIGINALES (Etiqueta 0)
    for pat_name in PATHOLOGY_MAP.keys():
        original_path = os.path.join(original_dir, f"{pat_name}.npy")
        if os.path.exists(original_path):
            # Cargamos el diccionario y extraemos 'list_total_user_data'
            data_dict = np.load(original_path, allow_pickle=True).item()
            # El data_process.py guarda esto como lista de arrays por sujeto, concatenamos
            X_original_raw = np.concatenate(data_dict["list_total_user_data"], axis=0) 
            
            # Submuestreo para que las clases 0 y 1 estén balanceadas
            X_original_raw = X_original_raw[np.random.choice(len(X_original_raw), size=min(50000, len(X_original_raw)), replace=False)]
            
            X_original.append(X_original_raw)
        else:
            print(f"Advertencia: Archivo ORIGINAL {pat_name}.npy no encontrado. Saltando.")


    # 2. Cargar datos GENERADOS (Etiqueta 1)
    for pat_name in PATHOLOGY_MAP.keys():
        generated_path = os.path.join(generated_dir, f"generated_{pat_name}.npy")
        if os.path.exists(generated_path):
            data_dict = np.load(generated_path, allow_pickle=True).item()
            X_generated_raw = data_dict["X"]
            
            # 🚨 Adaptación de forma (similar al clasificador de calidad)
            if X_generated_raw.ndim == 4 and X_generated_raw.shape[1] > 1:
                X_generated_raw = np.mean(X_generated_raw, axis=1, keepdims=True)
            
            # Submuestreo para balancear
            X_generated_raw = X_generated_raw[np.random.choice(len(X_generated_raw), size=min(50000, len(X_generated_raw)), replace=False)]
            
            X_generated.append(X_generated_raw)
        else:
            print(f"Advertencia: Archivo GENERADO {pat_name}.npy no encontrado. Saltando.")


    if not X_original or not X_generated:
        raise FileNotFoundError("No se pudieron cargar suficientes datos originales y generados para la prueba de fidelidad.")

    # 3. Concatenar y asignar etiquetas binarias
    X_concat = np.concatenate(X_original + X_generated, axis=0)
    Y_concat = np.concatenate([np.zeros(sum(len(x) for x in X_original)), 
                               np.ones(sum(len(x) for x in X_generated))], axis=0).astype(np.int64)

    # 4. Shuffle y Split
    p = np.random.permutation(X_concat.shape[0])
    X_concat, Y_concat = X_concat[p], Y_concat[p]
    
    split_idx = int(X_concat.shape[0] * (1 - test_split))
    
    X_train, X_test = X_concat[:split_idx], X_concat[split_idx:]
    Y_train, Y_test = Y_concat[:split_idx], Y_concat[split_idx:]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # La dimensión de entrada (416) sigue siendo la misma
    input_dim = X_train_tensor.shape[1] * X_train_tensor.shape[2] * X_train_tensor.shape[3]
    
    print(f"Total de muestras para Fidelidad: {X_concat.shape[0]}. (Clases: 2)")
    return train_loader, test_loader, input_dim


# -------------------------- Main Execution --------------------------
def main_fidelity(args):
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)

    # Paso 1: Cargar datos
    train_loader, test_loader, input_dim = load_fidelity_data(
        ROOT_GENERATED_BANK, ROOT_SYNTHETIC_ORIGINAL, args.batch_size, args.test_split
    )

    # Paso 2: Inicializar modelo BINARIO (2 clases: Original vs. Generado)
    model = SimpleMLP(input_dim, 2).to(device) # Usamos 2 clases
    
    # Paso 3: Configuración de entrenamiento
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    print("\n--- Iniciando Entrenamiento del Detector de Fidelidad (2 Clases) ---")
    
    best_accuracy = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # Usamos las funciones de MLP ya definidas
        loss, train_acc = train_mlp(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate_mlp(model, test_loader, device)

        print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            
    print(f"\nRESULTADO FINAL DE FIDELIDAD: Precisión del Detector (50% es perfecto): {best_accuracy:.4f}")
    if best_accuracy <= 0.60:
         print("✅ ¡Alta Fidelidad! La data generada es muy similar a la original.")
    else:
         print("❌ Baja Fidelidad. El clasificador distingue fácilmente lo generado.")

# -------------------------- Argparse (adaptado) --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP Detector for Fidelity Evaluation")
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-split', type=float, default=0.2)
    parser.add_argument('--use-cuda', action='store_true', default=True)
    
    args = parser.parse_args()
    main_fidelity(args)