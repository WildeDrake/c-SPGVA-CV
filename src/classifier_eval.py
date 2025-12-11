import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# Definición de las Clases (Debe coincidir con utils/semgdata_loader.py)
PATHOLOGY_MAP = {
    "Healthy": 0, "DMD": 1, "Neuropathy": 2, "Parkinson": 3, 
    "Stroke": 4, "ALS": 5, "Artifact": 6 
}
GESTURE_CLASSES = 5 # Asumimos 5 gestos (0-4)

# -------------------------- 1. Modelo de Clasificación (MLP) --------------------------
class SimpleMLP(nn.Module):
    """
    MLP que puede clasificar N clases (7 para patología, 5 para gesto).
    """
    def __init__(self, input_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.flatten = nn.Flatten()
        self.num_classes = num_classes # Se guarda el número de clases final
        
        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, num_classes) # Salida dinámica
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        x = self.flatten(x) 
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x # Logits


# -------------------------- 2. Data Loader para Datos Generados --------------------------
def load_generated_data(generated_dir, batch_size, test_split, target_mode):
    """
    Carga los archivos generados y extrae la etiqueta TARGET (C o Y)
    """
    data_list, labels_list = [], []

    print(f"\n--- Preparando data para target: {target_mode.upper()} ---")

    for name in PATHOLOGY_MAP.keys():
        file_path = os.path.join(generated_dir, f"generated_{name}.npy")
        
        if not os.path.exists(file_path):
            print(f"Advertencia: Archivo {name} no encontrado. Saltando.")
            continue
            
        data_dict = np.load(file_path, allow_pickle=True).item()
        
        X_raw = data_dict["X"]
        
        # 🚨 SELECCIÓN DE ETIQUETA TARGET
        if target_mode == 'pathology':
            # Target C (7 clases)
            Y_raw = data_dict["C"] 
        elif target_mode == 'gesture':
            # Target Y (5 clases)
            Y_raw = data_dict["Y"] 
        else:
            raise ValueError(f"Target mode '{target_mode}' no es válido.")


        # Adaptación de forma: Reducir los 48 canales a 1 (tomando el promedio)
        if X_raw.ndim == 4 and X_raw.shape[1] > 1:
            X_raw = np.mean(X_raw, axis=1, keepdims=True)
            
        data_list.append(X_raw)
        labels_list.append(Y_raw)

    if not data_list:
        raise FileNotFoundError("No se encontró ningún archivo generado para clasificar.")

    X = np.concatenate(data_list, axis=0)
    Y = np.concatenate(labels_list, axis=0)
    
    # Random shuffle y split
    np.random.seed(42)
    p = np.random.permutation(X.shape[0])
    X, Y = X[p], Y[p]
    
    split_idx = int(X.shape[0] * (1 - test_split))
    
    X_train, X_test = X[:split_idx], X[split_idx:]
    Y_train, Y_test = Y[:split_idx], Y[split_idx:]

    X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
    Y_train_tensor = torch.tensor(Y_train, dtype=torch.long)
    
    X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
    Y_test_tensor = torch.tensor(Y_test, dtype=torch.long)

    train_dataset = TensorDataset(X_train_tensor, Y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, Y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    input_dim = X_train_tensor.shape[1] * X_train_tensor.shape[2] * X_train_tensor.shape[3]
    
    print(f"Total muestras: {X.shape[0]}. Input Dim: {input_dim}")
    num_classes_final = Y.max() + 1
    
    return train_loader, test_loader, input_dim, num_classes_final


# -------------------------- 3. Entrenamiento y Evaluación --------------------------
# (Las funciones train_mlp y evaluate_mlp no necesitan cambios)
def train_mlp(model, train_loader, optimizer, criterion, device):
    # ... (código train_mlp sin cambios) ...
    model.train()
    total_loss = 0
    correct_predictions = 0
    
    for X, Y in train_loader:
        X, Y = X.to(device), Y.to(device)
        
        optimizer.zero_grad()
        outputs = model(X)
        loss = criterion(outputs, Y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        correct_predictions += (predicted == Y).sum().item()
        
    avg_loss = total_loss / len(train_loader)
    accuracy = correct_predictions / len(train_loader.dataset)
    return avg_loss, accuracy

def evaluate_mlp(model, test_loader, device):
    # ... (código evaluate_mlp sin cambios) ...
    model.eval()
    correct_predictions = 0
    
    with torch.no_grad():
        for X, Y in test_loader:
            X, Y = X.to(device), Y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs.data, 1)
            correct_predictions += (predicted == Y).sum().item()
            
    accuracy = correct_predictions / len(test_loader.dataset)
    return accuracy


# -------------------------- 4. Main Execution (DUAL) --------------------------
def run_evaluation_task(args, target_mode, num_classes_target):
    device = torch.device("cuda" if args.use_cuda and torch.cuda.is_available() else "cpu")
    
    # 1. Cargar datos generados, obteniendo la etiqueta TARGET
    train_loader, test_loader, input_dim, num_classes_actual = load_generated_data(
        args.generated_dir, args.batch_size, args.test_split, target_mode
    )

    if num_classes_actual != num_classes_target:
        print(f"Advertencia: Clases esperadas ({num_classes_target}) no coinciden con las cargadas ({num_classes_actual}). Ajustando...")
        num_classes_target = num_classes_actual # Ajustar si falta alguna clase en el generated bank

    # 2. Inicializar modelo con el número correcto de clases de salida
    model = SimpleMLP(input_dim, num_classes_target).to(device)
    
    # 3. Configuración de entrenamiento
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()
    
    print(f"\n--- Tarea: Clasificación de {target_mode.upper()} ({num_classes_target} Clases) ---")
    
    best_accuracy = 0.0
    
    for epoch in range(1, args.epochs + 1):
        loss, train_acc = train_mlp(model, train_loader, optimizer, criterion, device)
        test_acc = evaluate_mlp(model, test_loader, device)

        print(f"Epoch {epoch}/{args.epochs} - Loss: {loss:.4f}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}")
        
        if test_acc > best_accuracy:
            best_accuracy = test_acc
            model_name = f"mlp_{target_mode}_best.pth"
            torch.save(model.state_dict(), os.path.join(args.outpath, model_name))

    print(f"\nRESULTADO FINAL {target_mode.upper()}: Máxima Precisión: {best_accuracy:.4f}")
    
    # 🚨 INTERPRETACIÓN DEL RESULTADO
    if target_mode == 'pathology':
        print(f" -> Métrica de Calidad: Si Acc > 90%, el SGVA generó patologías distinguibles.")
    elif target_mode == 'gesture':
        print(f" -> Métrica de Pureza: Si Acc < 75% (similar a Cross-Subject), la patología no arruinó el gesto, indicando buen disentanglement.")
        
    return best_accuracy

def main_classifier(args):
    # Tarea 1: Evaluación de Calidad (Patología C)
    pathology_acc = run_evaluation_task(args, 'pathology', len(PATHOLOGY_MAP))
    
    # Tarea 2: Evaluación de Pureza/Disentanglement (Gesto Y)
    gesture_acc = run_evaluation_task(args, 'gesture', GESTURE_CLASSES)
    
    print("\n====================================")
    print(f"Resumen Final de Validación:")
    print(f"Patología (Calidad): {pathology_acc:.4f} (Debe ser ALTA)")
    print(f"Gesto (Pureza): {gesture_acc:.4f} (Debe ser ALTA, demostrando que el gesto se conservó)")
    print("====================================")


# -------------------------- 5. Argparse --------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MLP Classifier for Synthetic Pathology Quality")
    parser.add_argument('--generated-dir', type=str, default='./generated_bank',
                        help="Ruta a la carpeta donde se encuentran los archivos generated_X.npy")
    parser.add_argument('--outpath', type=str, default='./saved_model/classifier_results',
                        help="Ruta para guardar los logs y el modelo MLP")
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--test-split', type=float, default=0.2)
    parser.add_argument('--use-cuda', action='store_true', default=True)
    
    args = parser.parse_args()
    os.makedirs(args.outpath, exist_ok=True)
    main_classifier(args)