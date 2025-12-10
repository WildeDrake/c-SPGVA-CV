import os
import numpy as np
import torch
import torch.utils.data as data_utils
from torch.utils.data import DataLoader, ConcatDataset

# Definición de la Patología
PATHOLOGY_MAP = {
    "Healthy": 0, "DMD": 1, "Neuropathy": 2, "Parkinson": 3, 
    "Stroke": 4, "ALS": 5, "Artifact": 6 
}
C_DIM = len(PATHOLOGY_MAP) 

# Carga datos sEMG preprocesados desde archivos .npy
class semgdata_load(data_utils.Dataset):
    def __init__(self, root, split="training", pathology_name="Healthy", subjects=None, transform=None, shuffle=False):
        self.root = os.path.expanduser(root)
        self.split = split
        self.subjects = subjects
        self.transform = transform

        pathology_id = PATHOLOGY_MAP.get(pathology_name, 0)
        data_path = os.path.join(self.root, split, f"{pathology_name}.npy")
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"No se encontró el archivo {data_path}")

        dataset = np.load(data_path, allow_pickle=True).item()
        all_data = dataset["list_total_user_data"]
        all_labels = dataset["list_total_user_labels"]

        if subjects is not None:
            all_data = [all_data[i] for i in subjects]
            all_labels = [all_labels[i] for i in subjects]
        if shuffle:
            for i in range(len(all_data)):
                perm = np.random.permutation(len(all_data[i]))
                all_data[i] = all_data[i][perm]
                all_labels[i] = all_labels[i][perm]
        
        self.data = np.concatenate(all_data)
        self.labels = np.concatenate(all_labels)

        self.domains = []
        for i, subj_labels in enumerate(all_labels):
            self.domains.extend([i] * len(subj_labels))
        self.domains = np.array(self.domains)

        self.data = torch.tensor(self.data, dtype=torch.float32).unsqueeze(1)
        self.labels = torch.tensor(self.labels, dtype=torch.long)
        self.domains = torch.tensor(self.domains, dtype=torch.long)

        num_classes = self.labels.max().item() + 1
        num_domains = len(all_data)
        self.labels = torch.eye(num_classes)[self.labels]
        self.domains = torch.eye(num_domains)[self.domains]
        
        # Crear el tensor de Condición C
        pathology_id_tensor = torch.tensor([pathology_id], dtype=torch.long) 
        self.C = torch.eye(C_DIM)[pathology_id_tensor].repeat(len(self.labels), 1) 

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        x = self.data[index]
        y = self.labels[index]
        d = self.domains[index]
        c = self.C[index] 
        if self.transform is not None:
            x = self.transform(x)
        return x, y, d, c 

# Funcion para cargar un split
def load_split(root, split, pathology_name, batch_size=128, shuffle=False):
    dataset = semgdata_load(root=root, split=split, pathology_name=pathology_name)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    x, y, d, c = next(iter(dataloader))
    assert isinstance(x, torch.Tensor) and isinstance(y, torch.Tensor) and isinstance(d, torch.Tensor) and isinstance(c, torch.Tensor)
    return dataloader

# Lista completa de patologías para el fine-tuning
ALL_PATHOLOGIES = list(PATHOLOGY_MAP.keys()) 

# Función para cargar múltiples splits
def load_multiple_splits(root, split, pathology_list=ALL_PATHOLOGIES, batch_size=128, shuffle=False):
    datasets = []
    
    print(f"\n[Multi-Loader] Cargando split '{split}' para {len(pathology_list)} patologías:")
    
    for pat_name in pathology_list:
        try:
            print(f"  → Añadiendo: {pat_name}")
            dataset = semgdata_load(root=root, split=split, pathology_name=pat_name, shuffle=shuffle)
            datasets.append(dataset)
        except FileNotFoundError:
            print(f"  → Advertencia: Archivo {pat_name}.npy no encontrado en {root}/{split}. Saltando.")
            continue
        
    if not datasets:
        raise ValueError(f"No se pudo cargar ningún dataset para el split '{split}'.")
        
    combined_dataset = ConcatDataset(datasets)
    print(f"  → Datasets combinados. Tamaño total: {len(combined_dataset)} ventanas.")

    dataloader = DataLoader(combined_dataset, batch_size=batch_size, shuffle=shuffle)
    
    return dataloader