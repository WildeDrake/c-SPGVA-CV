import argparse
import os
import random
import numpy as np
import torch
import torch.optim as optim
from model_diva import DIVA
from utils.semgdata_loader import load_split, load_multiple_splits
from utils.logger import TrainerLogger
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F # Necesario para MSE

ROOT_preprocessed = "./preprocessed_dataset"
TOTAL_SUBJECTS = 12
CROSS_SUBJECT = 2
TRAIN_SUBJECTS = 10
TOTAL_GESTURES = 5
MODO = "train" 
PATOLOGIAS_FT = ["Healthy", "DMD", "Neuropathy", "Parkinson", "Stroke", "ALS", "Artifact"]

# ------------------------------ Utils ------------------------------ #
def train_one_epoch(train_loader, model, optimizer, device):
    model.train()
    total_loss = 0.0
    total_class_y_loss = 0.0
    for x, y, d, c in train_loader: 
        x, y, d, c = x.to(device), y.to(device), d.to(device), c.to(device) 
        optimizer.zero_grad()
        loss, class_y_loss, *_ = model.loss_function(d, x, y, c) 
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_class_y_loss += class_y_loss.item() if hasattr(class_y_loss, 'item') else float(class_y_loss)
    n_batches = len(train_loader)
    return total_loss / n_batches, total_class_y_loss / n_batches

def compute_accuracy_from_logits(pred_logits, y_onehot):
    pred_labels = pred_logits.argmax(dim=1)
    true_labels = y_onehot.argmax(dim=1)
    correct = (pred_labels == true_labels).sum().item()
    return correct / float(pred_labels.size(0))

def evaluate(loader, model, device):
    model.eval()
    acc_y, acc_d = [], []
    with torch.no_grad():
        for x, y, d, c in loader: 
            x, y, d, c = x.to(device), y.to(device), d.to(device), c.to(device) 
            pred_d, pred_y = model.classifier(x)
            acc_y.append(compute_accuracy_from_logits(pred_y, y))
            acc_d.append(compute_accuracy_from_logits(pred_d, d))
            
    return float(np.mean(acc_d)), float(np.mean(acc_y))

def extract_features_for_tsne(loader, model, device):
    model.eval()
    all_zy, all_y = [], []
    with torch.no_grad():
        for x, y, d, c in loader:
            x, y = x.to(device), y.to(device)
            # El modelo solo necesita X, pero usamos el clasificador para t-SNE
            _, _, zy = model.classifier_for_tsne(x)
            
            all_zy.append(zy.cpu().numpy())
            all_y.append(y.argmax(dim=1).cpu().numpy())
            
    return np.concatenate(all_zy, axis=0), np.concatenate(all_y, axis=0)

def generate_reconstruction_sample(loader, model, device, num_samples=5):
    model.eval()
    x_original_samples = []
    x_reconstruido_samples = []
    
    with torch.no_grad():
        for x, y, d, c in loader:
            x, y, d, c = x.to(device), y.to(device), d.to(device), c.to(device)
            
            # Usar forward para obtener la reconstrucción
            x_recon, *_ = model.forward(d, x, y, c)
            
            # Procesar la reconstrucción (similar a la loss function)
            x_recon_final = x_recon.mean(dim=1, keepdim=True) # (N, 1, 8, 52)
            
            # Guardar el número deseado de muestras
            current_samples = x.size(0)
            samples_to_take = min(num_samples - len(x_original_samples), current_samples)
            
            if samples_to_take > 0:
                x_original_samples.append(x.cpu().numpy()[:samples_to_take])
                x_reconstruido_samples.append(x_recon_final.cpu().numpy()[:samples_to_take])
            
            if len(x_original_samples) >= num_samples:
                break
                
    if x_original_samples:
        return np.concatenate(x_original_samples, axis=0)[:num_samples], np.concatenate(x_reconstruido_samples, axis=0)[:num_samples]
    else:
        return np.array([]), np.array([])


# ------------------------------ Main ------------------------------ #
def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    device = torch.device("cuda" if (not args.no_cuda and torch.cuda.is_available()) else "cpu")

    # ---------------- DataLoaders ---------------- #
    root_dir = ROOT_preprocessed 
    
    if args.ft_mode == 'base':
        print("\n=== MODO BASE: Entrenamiento solo con Healthy (Paso 2) ===")
        train_loader = load_split(root_dir, "training", "Healthy", batch_size=args.batch_size, shuffle=True)
        val_loader = load_split(root_dir, "validation", "Healthy", batch_size=args.batch_size, shuffle=False)
        test_loader = load_split(root_dir, "testing", "Healthy", batch_size=args.batch_size, shuffle=False)
        cross_loader = load_split(root_dir, "cross_subject", "Healthy", batch_size=args.batch_size, shuffle=False)
        
    elif args.ft_mode == 'finetune':
        print("\n=== MODO FINE-TUNING: Ajuste con todas las patologías (Paso 3) ===")
        train_loader = load_multiple_splits(root_dir, "training", PATOLOGIAS_FT, batch_size=args.batch_size, shuffle=True)
        val_loader = load_multiple_splits(root_dir, "validation", PATOLOGIAS_FT, batch_size=args.batch_size, shuffle=False)
        test_loader = load_multiple_splits(root_dir, "testing", PATOLOGIAS_FT, batch_size=args.batch_size, shuffle=False)
        cross_loader = load_multiple_splits(root_dir, "cross_subject", PATOLOGIAS_FT, batch_size=args.batch_size, shuffle=False)

    else:
        raise ValueError("ft_mode debe ser 'base' o 'finetune'")
    
    # ---------------- Modelo y optimizador ---------------- #
    model = DIVA(args).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    if args.pretrained_model is not None:
        print(f"Loading pretrained model from {args.pretrained_model}")
        model.load_state_dict(torch.load(args.pretrained_model, map_location=device))

    # ---------------- Carpetas de checkpoints y logger ---------------- #
    checkpoint_dir = os.path.join(args.outpath, "checkpoints_models")
    os.makedirs(checkpoint_dir, exist_ok=True)
    best_model_path = os.path.join(args.outpath, f"diva_best_seed{args.seed}.model")
    logger = TrainerLogger(args.outpath) 

    # ---------------- Early stopping ---------------- #
    best_y_acc = 0.0
    best_loss = float('inf')
    early_counter = 0

    # ---------------- Entrenamiento ---------------- #
    for epoch in range(1, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs}")

        # Warm-up beta
        if args.warmup > 0 and epoch <= args.warmup:
            frac = epoch / args.warmup
            model.beta_d = args.min_beta + frac * (args.max_beta - args.min_beta)
            model.beta_x = args.min_beta + frac * (args.max_beta - args.min_beta)
            model.beta_y = args.min_beta + frac * (args.max_beta - args.min_beta)
        else:
            model.beta_d = args.max_beta
            model.beta_x = args.max_beta
            model.beta_y = args.max_beta

        # Selecciona loader según modo
        if args.mode == "train":
            current_loader = train_loader
        elif args.mode == "cross":
            current_loader = cross_loader
        else:
            raise ValueError("mode debe ser 'train' o 'cross'")

        train_loss, class_y_loss = train_one_epoch(current_loader, model, optimizer, device)
        acc_d_train, acc_y_train = evaluate(current_loader, model, device)
        print(f"Train - loss: {train_loss:.4f}, class_y_loss: {class_y_loss:.4f}, acc_y: {acc_y_train:.4f}")

        # Validación solo en modo 'train' para early stopping
        acc_d_val, acc_y_val = None, None
        if args.mode == "train":
            acc_d_val, acc_y_val = evaluate(val_loader, model, device)
            print(f"Validation - acc_y: {acc_y_val:.4f}, acc_d: {acc_d_val:.4f}")

            if acc_y_val > best_y_acc or (acc_y_val == best_y_acc and class_y_loss < best_loss):
                best_y_acc = acc_y_val
                best_loss = class_y_loss
                early_counter = 0
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with acc_y={best_y_acc:.4f}")
            else:
                early_counter += 1
                if early_counter >= args.early_stop_patience:
                    print(f"Early stopping triggered at epoch {epoch}")
                    break

        # Checkpoint por época
        if epoch % 8 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f"{args.mode}_epoch{epoch}_seed{args.seed}.pt")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved: {checkpoint_path}")

        # ---------------- Logging ---------------- #
        cross_acc_d, cross_acc_y = evaluate(cross_loader, model, device)
        
        # Corrección del logger: se usa la lista posicional y keyword arguments
        log_kwargs = {
            'epoch': epoch,
            'train_loss': train_loss, 
            'train_class_y_loss': class_y_loss, 
            'train_acc_y': acc_y_train, 
            'train_acc_d': acc_d_train,
            'cross_acc_y': cross_acc_y, 
            'cross_acc_d': cross_acc_d,
        }
        if acc_y_val is not None:
            log_kwargs['val_loss'] = train_loss 
            log_kwargs['val_class_y_loss'] = class_y_loss
            log_kwargs['val_acc_y'] = acc_y_val
            log_kwargs['val_acc_d'] = acc_d_val
            
        # Asumiendo que TrainerLogger.log_epoch acepta estos nombres
        logger.log_epoch(**log_kwargs) 
        
    # --- Generación de Datos de Diagnóstico (Paso 6a) ---
    print("\nGenerando características para t-SNE...")
    # Usar el loader de validación (o un loader combinado) para obtener una muestra diversa
    zy_features, y_labels = extract_features_for_tsne(val_loader, model, device)
    # Guardar features
    np.save(os.path.join(args.outpath, "tsne_zy_features.npy"), zy_features)
    np.save(os.path.join(args.outpath, "tsne_y_labels.npy"), y_labels)
    print("Características guardadas para t-SNE (zy_features.npy).")

    print("Generando muestras de reconstrucción...")
    x_original, x_reconstruido = generate_reconstruction_sample(val_loader, model, device, num_samples=5)
    # Guardar reconstrucción
    np.save(os.path.join(args.outpath, "x_original_sample.npy"), x_original)
    np.save(os.path.join(args.outpath, "x_reconstruido_sample.npy"), x_reconstruido)
    print("Muestras de reconstrucción guardadas (x_original_sample.npy y x_reconstruido_sample.npy).")
    
    # ---------------- Test final ---------------- #
    model.load_state_dict(torch.load(best_model_path if args.mode=="train" else checkpoint_path, map_location=device))
    test_acc_d, test_acc_y = evaluate(test_loader, model, device)
    print(f"\nTest accuracy - y: {test_acc_y:.4f}, d: {test_acc_d:.4f}")

    # ---------------- Cross-subject evaluación ---------------- #
    cross_acc_d, cross_acc_y = evaluate(cross_loader, model, device)
    print(f"\nCross-subject accuracy - y: {cross_acc_y:.4f}, d: {cross_acc_d:.4f}")


# ------------------------------ Argparse ------------------------------ #
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--ft-mode', type=str, default='base', choices=['base', 'finetune'],
                        help="Fase del entrenamiento: 'base' (solo Healthy) o 'finetune' (Healthy + Patologías).")
    parser.add_argument('--c-dim', type=int, default=7)
    parser.add_argument('--no-cuda', action='store_true', default=False)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--outpath', type=str, default='./saved_model')
    parser.add_argument('--pretrained-model', type=str, default=None,
                        help="Ruta a un modelo pre-entrenado para continuar entrenamiento / fine-tuning")
    parser.add_argument('--mode', type=str, default=MODO, choices=['train', 'cross'], help="Modo de entrenamiento")
    parser.add_argument('--d-dim', type=int, default=TRAIN_SUBJECTS)
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
    parser.add_argument('--warmup', type=int, default=100)
    parser.add_argument('--min_beta', type=float, default=0.0)
    parser.add_argument('--max_beta', type=float, default=3.0)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    args = parser.parse_args()
    os.makedirs(args.outpath, exist_ok=True)
    main(args)