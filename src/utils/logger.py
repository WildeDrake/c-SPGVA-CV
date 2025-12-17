import os
import csv
import matplotlib.pyplot as plt
import json

class TrainerLogger:
    def __init__(self, outdir, log_name="metrics"):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.csv_path = os.path.join(outdir, f"{log_name}.csv")
        self.plot_loss_path = os.path.join(outdir, "loss_metrics.png")
        self.plot_acc_path = os.path.join(outdir, "accuracy_metrics.png")
        self.metrics = []

        if not os.path.exists(self.csv_path):
            with open(self.csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch",
                    "train_loss", "train_class_y_loss", "train_acc_y", "train_acc_d",
                    "val_loss", "val_class_y_loss", "val_acc_y", "val_acc_d",
                    "cross_acc_y", "cross_acc_d"
                ])

    def log_epoch(self, epoch, train_loss, train_class_y_loss, train_acc_y, train_acc_d,
                  val_loss=None, val_class_y_loss=None, val_acc_y=None, val_acc_d=None,
                  cross_acc_y=None, cross_acc_d=None):
        
        # Guardamos todos los valores, usando None para los que no se proporcionaron
        row = [epoch, train_loss, train_class_y_loss, train_acc_y, train_acc_d,
               val_loss, val_class_y_loss, val_acc_y, val_acc_d,
               cross_acc_y, cross_acc_d]
        self.metrics.append(row)

        # Guardar en CSV
        with open(self.csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(row)

        # Actualizar gráfico
        self._plot_metrics()

    def _plot_metrics(self):
        if len(self.metrics) == 0:
            return

        # ------------------- Extracción de Métricas -------------------
        # Índices: [0: epoch, 1: train_loss, 2: train_class_y_loss, 5: val_loss, 6: val_class_y_loss, 
        #           7: val_acc_y, 8: val_acc_d, 9: cross_acc_y, 10: cross_acc_d]
        
        epochs = [m[0] for m in self.metrics]
        
        # Métrica de Loss (Usamos la Loss Total del VAE)
        train_total_loss = [m[1] for m in self.metrics]
        val_total_loss = [m[5] for m in self.metrics if m[5] is not None]
        
        # Métrica de Accuracy (Clasificación de Gesto - Y)
        train_acc_y = [m[3] for m in self.metrics]
        val_acc_y = [m[7] for m in self.metrics if m[7] is not None]
        cross_acc_y = [m[9] for m in self.metrics if m[9] is not None]

        # ------------------- Gráfico 1: Loss Total -------------------
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_total_loss, label="Train Total Loss", color='blue')
        
        if val_total_loss:
            # Asegurar que la longitud de val_total_loss coincida con las épocas
            plt.plot(epochs[:len(val_total_loss)], val_total_loss, label="Val Total Loss", color='orange')
            
        plt.title("VAE Total Loss Over Epochs")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_loss_path)
        plt.close()
        
        # ------------------- Gráfico 2: Accuracy (Gesto) -------------------
        plt.figure(figsize=(10, 5))
        plt.plot(epochs, train_acc_y, label="Train Gesture Acc", color='green')
        
        if val_acc_y:
            plt.plot(epochs[:len(val_acc_y)], val_acc_y, label="Val Gesture Acc", color='red')
            
        if cross_acc_y:
            plt.plot(epochs[:len(cross_acc_y)], cross_acc_y, label="Cross-Subject Acc", color='purple', linestyle='--')
            
        plt.title("Gesture Classification Accuracy (Y)")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_acc_path)
        plt.close()