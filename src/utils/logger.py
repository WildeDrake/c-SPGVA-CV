import os
import csv
import matplotlib.pyplot as plt
import json

class TrainerLogger:
    def __init__(self, outdir, log_name="metrics"):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.csv_path = os.path.join(outdir, f"{log_name}.csv")
        self.plot_path = os.path.join(outdir, f"{log_name}.png")
        self.metrics = []

        # Crear CSV vacío si no existe
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

        epochs = [m[0] for m in self.metrics]
        train_acc = [m[2] for m in self.metrics]
        val_acc = [m[6] for m in self.metrics if m[6] is not None]
        cross_acc = [m[9] for m in self.metrics if m[9] is not None]

        plt.figure(figsize=(8,5))
        plt.plot(epochs, train_acc, label="Train acc")
        if val_acc:
            plt.plot(epochs[:len(val_acc)], val_acc, label="Val acc")
        if cross_acc:
            plt.plot(epochs[:len(cross_acc)], cross_acc, label="Cross acc")
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(self.plot_path)
        plt.close()
