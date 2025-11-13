import os
import numpy as np
import data_process as dp

# --------------------- Configuración --------------------- #
BASE_DATASET = "../../split_dataset"
OUTPUT_DIR = "../preprocessed_dataset"
WINDOW_SIZE = 52   # longitud de ventana usada en DIVA
os.makedirs(OUTPUT_DIR, exist_ok=True)

SPLITS = ["training", "validation", "testing", "cross_subject"]
HANDS = ["LEFT", "RIGHT"]


# --------------------- Función para procesar un split --------------------- #
def process_split(split_name):
    list_total_user_data = []
    list_total_user_labels = []

    # Recorrer sujetos
    subjects = sorted([s for s in os.listdir(BASE_DATASET) if s.startswith("s")])
    for subj in subjects:
        subj_path = os.path.join(BASE_DATASET, subj)
        split_path = os.path.join(subj_path, split_name)

        subj_data = []
        subj_labels = []
        for hand in HANDS:
            hand_path = os.path.join(split_path, hand)
            if not os.path.exists(hand_path):
                continue
            for gesture_folder in os.listdir(hand_path):
                gesture_path = os.path.join(hand_path, gesture_folder)
                label = dp.label_indicator(gesture_folder)
                for file in os.listdir(gesture_path):
                    file_path = os.path.join(gesture_path, file)
                    data = dp.txt2array(file_path)
                    data = dp.preprocessing(data)
                    start, end = dp.detect_muscle_activity(data)
                    start, end = int(start), int(end)
                    activation = data[:, start:end]
                    
                    # Ventanas deslizantes
                    for i in range(0, activation.shape[1] - WINDOW_SIZE + 1):
                        window = activation[:, i:i + WINDOW_SIZE].astype(np.float32)
                        # calcular label considerando relax
                        label = dp.label_indicator(gesture_folder, emg_data=window)
                        if label is None:
                            continue  # descartamos si no corresponde
                        subj_data.append(window)
                        subj_labels.append(label)
        if subj_data:  # solo agregar si hay datos
            list_total_user_data.append(subj_data)
            list_total_user_labels.append(subj_labels)
    
    return list_total_user_data, list_total_user_labels

# --------------------- Guardar datasets --------------------- #
def save_all_splits():
    for split in SPLITS:
        data, labels = process_split(split)

        output_file = os.path.join(OUTPUT_DIR, f"{split}.npy")
        np.save(output_file, {
            "list_total_user_data": data,
            "list_total_user_labels": labels
        }, allow_pickle=True)
        total_windows = sum(len(u) for u in data)
        print(f"Guardado {split} con {len(data)} sujetos y {total_windows} ventanas en {output_file}")

# --------------------- Main --------------------- #
if __name__ == "__main__":
    save_all_splits()


    
    # Ejemplo de carga
    label_0 = 0
    label_1 = 0
    label_2 = 0
    label_3 = 0
    label_4 = 0
    label_5 = 0
    example = np.load(os.path.join(OUTPUT_DIR, "training.npy"), allow_pickle=True).item()
    label_counts = {}
    for user_labels in example["list_total_user_labels"]:
        for label in user_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
            if label == 0:
                label_0 += 1
            elif label == 1:
                label_1 += 1
            elif label == 2:
                label_2 += 1
            elif label == 3:
                label_3 += 1
            elif label == 4:
                label_4 += 1
            elif label == 5:
                label_5 += 1
    print("Cantidad de ejemplares por label:", label_counts, " para el split de training.")

    example = np.load(os.path.join(OUTPUT_DIR, "testing.npy"), allow_pickle=True).item()
    label_counts = {}
    for user_labels in example["list_total_user_labels"]:
        for label in user_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
            if label == 0:
                label_0 += 1
            elif label == 1:
                label_1 += 1
            elif label == 2:
                label_2 += 1
            elif label == 3:
                label_3 += 1
            elif label == 4:
                label_4 += 1
            elif label == 5:
                label_5 += 1
    print("Cantidad de ejemplares por label:", label_counts, " para el split de testing.")

    example = np.load(os.path.join(OUTPUT_DIR, "cross_subject.npy"), allow_pickle=True).item()
    label_counts = {}
    for user_labels in example["list_total_user_labels"]:
        for label in user_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
            if label == 0:
                label_0 += 1
            elif label == 1:
                label_1 += 1
            elif label == 2:
                label_2 += 1
            elif label == 3:
                label_3 += 1
            elif label == 4:
                label_4 += 1
            elif label == 5:
                label_5 += 1
    print("Cantidad de ejemplares por label:", label_counts, " para el split de cross_subject.")

    example = np.load(os.path.join(OUTPUT_DIR, "validation.npy"), allow_pickle=True).item()
    label_counts = {}
    for user_labels in example["list_total_user_labels"]:
        for label in user_labels:
            label_counts[label] = label_counts.get(label, 0) + 1
            if label == 0:
                label_0 += 1
            elif label == 1:
                label_1 += 1
            elif label == 2:
                label_2 += 1
            elif label == 3:
                label_3 += 1
            elif label == 4:
                label_4 += 1
            elif label == 5:
                label_5 += 1
    print("Cantidad de ejemplares por label:", label_counts, " para el split de validation.")

    print("Totales generales por label en todos los splits:")
    print("Label 0:", label_0)
    print("Label 1:", label_1)
    print("Label 2:", label_2)
    print("Label 3:", label_3)
    print("Label 4:", label_4)
    print("Label 5:", label_5)