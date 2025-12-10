import numpy as np
import pandas as pd
from scipy import signal
import logging

# --------------------- Procesamiento de datos de archivos txt a matrices numpy ---------------------#
def txt2array(txt_path):
    table_file = pd.read_table(txt_path, header=None)
    txt_file = table_file.iloc[:, :-1]
    txt_array = txt_file.values
    return txt_array

# --------------------- Preprocesamiento de datos EMG --------------------#
def preprocessing(data):
    """
    Preprocesa una matriz EMG:
    - Normaliza la amplitud al rango [-1, 1] (necesario para estabilizar el VAE).
    - Rectifica (abs), transpone a (channels, samples).
    - Filtra con lowpass Butterworth.
    """
    data = np.asarray(data, dtype=float)

    # 🚨 SOLUCIÓN RÁPIDA: Normalización a [-1, 1] (Min/Max Absoluto)
    # Requerido porque el VAE fue diseñado para inputs pequeños.
    max_val = np.max(np.abs(data))
    if max_val > 1e-6:
        data_norm = data / max_val
    else:
        data_norm = data
        
    # Rectificar
    arr = np.abs(data_norm)

    # Transponer (channels, samples)
    if arr.shape[0] < arr.shape[-1] and arr.shape[0] <= 8:
        arr_proc = arr
    else:
        arr_proc = arr.T

    # Filtrado (lowpass)
    wn = 0.05
    order = 4
    try:
        b, a = signal.butter(order, wn, btype='low')
        arr_filtered = signal.filtfilt(b, a, arr_proc, axis=1)
    except Exception:
        arr_filtered = arr_proc

    return arr_filtered.astype(np.float32)

# --------------------- Detección de la región de actividad muscular --------------------#
# (Se mantiene la versión de detección de actividad muscular con RMS de tu código más reciente)
def detect_muscle_activity(emg_data,
                            fs=200,
                            rms_win_ms=50,
                            rms_step_ms=10,
                            energy_threshold_factor=3.0,
                            percentile_threshold=75,
                            min_activation_length=40,
                            max_gap_ms=100,
                            padding_ms=100):
    emg = np.asarray(emg_data, dtype=float)
    if emg.ndim != 2:
        raise ValueError("emg_data debe ser (channels, samples)")

    n_samples = emg.shape[1]
    if n_samples <= 0:
        return 0, 0

    win = int(rms_win_ms * fs / 1000)
    step = int(rms_step_ms * fs / 1000)
    win = max(win, 4)
    step = max(step, 1)
    
    squared = emg ** 2
    kernel = np.ones(win)
    rms_per_channel = np.zeros_like(squared)
    for ch in range(squared.shape[0]):
        rms_per_channel[ch] = np.sqrt(
            signal.convolve(squared[ch], kernel / win, mode='same')
        )
        
    energy = rms_per_channel.mean(axis=0)

    global_mean = np.mean(energy)
    global_median = np.median(energy)
    perc = np.percentile(energy, percentile_threshold)
    thr1 = global_median * energy_threshold_factor
    thr2 = perc
    threshold = max(thr1, thr2, global_mean * 0.5 * energy_threshold_factor)
    if global_median < 1e-6:
        threshold = max(threshold * 0.1, 1e-6)

    active_mask = energy > threshold

    max_gap = int(max_gap_ms * fs / 1000)
    if max_gap > 0:
        fill_kernel = np.ones(max_gap)
        fill_conv = signal.convolve(active_mask.astype(int), fill_kernel, mode='same')
        active_mask = fill_conv > 0

    dif = np.diff(np.concatenate(([0], active_mask.astype(int), [0])))
    starts = np.where(dif == 1)[0]
    ends = np.where(dif == -1)[0] 
    
    if len(starts) == 0:
        start_idx = 0
        end_idx = n_samples - 1
        return int(start_idx), int(end_idx)

    merged = []
    cur_s, cur_e = starts[0], ends[0]
    for s, e in zip(starts[1:], ends[1:]):
        if s <= cur_e + max_gap:
            cur_e = e
        else:
            merged.append((cur_s, cur_e))
            cur_s, cur_e = s, e
    merged.append((cur_s, cur_e))

    best_seg = max(merged, key=lambda se: se[1] - se[0])
    start_idx, end_idx = best_seg

    pad = int(padding_ms * fs / 1000)
    start_idx = max(0, start_idx - pad)
    end_idx = min(n_samples - 1, end_idx + pad)

    min_activation_length = 40 # Usar la constante definida al inicio
    if (end_idx - start_idx) < min_activation_length:
        start_idx = 0
        end_idx = n_samples - 1

    return int(start_idx), int(end_idx)

def label_indicator(path, emg_data=None):
    label = None
    if 'Fist' in path:
        label = 0
    elif 'Open' in path:
        label = 1
    elif 'Tap' in path:
        label = 2
    elif 'WaveIn' in path:
        label = 3
    elif 'WaveOut' in path:
        label = 4
    return label