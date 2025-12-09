import numpy as np

# --------------------- extracción de características del espectrograma --------------------#
# calcular el espectrograma de un vector EMG
def cal_spectrogram_vector(vector, fs=200, npserseg=57, noverlap=0):
    frequencies_samples, time_segment_sample, spectrogram_of_vector = signal.spectrogram(x=vector, fs=fs,
                                                                                         nperseg=npserseg,
                                                                                         noverlap=noverlap,
                                                                                         window='hann',
                                                                                         scaling='spectrum')
    return spectrogram_of_vector, time_segment_sample, frequencies_samples

# calcular el espectrograma de un conjunto de datos EMG
def calculate_spectrogram(dataset):
    """
    :param dataset: (189, (8, 52))
    :return: dataset_spectrogram: (189, (4, 8, 14))
    """
    dataset_spectrogram = [] # inicializar la lista para almacenar los espectrogramas de todo el conjunto de datos
    # calcular el espectrograma para cada ejemplo en el conjunto de datos
    for examples in dataset:    # examples -> (8, 52)
        canals = [] # inicializar la lista para almacenar los espectrogramas de cada canal
        # calcular el espectrograma para cada canal
        for electrode_vector in examples:   # electrode -> (52, )
            spectrogram_of_vector, time_segment_sample, frequencies_samples = \
                cal_spectrogram_vector(electrode_vector, npserseg=28, noverlap=20)
            # spectrogram_of_vector <ndarray: (15, 4)>
            # remover frecuencias bajas inutiles del sEMG (0-5Hz)
            spectrogram_of_vector = spectrogram_of_vector[1:]   # spectrogram_of_vector <ndarray: (14, 4)>
            canals.append(np.swapaxes(spectrogram_of_vector, 0, 1))     # canals (8, (4, 14))
        # intercambiar las dimensiones de los canales y el tiempo
        example_to_classify = np.swapaxes(canals, 0, 1)     # example_to_classify <tuple: (4, 8, 14)>
        dataset_spectrogram.append(example_to_classify)
    # retornar el conjunto de datos de espectrogramas
    return dataset_spectrogram


# --------------------- extracción de características MAV --------------------#
# MAV: Músculo de Activación Media
def mav(emg_data):
    """
    :param emg_data:    <class 'np.ndarray'>    (8, 40)
    :return:            mav feature vector of the input emg matrix     (8, )
    """
    mav_result = np.mean(abs(emg_data), axis=1)
    return mav_result