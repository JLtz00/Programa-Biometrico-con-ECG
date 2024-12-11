import numpy as np
import os
import pandas as pd
import json
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks, welch
from scipy.stats import entropy, skew, kurtosis
from scipy.spatial.distance import mahalanobis
from numpy.linalg import inv
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives.padding import PKCS7
from base64 import b64encode, b64decode

# Filtros
# Función que implementa un filtro pasa-banda de Butterworth
def butter_bandpass(lowcut, highcut, fs, order=4):
    nyquist = 0.5 * fs  # Frecuencia de Nyquist
    low = lowcut / nyquist  # Normalizar la frecuencia de corte baja
    high = highcut / nyquist  # Normalizar la frecuencia de corte alta
    b, a = butter(order, [low, high], btype='band')  # Crear el filtro pasa-banda
    return b, a

# Función que aplica el filtro pasa-banda a los datos
def apply_bandpass_filter(data, lowcut, highcut, fs):
    b, a = butter_bandpass(lowcut, highcut, fs)  # Obtener los coeficientes del filtro
    return filtfilt(b, a, data)  # Aplicar el filtro a los datos

# Lectura de archivos
# Función para cargar los datos ECG desde un archivo CSV
def load_ecg_csv_updated(filename, column_name='MLII'):
    try:
        df = pd.read_csv(filename)  # Leer el archivo CSV
        if column_name in df.columns:  # Comprobar si la columna con los datos ECG existe
            ecg_data = df[column_name].values  # Extraer los datos de la columna ECG
        else:
            raise ValueError(f"El archivo no tiene una columna '{column_name}'")
        return ecg_data  # Devolver los datos ECG
    except Exception as e:
        print(f"Error al leer el archivo CSV: {e}")
        return None  # Devolver None si hay un error

# Detección de ondas (R, P, T)
# Función para detectar los picos R en los datos ECG
def detect_r_peaks(data, fs):
    filtered_data = apply_bandpass_filter(data, 0.5, 50, fs)  # Filtrar la señal ECG
    r_peaks, _ = find_peaks(filtered_data, distance=int(fs * 0.6))  # Detectar picos R con una distancia mínima
    return r_peaks, filtered_data  # Devolver los picos R y la señal filtrada

# Función para detectar las ondas P y T a partir de los picos R
def detect_pt_waves(data, r_peaks, fs):
    p_peaks, t_peaks = [], []  # Listas para almacenar los picos P y T
    window_size = int(0.2 * fs)  # Tamaño de la ventana (200 ms antes/después del pico R)
    for r_peak in r_peaks:
        # Buscar onda P
        p_region = data[max(0, r_peak - window_size):r_peak]  # Región antes del pico R para onda P
        p_idx = np.argmax(p_region)  # Detectar el pico P dentro de la región
        p_peaks.append(max(0, r_peak - window_size) + p_idx)  # Agregar el pico P a la lista

        # Buscar onda T
        t_region = data[r_peak:min(len(data), r_peak + window_size)]  # Región después del pico R para onda T
        t_idx = np.argmax(t_region)  # Detectar el pico T dentro de la región
        t_peaks.append(r_peak + t_idx)  # Agregar el pico T a la lista

    return np.array(p_peaks), np.array(t_peaks)  # Devolver los picos P y T

# Extracción de características avanzadas
# Función para extraer características relevantes del ECG
def extract_ecg_features(data, fs):
    features = {}  # Diccionario para almacenar las características extraídas

    # Detección de picos (R, P, T)
    r_peaks, filtered_data = detect_r_peaks(data, fs)
    p_peaks, t_peaks = detect_pt_waves(filtered_data, r_peaks, fs)

    # Cálculo de intervalos R-R
    rr_intervals = np.diff(r_peaks) / fs  # Intervalos entre picos R
    features['mean_rr'] = np.mean(rr_intervals)  # Media de los intervalos R-R
    features['std_rr'] = np.std(rr_intervals)  # Desviación estándar de los intervalos R-R
    features['rmssd'] = np.sqrt(np.mean(np.square(np.diff(rr_intervals))))  # RMSSD (raíz cuadrada de la media de las diferencias cuadradas)
    features['pnn50'] = np.sum(np.abs(np.diff(rr_intervals)) > 0.05) / len(rr_intervals)  # PNN50 (porcentaje de intervalos R-R que difieren más de 50 ms)

    # Análisis en el dominio de frecuencia (espectro de potencia)
    f, Pxx = welch(filtered_data, fs=fs, nperseg=1024)  # Espectro de potencia usando la transformada de Fourier
    lf_band = (f >= 0.04) & (f <= 0.15)  # Banda de baja frecuencia (LF)
    hf_band = (f >= 0.15) & (f <= 0.4)   # Banda de alta frecuencia (HF)
    features['lf_power'] = np.sum(Pxx[lf_band])  # Potencia en la banda LF
    features['hf_power'] = np.sum(Pxx[hf_band])  # Potencia en la banda HF
    features['lf_hf_ratio'] = features['lf_power'] / features['hf_power'] if features['hf_power'] > 0 else 0  # Relación LF/HF

    # Características estadísticas y de entropía
    features['entropy'] = calculate_entropy(filtered_data)  # Entropía de la señal
    features['mean_signal'] = np.mean(filtered_data)  # Media de la señal
    features['std_signal'] = np.std(filtered_data)  # Desviación estándar de la señal
    features['skewness_signal'] = skew(filtered_data)  # Asimetría de la señal
    features['kurtosis_signal'] = kurtosis(filtered_data)  # Curtosis de la señal

    # Características de morfología de las ondas
    features['wave_amplitudes'] = {
        'p_mean': np.mean(filtered_data[p_peaks]),
        'r_mean': np.mean(filtered_data[r_peaks]),
        't_mean': np.mean(filtered_data[t_peaks])
    }
    features['wave_durations'] = {
        'pr_interval': np.mean((r_peaks - p_peaks) / fs),  # Duración del intervalo P-R
        'qt_interval': np.mean((t_peaks - r_peaks) / fs)  # Duración del intervalo Q-T
    }

    return features  # Devolver las características extraídas

# Cálculo de la entropía de la señal
def calculate_entropy(data):
    hist, _ = np.histogram(data, bins=10, density=True)  # Histograma de la señal
    return entropy(hist)  # Calcular la entropía a partir del histograma

# Gestión de plantillas biométricas
class ECGTemplateManager:
    def __init__(self):
        self.templates = {}  # Diccionario para almacenar plantillas
        self.aes_key = None  # Clave para cifrar/descifrar plantillas
        self.salt = None  # Salt para derivar la clave

    # Configurar clave AES
    def set_aes_key(self, password: str, salt: bytes):
        self.salt = salt  # Guardar el salt para usarlo al guardar plantillas
        kdf = PBKDF2HMAC(
            algorithm=hashes.SHA256(),
            length=32,
            salt=salt,
            iterations=100_000,
            backend=default_backend()
        )
        self.aes_key = kdf.derive(password.encode())  # Derivar la clave de la contraseña

    # Encriptar datos usando AES
    def encrypt_data(self, data: dict):
        if self.aes_key is None:
            raise ValueError("La clave AES no está configurada.")
        
        try:
            plaintext = json.dumps(data).encode()
            iv = os.urandom(16)  # Generar un IV aleatorio
            cipher = Cipher(algorithms.AES(self.aes_key), modes.CBC(iv), backend=default_backend())
            encryptor = cipher.encryptor()
            
            padder = PKCS7(128).padder()
            padded_data = padder.update(plaintext) + padder.finalize()
            ciphertext = encryptor.update(padded_data) + encryptor.finalize()
            
            return b64encode(iv + ciphertext).decode()
        except Exception as e:
            raise ValueError(f"Error al cifrar los datos: {e}")

    # Desencriptar datos usando AES
    def decrypt_data(self, encrypted_data: str):
        try:
            if self.aes_key is None:
                raise ValueError("La clave AES no está configurada.")
            
            # Decodificar los datos encriptados
            encrypted_bytes = b64decode(encrypted_data.encode())
            iv = encrypted_bytes[:16]
            ciphertext = encrypted_bytes[16:]
            
            # Inicializar descifrado AES
            cipher = Cipher(algorithms.AES(self.aes_key), modes.CBC(iv), backend=default_backend())
            decryptor = cipher.decryptor()

            # Desencriptar
            padded_data = decryptor.update(ciphertext) + decryptor.finalize()
            
            # Remover padding
            unpadder = PKCS7(128).unpadder()
            plaintext = unpadder.update(padded_data) + unpadder.finalize()
            
            # Convertir de JSON a diccionario
            return json.loads(plaintext.decode())
        except Exception as e:
            print(f"Error al descifrar los datos: {e}")
            raise

    # Guardar las plantillas en un archivo JSON
    def save_templates(self, filename):
        if not self.salt:
            raise ValueError("El salt no está configurado.")
        
        try:
            encrypted_data = self.encrypt_data(self.templates)
            with open(filename, 'w') as file:
                data_to_save = {
                    'salt': b64encode(self.salt).decode(),
                    'data': encrypted_data
                }
                json.dump(data_to_save, file)
            print("Plantillas guardadas exitosamente.")
        except Exception as e:
            print(f"Error al guardar plantillas: {e}")

    # Cargar plantillas desde un archivo JSON
    def load_templates(self, filename):
        try:
            with open(filename, 'r') as file:
                saved_data = json.load(file)
                self.salt = b64decode(saved_data['salt'])  # Decodificar el salt
                password = input("Ingrese la contraseña para cargar las plantillas: ")  # Solicitar la contraseña
                self.set_aes_key(password, self.salt)  # Configurar clave AES
                encrypted_data = saved_data['data']
                self.templates = self.decrypt_data(encrypted_data)  # Desencriptar plantillas
            print("Plantillas cargadas exitosamente.")
        except Exception as e:
            print(f"Error al cargar plantillas: {e}")

    # Agregar una nueva plantilla al conjunto de plantillas
    def add_template(self, name, features):
        self.templates[name] = features  # Almacenar la plantilla
        print(f"Plantilla '{name}' agregada exitosamente.")

    # Verificar si un archivo ECG coincide con alguna plantilla
    def verify_template(self, ecg_data, fs):
        if ecg_data is None:
            return False  # Si no hay datos ECG, no se puede verificar

        features = extract_ecg_features(ecg_data, fs)  # Extraer características del ECG

        # Comparar las características con las plantillas almacenadas
        for name, template in self.templates.items():
            if self.compare_features(features, template):
                print(f"Archivo coincide con la plantilla: {name}")  # Si coincide, mostrar el nombre de la plantilla
                return True

        print("No se encontró coincidencia con las plantillas almacenadas.")  # Si no coincide con ninguna plantilla
        return False

    # Función para comparar características de dos señales ECG
    def compare_features(self, features1, features2):
        thresholds = {
            'mean_rr': 0.05,
            'lf_hf_ratio': 0.2,
            'entropy': 0.1
        }
        for key, threshold in thresholds.items():
            if abs(features1[key] - features2[key]) > threshold:
                return False  # Si la diferencia supera el umbral, no coincide
        return True  # Si todas las características coinciden dentro de los umbrales

# Programa principal
def main():
    fs = 360  # Frecuencia de muestreo (Hz)
    manager = ECGTemplateManager()  # Crear un gestor de plantillas ECG

    # Configurar la clave AES
    password = input("Ingrese una contraseña para proteger las plantillas: ")
    salt = os.urandom(16)  # Generar un salt aleatorio
    manager.set_aes_key(password, salt)

    while True:
        print("\n==== MENÚ PRINCIPAL ====")
        print("1. Crear nueva plantilla")
        print("2. Verificar archivo con plantillas existentes")
        print("3. Cargar plantillas desde archivo")
        print("4. Guardar plantillas en archivo")
        print("5. Salir")
        print("========================")
        opcion = input("Seleccione una opción: ")

        if opcion == "1":
            print("\n---- CREAR NUEVA PLANTILLA ----")
            nombre_archivo = input("Ingrese el nombre del archivo CSV: ")
            columna_ecg = input("Ingrese el nombre de la columna con los datos ECG (e.g., 'MLII' o 'V1'): ")
            nombre_plantilla = input("Ingrese el nombre para la nueva plantilla: ")

            ecg_data = load_ecg_csv_updated(nombre_archivo, column_name=columna_ecg)  # Cargar los datos ECG
            if ecg_data is not None:
                # Extraer características
                caracteristicas = extract_ecg_features(ecg_data, fs)

                # Mostrar las características extraídas
                print("\n** Características extraídas de la señal ECG **")
                traduccion = {
                    "mean_rr": "Media de intervalos R-R",
                    "std_rr": "Desviación estándar de intervalos R-R",
                    "rmssd": "Raíz cuadrada de la media de las diferencias cuadradas sucesivas",
                    "pnn50": "Porcentaje de intervalos R-R sucesivos que difieren más de 50 ms",
                    "lf_power": "Potencia en baja frecuencia (LF)",
                    "hf_power": "Potencia en alta frecuencia (HF)",
                    "lf_hf_ratio": "Relación LF/HF",
                    "entropy": "Entropía",
                    "mean_signal": "Media de la señal filtrada",
                    "std_signal": "Desviación estándar de la señal",
                    "skewness_signal": "Asimetría de la señal",
                    "kurtosis_signal": "Curtosis de la señal",
                    "wave_amplitudes": "Amplitudes de las ondas",
                    "wave_durations": "Duraciones de las ondas"
                }

                # Mostrar las características traducidas
                for key, value in caracteristicas.items():
                    if key in traduccion:
                        nombre_caracteristica = traduccion[key]
                        if isinstance(value, dict):  # Mostrar subcaracterísticas
                            print(f"{nombre_caracteristica}:")
                            for sub_key, sub_value in value.items():
                                print(f"  {sub_key}: {sub_value}")
                        else:
                            print(f"{nombre_caracteristica}: {value}")
                    else:
                        print(f"{key}: {value}")

                # Confirmar antes de guardar la plantilla
                guardar = input("\n¿Desea guardar estas características como una nueva plantilla? (s/n): ").strip().lower()
                if guardar == 's':
                    manager.add_template(nombre_plantilla, caracteristicas)
                else:
                    print("Plantilla descartada.")
        elif opcion == "2":
            print("\n---- VERIFICAR ARCHIVO ----")
            nombre_archivo = input("Ingrese el nombre del archivo CSV: ")
            columna_ecg = input("Ingrese el nombre de la columna con los datos ECG (e.g., 'MLII' o 'V1'): ")

            ecg_data = load_ecg_csv_updated(nombre_archivo, column_name=columna_ecg)  # Cargar los datos ECG
            if ecg_data is not None:
                print("\n** Verificando coincidencias con plantillas existentes... **")
                manager.verify_template(ecg_data, fs)  # Verificar el archivo con las plantillas
        elif opcion == "3":
            print("\n---- CARGAR PLANTILLAS ----")
            nombre_archivo = input("Ingrese el nombre del archivo de plantillas (JSON): ")
            manager.load_templates(nombre_archivo)  # Cargar plantillas desde un archivo JSON
        elif opcion == "4":
            print("\n---- GUARDAR PLANTILLAS ----")
            nombre_archivo = input("Ingrese el nombre del archivo para guardar las plantillas (JSON): ")
            manager.save_templates(nombre_archivo)  # Guardar las plantillas en un archivo JSON
        elif opcion == "5":
            print("\nSaliendo del programa... ¡Hasta pronto!")  # Salir del programa
            break
        else:
            print("Opción no válida. Intente nuevamente.")  # Opción no válida

# Ejecutar el programa principal
if __name__ == "__main__":
    main()
