#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>
#include <map>
#include <numeric>
#include <limits>
#include <algorithm>
#include <complex>
#include <fftw3.h>

// Constantes de frecuencia de corte para los filtros
const double LOW_CUTOFF = 0.5;   // Filtro pasa bajas para eliminar el ruido de alta frecuencia
const double HIGH_CUTOFF = 50.0; // Filtro pasa altas para eliminar la interferencia de línea base

// Parámetros para detección de picos
const double THRESHOLD_SCALE = 0.6;
const int INTEGRATION_WINDOW = 30;

// Estructura para almacenar características de ondas morfológicas
struct morphology_waveform {
    double onset;     // Inicio de la onda en milisegundos
    double offset;    // Fin de la onda en milisegundos
    double amplitude; // Amplitud de la onda
    double width;     // Duración de la onda en milisegundos
};

struct WaveAmplitudes {
    double pAmplitude;
    double qrsAmplitude;
    double tAmplitude;
};

// Estructura para almacenar las características del ECG
struct ECGFeatures {
    std::map<std::string, morphology_waveform> morphology_waveforms;
    std::vector<double> rr_intervals;
    std::tuple<double, double, double> amplitude_analysis;
    std::map<std::string, double> amplitude_ratios;
    std::pair<double, double> heart_rate_vrc;
    std::vector<double> energy_analysis;
    std::map<std::string, std::pair<double, double>> wave_symmetry;
    std::vector<double> frequency_domain;
    double entropy_index;
    std::map<std::string, std::pair<double, double>> segment_slopes;
};

// Cargar datos de ECG desde archivo CSV
std::vector<double> loadECGData(const std::string& filename) {
    std::vector<double> ecgData;
    std::ifstream file(filename);
    std::string line;
    if (!file.is_open()) {
        std::cerr << "Error abriendo el archivo: " << filename << std::endl;
        return ecgData;
    }

    // Saltar la cabecera
    std::getline(file, line);
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string value;
        int column = 0;
        double mlii_value;

        // Asume que la columna 2 tiene la señal MLII
        while (std::getline(ss, value, ',')) {
            if (column == 2) { // Cambia esto si la columna es diferente
                mlii_value = std::stod(value);
                ecgData.push_back(mlii_value);
                break;
            }
            column++;
        }
    }
    file.close();
    return ecgData;
}

// Realiza la FFT y calcula la magnitud del espectro de frecuencia
std::vector<double> calculateFrequencyDomain(const std::vector<double>& ecgData, double samplingRate) {
    size_t N = ecgData.size();
    std::vector<double> magnitudeSpectrum(N / 2); // Solo mitad del espectro (frecuencias positivas)

    // Arreglo para entrada y salida de FFTW
    fftw_complex *out = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * N);
    double *in = (double*) fftw_malloc(sizeof(double) * N);
    for (size_t i = 0; i < N; ++i) in[i] = ecgData[i];

    // Planificación y ejecución de la FFT
    fftw_plan plan = fftw_plan_dft_r2c_1d(N, in, out, FFTW_ESTIMATE);
    fftw_execute(plan);

    // Calcular la magnitud del espectro
    for (size_t i = 0; i < N / 2; ++i) {
        magnitudeSpectrum[i] = sqrt(out[i][0] * out[i][0] + out[i][1] * out[i][1]);
    }

    // Liberar memoria
    fftw_destroy_plan(plan);
    fftw_free(in);
    fftw_free(out);

    return magnitudeSpectrum;
}

void analyzeFrequencyDomain(const std::vector<double>& ecgData, double samplingRate) {
    auto spectrum = calculateFrequencyDomain(ecgData, samplingRate);
    size_t N = spectrum.size();
    double frequencyResolution = samplingRate / (2.0 * N); // Resolución de frecuencia

    std::cout << "\n--- Análisis en el Dominio de la Frecuencia ---\n";
    
    // Identificar la frecuencia dominante
    auto maxElement = std::max_element(spectrum.begin(), spectrum.end());
    double maxSpectrumValue = *maxElement;

    if (maxSpectrumValue > 0) { // Evitar casos donde el espectro sea completamente 0
        size_t index = std::distance(spectrum.begin(), maxElement);
        double dominantFrequency = index * frequencyResolution;
        std::cout << "Frecuencia dominante: " << dominantFrequency << " Hz\n";
    } else {
        std::cout << "No se encontró una frecuencia dominante significativa en el espectro.\n";
    }
}


// Función para derivar la señal
std::vector<double> derivate(const std::vector<double>& signal) {
    std::vector<double> derivative(signal.size());
    derivative[0] = 0;  // Inicializa el primer valor a cero
    for (size_t i = 1; i < signal.size(); ++i) {
        derivative[i] = signal[i] - signal[i - 1];
    }
    return derivative;
}

// Función para elevar al cuadrado cada valor de la señal
std::vector<double> squared(const std::vector<double>& signal) {
    std::vector<double> squaredSignal(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        squaredSignal[i] = signal[i] * signal[i];
    }
    return squaredSignal;
}

// Función para la integración en ventana
std::vector<double> integrated(const std::vector<double>& signal, int windowSize) {
    std::vector<double> integratedSignal(signal.size(), 0.0);
    
    for (size_t i = 0; i < signal.size(); ++i) {
        double sum = 0.0;
        for (int j = 0; j < windowSize; ++j) {
            if ((int)i - j >= 0) {
                sum += signal[i - j];
            }
        }
        integratedSignal[i] = sum / windowSize;
    }
    return integratedSignal;
}

// Normalización de datos por segmento
std::vector<double> normalizeSegment(const std::vector<double>& segment) {
    double maxVal = *std::max_element(segment.begin(), segment.end());
    double minVal = *std::min_element(segment.begin(), segment.end());
    std::vector<double> normalizedSegment(segment.size());

    for (size_t i = 0; i < segment.size(); ++i) {
        normalizedSegment[i] = (segment[i] - minVal) / (maxVal - minVal);
    }
    return normalizedSegment;
}

// Filtro pasa altas (Butterworth de primer orden)
std::vector<double> highPassFilter(const std::vector<double>& signal, double samplingRate, double cutoffFrequency) {
    double RC = 1.0 / (cutoffFrequency * 2 * M_PI);
    double dt = 1.0 / samplingRate;
    double alpha = RC / (RC + dt);
    std::vector<double> filteredSignal(signal.size());
    filteredSignal[0] = signal[0];
    
    for (size_t i = 1; i < signal.size(); ++i) {
        filteredSignal[i] = alpha * (filteredSignal[i - 1] + signal[i] - signal[i - 1]);
    }
    return filteredSignal;
}

// Filtro pasa bajas (Butterworth de primer orden)
std::vector<double> lowPassFilter(const std::vector<double>& signal, double samplingRate, double cutoffFrequency) {
    double RC = 1.0 / (cutoffFrequency * 2 * M_PI);
    double dt = 1.0 / samplingRate;
    double alpha = dt / (RC + dt);
    std::vector<double> filteredSignal(signal.size());
    filteredSignal[0] = signal[0];
    
    for (size_t i = 1; i < signal.size(); ++i) {
        filteredSignal[i] = filteredSignal[i - 1] + alpha * (signal[i] - filteredSignal[i - 1]);
    }
    return filteredSignal;
}

// Filtro pasa banda (combina pasa bajas y pasa altas)
std::vector<double> bandPassFilter(const std::vector<double>& signal, double samplingRate, double lowCutoff, double highCutoff) {
    std::vector<double> lowPassedSignal = lowPassFilter(signal, samplingRate, highCutoff);
    std::vector<double> bandPassedSignal = highPassFilter(lowPassedSignal, samplingRate, lowCutoff);
    return bandPassedSignal;
}

// Función para la detección de picos R con un umbral adaptativo
std::vector<int> detectRPeaks(const std::vector<double>& ecgData, double thresholdScale) {
    std::vector<double> signal = derivate(ecgData);
    signal = squared(signal);
    signal = integrated(signal, INTEGRATION_WINDOW);

    double maxSignalValue = *std::max_element(signal.begin(), signal.end());
    double threshold = thresholdScale * maxSignalValue;
    std::vector<int> rPeaks;

    for (size_t i = 1; i < signal.size() - 1; ++i) {
        // Detecta un pico si el valor actual es mayor que el umbral y es un máximo local
        if (signal[i] > threshold && signal[i] > signal[i - 1] && signal[i] > signal[i + 1]) {
            rPeaks.push_back(i);
        }
    }
    return rPeaks;
}

// Calcular Intervalos R-R
std::vector<double> calculateRRIntervals(const std::vector<double>& ecgData, double samplingRate) {
    std::vector<double> filteredECG = bandPassFilter(ecgData, samplingRate, LOW_CUTOFF, HIGH_CUTOFF);
    std::vector<int> rPeaks = detectRPeaks(filteredECG, THRESHOLD_SCALE);

    std::vector<double> rIntervals;
    for (size_t i = 1; i < rPeaks.size(); ++i) {
        double interval = (rPeaks[i] - rPeaks[i - 1]) / samplingRate * 1000;
        rIntervals.push_back(interval);
    }
    return rIntervals;
}

std::map<std::string, morphology_waveform> analyzeWaveforms(const std::vector<double>& ecgData, double samplingRate) {
    // Filtrar la señal para reducir ruido
    std::vector<double> filteredECG = bandPassFilter(ecgData, samplingRate, LOW_CUTOFF, HIGH_CUTOFF);

    // Detectar picos R con umbral adaptativo
    std::vector<int> rPeaks = detectRPeaks(filteredECG, THRESHOLD_SCALE);

    // Estructura para almacenar los resultados
    std::map<std::string, morphology_waveform> waveforms;

    for (int rPeak : rPeaks) {
        // Parámetros básicos
        int signalSize = filteredECG.size();
        
        // --- Onda P ---
        int pOnset = std::max(0, rPeak - static_cast<int>(samplingRate * 0.2)); // Buscar antes del pico R
        int pOffset = rPeak - static_cast<int>(samplingRate * 0.05);            // Fin antes del R
        auto pMax = std::max_element(filteredECG.begin() + pOnset, filteredECG.begin() + pOffset);
        int pPeak = std::distance(filteredECG.begin(), pMax);
        
        // Detección del inicio y fin basado en derivadas
        int pWaveStart = pPeak, pWaveEnd = pPeak;
        for (int i = pPeak - 1; i > pOnset; --i) {
            if (filteredECG[i] < filteredECG[i - 1]) { // Descenso significativo
                pWaveStart = i;
                break;
            }
        }
        for (int i = pPeak + 1; i < pOffset; ++i) {
            if (filteredECG[i] > filteredECG[i + 1]) { // Ascenso significativo
                pWaveEnd = i;
                break;
            }
        }

        waveforms["P"] = {
            pWaveStart / samplingRate * 1000.0,
            pWaveEnd / samplingRate * 1000.0,
            filteredECG[pPeak],
            (pWaveEnd - pWaveStart) / samplingRate * 1000.0
        };

        // --- Complejo QRS ---
        int qrsOnset = rPeak - static_cast<int>(samplingRate * 0.03); // Buscar cerca del R
        int qrsOffset = std::min(signalSize - 1, rPeak + static_cast<int>(samplingRate * 0.1));
        
        // Derivada para detectar inicio y fin
        int qrsStart = qrsOnset, qrsEnd = qrsOffset;
        for (int i = rPeak - 1; i > qrsOnset; --i) {
            if (std::abs(filteredECG[i] - filteredECG[i - 1]) < 0.1) { // Cambio menor en pendiente
                qrsStart = i;
                break;
            }
        }
        for (int i = rPeak + 1; i < qrsOffset; ++i) {
            if (std::abs(filteredECG[i] - filteredECG[i + 1]) < 0.1) {
                qrsEnd = i;
                break;
            }
        }

        waveforms["QRS"] = {
            qrsStart / samplingRate * 1000.0,
            qrsEnd / samplingRate * 1000.0,
            filteredECG[rPeak],
            (qrsEnd - qrsStart) / samplingRate * 1000.0
        };

        // --- Onda T ---
        int tOnset = rPeak + static_cast<int>(samplingRate * 0.1); // Después del QRS
        int tOffset = std::min(rPeak + static_cast<int>(samplingRate * 0.4), signalSize - 1);
        auto tMax = std::max_element(filteredECG.begin() + tOnset, filteredECG.begin() + tOffset);
        int tPeak = std::distance(filteredECG.begin(), tMax);

        // Derivada para detectar inicio y fin
        int tWaveStart = tPeak, tWaveEnd = tPeak;
        for (int i = tPeak - 1; i > tOnset; --i) {
            if (filteredECG[i] < filteredECG[i - 1]) {
                tWaveStart = i;
                break;
            }
        }
        for (int i = tPeak + 1; i < tOffset; ++i) {
            if (filteredECG[i] > filteredECG[i + 1]) {
                tWaveEnd = i;
                break;
            }
        }

        waveforms["T"] = {
            tWaveStart / samplingRate * 1000.0,
            tWaveEnd / samplingRate * 1000.0,
            filteredECG[tPeak],
            (tWaveEnd - tWaveStart) / samplingRate * 1000.0
        };
    }

    return waveforms;
}

WaveAmplitudes calculateAmplitudes(const std::map<std::string, morphology_waveform>& waveforms) {
    return {
        waveforms.at("P").amplitude,
        waveforms.at("QRS").amplitude,
        waveforms.at("T").amplitude
    };
}


// Cálculo de relaciones de amplitud entre ondas
std::map<std::string, double> calculateAmplitudeRatios(const WaveAmplitudes& amplitudes) {
    std::map<std::string, double> amplitudeRatios;

    // Evitar divisiones por cero en los cálculos
    if (amplitudes.qrsAmplitude != 0) {
        amplitudeRatios["P/QRS"] = amplitudes.pAmplitude / amplitudes.qrsAmplitude;
        amplitudeRatios["T/QRS"] = amplitudes.tAmplitude / amplitudes.qrsAmplitude;
    } else {
        amplitudeRatios["P/QRS"] = 0;
        amplitudeRatios["T/QRS"] = 0;
    }

    if (amplitudes.tAmplitude != 0) {
        amplitudeRatios["P/T"] = amplitudes.pAmplitude / amplitudes.tAmplitude;
        amplitudeRatios["QRS/T"] = amplitudes.qrsAmplitude / amplitudes.tAmplitude;
    } else {
        amplitudeRatios["P/T"] = 0;
        amplitudeRatios["QRS/T"] = 0;
    }

    return amplitudeRatios;
}

// Función principal para obtener amplitudes de las ondas y calcular las relaciones
std::map<std::string, double> analyzeAmplitudeRatios(const std::vector<double>& ecgData, double samplingRate) {

    // Detectar picos R y otras ondas
    std::map<std::string, morphology_waveform> waveforms = analyzeWaveforms(ecgData, samplingRate);

    // Extraer amplitudes de las ondas P, QRS y T
    WaveAmplitudes amplitudes = { waveforms["P"].amplitude, waveforms["QRS"].amplitude, waveforms["T"].amplitude };

    // Calcular relaciones entre amplitudes
    return calculateAmplitudeRatios(amplitudes);
}

int main() {
    double samplingRate = 360.0; // Frecuencia de muestreo en Hz
    std::string option;
    std::cout << "Seleccione una opción:\n1. Crear plantilla\n2. Verificar archivo\n";
    std::cin >> option;

    if (option == "1") {
        // Crear una nueva plantilla
        std::string filename;
        std::cout << "Ingrese el nombre del archivo CSV para crear la plantilla: ";
        std::cin >> filename;

        auto ecgData = loadECGData(filename);
        
        // Extracción de características
        ECGFeatures features;
        features.morphology_waveforms = analyzeWaveforms(ecgData, samplingRate);
        features.rr_intervals = calculateRRIntervals(ecgData, samplingRate);
        features.amplitude_ratios = analyzeAmplitudeRatios(ecgData, samplingRate);

        // Mostrar características morfológicas de las ondas
        std::cout << "\n--- Características de las Ondas Morfológicas ---\n";
        for (const auto& [waveName, waveFeatures] : features.morphology_waveforms) {
            std::cout << waveName << " Wave -> Onset: " << waveFeatures.onset
                      << " ms, Offset: " << waveFeatures.offset
                      << " ms, Amplitude: " << waveFeatures.amplitude
                      << ", Width: " << waveFeatures.width << " ms" << std::endl;
        }

        // Mostrar intervalos R-R
        std::cout << "\n--- Intervalos R-R ---\n";
        std::cout << "Cantidad de Intervalos R-R: " << features.rr_intervals.size() << std::endl;
        std::cout << "Primeros 10 Intervalos R-R (ms): ";
        for (size_t i = 0; i < 10 && i < features.rr_intervals.size(); ++i) {
            std::cout << features.rr_intervals[i] << " ";
        }
        std::cout << std::endl;

        // Mostrar relaciones de amplitud
        std::cout << "\n--- Relaciones entre Amplitudes de Ondas ---\n";
        for (const auto& [ratioName, value] : features.amplitude_ratios) {
            std::cout << ratioName << ": " << value << std::endl;
        }

        // Análisis del dominio de la frecuencia
        features.frequency_domain = calculateFrequencyDomain(ecgData, samplingRate);
        analyzeFrequencyDomain(ecgData, samplingRate);
    }


    else {
        std::cout << "Opción no válida" << std::endl;
    }

    return 0;
}