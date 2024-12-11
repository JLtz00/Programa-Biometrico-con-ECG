#include <iostream>
#include <vector>
#include <cmath>
#include <algorithm>

// Parámetros para detección de picos
const double THRESHOLD_SCALE = 0.6;
const int INTEGRATION_WINDOW = 30;

// Función para calcular la derivada de la señal
std::vector<double> differentiate(const std::vector<double>& signal) {
    std::vector<double> differentiatedSignal(signal.size());
    differentiatedSignal[0] = 0;  // Inicializa el primer valor a cero

    for (size_t i = 1; i < signal.size(); ++i) {
        differentiatedSignal[i] = signal[i] - signal[i - 1];
    }
    return differentiatedSignal;
}

// Función para calcular la señal cuadrada
std::vector<double> squareSignal(const std::vector<double>& signal) {
    std::vector<double> squaredSignal(signal.size());
    for (size_t i = 0; i < signal.size(); ++i) {
        squaredSignal[i] = signal[i] * signal[i];
    }
    return squaredSignal;
}

// Función para la integración en ventana
std::vector<double> movingWindowIntegration(const std::vector<double>& signal, int windowSize) {
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

// Función para la detección de picos R con un umbral adaptativo
std::vector<int> detectRPeaks(const std::vector<double>& signal, double thresholdScale) {
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

// Función principal para la detección de picos R optimizada
int main() {
    std::vector<double> ecgData = {/* Datos de la señal ECG */};
    double samplingRate = 360.0;

    // Paso 1: Diferenciación
    std::vector<double> differentiatedSignal = differentiate(ecgData);

    // Paso 2: Señal cuadrada
    std::vector<double> squaredSignal = squareSignal(differentiatedSignal);

    // Paso 3: Integración en ventana
    std::vector<double> integratedSignal = movingWindowIntegration(squaredSignal, INTEGRATION_WINDOW);

    // Paso 4: Detección de picos R usando umbral adaptativo
    std::vector<int> rPeaks = detectRPeaks(integratedSignal, THRESHOLD_SCALE);

    // Imprimir los índices de los picos R detectados
    std::cout << "Índices de picos R detectados:" << std::endl;
    for (int peak : rPeaks) {
        std::cout << peak << std::endl;
    }

    return 0;
}
