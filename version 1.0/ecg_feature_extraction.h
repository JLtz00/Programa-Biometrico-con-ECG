#ifndef ECG_FEATURE_EXTRACTION_H
#define ECG_FEATURE_EXTRACTION_H

#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>

using namespace std;

// Función para detectar picos en una señal de ECG
vector<size_t> detectPeaks(const vector<double>& signal, double threshold) {
    vector<size_t> peakIndices;
    for (size_t i = 1; i < signal.size() - 1; i++) {
        // Detectar un pico si el valor es mayor que el umbral y sus vecinos
        if (signal[i] > signal[i-1] && signal[i] > signal[i+1] && signal[i] > threshold) {
            peakIndices.push_back(i);
        }
    }
    return peakIndices;
}

// Función para extraer características P, QRS, T de la señal de ECG
vector<double> extractECGFeatures(const vector<double>& ecgData) {
    vector<double> features;

    // Simulamos la detección de picos para las ondas P, QRS, y T
    vector<size_t> peaks = detectPeaks(ecgData, 0.5);  // Detección de picos con un umbral

    // Si se detectan picos, se extraen como características
    if (peaks.size() >= 3) {
        double pWave = ecgData[peaks[0]];  // Onda P
        double qrsComplex = ecgData[peaks[1]];  // Complejo QRS
        double tWave = ecgData[peaks[2]];  // Onda T
        
        // Agregar las características extraídas al vector de características
        features.push_back(pWave);
        features.push_back(qrsComplex);
        features.push_back(tWave);
    } else {
        // En caso de que no se detecten suficientes picos, se agregan valores por defecto
        features.push_back(0.0);  // Onda P
        features.push_back(0.0);  // Complejo QRS
        features.push_back(0.0);  // Onda T
    }

    return features;
}

#endif // ECG_FEATURE_EXTRACTION_H

