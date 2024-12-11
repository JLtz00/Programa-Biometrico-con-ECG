#ifndef ECG_FEATURE_EXTRACTION_H
#define ECG_FEATURE_EXTRACTION_H

#include <vector>
#include <algorithm>
#include <numeric>

using namespace std;

// Extracción de características del ECG (ondas P, QRS, T)
vector<double> extractECGFeatures(const vector<double>& ecgData) {
    vector<double> features;
    double minVal = *min_element(ecgData.begin(), ecgData.end());
    double maxVal = *max_element(ecgData.begin(), ecgData.end());
    double meanVal = accumulate(ecgData.begin(), ecgData.end(), 0.0) / ecgData.size();

    // Simulación de extracción de características (valores básicos)
    features.push_back(minVal);   // Valor mínimo
    features.push_back(maxVal);   // Valor máximo
    features.push_back(meanVal);  // Valor medio

    return features;
}

// Creación de plantilla biométrica (simulación)
vector<double> createBiometricTemplate(const vector<double>& features) {
    // Aquí podrías aplicar una normalización o procesamiento específico
    vector<double> templateData = features;
    return templateData;
}

#endif
