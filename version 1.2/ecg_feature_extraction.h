#include <vector>
#include <algorithm>
#include <cmath>
#include <numeric>

std::vector<double> extractECGFeatures(const std::vector<double>& normalizedECG) {
    std::vector<double> features;

    double mean = std::accumulate(normalizedECG.begin(), normalizedECG.end(), 0.0) / normalizedECG.size();
    double maxVal = *std::max_element(normalizedECG.begin(), normalizedECG.end());
    double minVal = *std::min_element(normalizedECG.begin(), normalizedECG.end());

    features.push_back(mean);         // Media del ECG
    features.push_back(maxVal);       // Máximo valor
    features.push_back(minVal);       // Mínimo valor

    double pWaveAmp = 0, qrsAmp = 0, tWaveAmp = 0;
    int pWaveCount = 0, qrsCount = 0, tWaveCount = 0;

    for (size_t i = 0; i < normalizedECG.size(); ++i) {
        double value = normalizedECG[i];
        if (value > 0.8 * maxVal) {
            pWaveAmp += value;
            pWaveCount++;
        } else if (value < 0.2 * maxVal) {
            qrsAmp += value;
            qrsCount++;
        } else {
            tWaveAmp += value;
            tWaveCount++;
        }
    }

    features.push_back(pWaveAmp / (pWaveCount ? pWaveCount : 1));
    features.push_back(qrsAmp / (qrsCount ? qrsCount : 1));
    features.push_back(tWaveAmp / (tWaveCount ? tWaveCount : 1));

    double variance = 0;
    for (const auto& val : normalizedECG) {
        variance += std::pow(val - mean, 2);
    }
    features.push_back(std::sqrt(variance / normalizedECG.size()));

    return features;
}
