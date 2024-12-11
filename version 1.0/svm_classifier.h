#ifndef SVM_CLASSIFIER_H
#define SVM_CLASSIFIER_H

#include <iostream>
#include <vector>
#include <numeric>

using namespace std;

// Implementación simplificada de un clasificador SVM lineal
class SVMClassifier {
public:
    vector<double> weights;
    double bias;

    SVMClassifier() : bias(0.0) {}

    // Función de entrenamiento de SVM (datos linealmente separables)
    void train(const vector<vector<double>>& data, const vector<int>& labels, double learningRate = 0.01, int iterations = 1000) {
        size_t numFeatures = data[0].size();
        weights.resize(numFeatures, 0.0);

        for (int iter = 0; iter < iterations; ++iter) {
            for (size_t i = 0; i < data.size(); ++i) {
                double dotProduct = inner_product(data[i].begin(), data[i].end(), weights.begin(), 0.0);
                double prediction = dotProduct + bias;
                int label = labels[i];

                // Actualización de pesos
                if (label * prediction <= 0) {
                    for (size_t j = 0; j < numFeatures; ++j) {
                        weights[j] += learningRate * label * data[i][j];
                    }
                    bias += learningRate * label;
                }
            }
        }
    }

    // Función de predicción
    int predict(const vector<double>& features) const {
        double dotProduct = inner_product(features.begin(), features.end(), weights.begin(), 0.0);
        double prediction = dotProduct + bias;
        return (prediction >= 0) ? 1 : -1;  // Clasificación binaria
    }
};

#endif // SVM_CLASSIFIER_H

