#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <algorithm>
#include <cryptopp/aes.h>
#include <cryptopp/filters.h>
#include <cryptopp/modes.h>
#include <cryptopp/osrng.h>
#include "svm_classifier.h"
#include "ecg_feature_extraction.h"

using namespace CryptoPP;
using namespace std;

// Función para convertir un vector a texto (para encriptar)
string vectorToString(const vector<double>& data) {
    string result;
    for (double val : data) {
        result += to_string(val) + ",";
    }
    return result;
}

// Función para normalizar los datos en un rango de 0 a 1
vector<double> normalize(const vector<double>& data) {
    vector<double> normalizedData;
    if (data.empty()) return normalizedData;

    double minVal = *min_element(data.begin(), data.end());
    double maxVal = *max_element(data.begin(), data.end());

    if (minVal == maxVal) {  // Evitar división por cero si todos los valores son iguales
        normalizedData.assign(data.size(), 0.0);
        return normalizedData;
    }

    for (double val : data) {
        normalizedData.push_back((val - minVal) / (maxVal - minVal));
    }
    return normalizedData;
}

// Función de encriptación de AES
string encrypt(const string& plainText, const SecByteBlock& key, const CryptoPP::byte iv[AES::BLOCKSIZE]) {
    string cipherText;
    try {
        CBC_Mode<AES>::Encryption encryption;
        encryption.SetKeyWithIV(key, key.size(), iv);

        StringSource(plainText, true,
                     new StreamTransformationFilter(encryption,
                     new StringSink(cipherText)));
    } catch (const Exception& e) {
        cerr << e.what() << endl;
        exit(1);
    }
    return cipherText;
}

// Cargar datos reales de ECG desde un archivo CSV
vector<double> loadECGData(const string& filename) {
    vector<double> ecgData;
    ifstream file(filename);
    string line;

    getline(file, line); // Saltar la cabecera
    while (getline(file, line)) {
        stringstream ss(line);
        string value;
        int column = 0;
        double mlii_value;

        while (getline(ss, value, ',')) {
            if (column == 2) { // Asumiendo columna MLII para el valor del ECG
                mlii_value = stod(value);
            }
            column++;
        }
        ecgData.push_back(mlii_value);
    }

    return ecgData;
}

int main() {
    // Cargar los datos de ECG
    vector<double> ecgData = loadECGData("100.csv");

    // Mostrar los primeros 10 datos originales
    cout << "Loaded ECG Data (first 10 samples): ";
    for (int i = 0; i < 10 && i < ecgData.size(); ++i) {
        cout << ecgData[i] << " ";
    }
    cout << "...\n";

    // Normalizar los datos de ECG
    vector<double> normalizedECG = normalize(ecgData);
    cout << "Normalized ECG Data (first 10 samples): ";
    for (int i = 0; i < 10 && i < normalizedECG.size(); ++i) {
        cout << normalizedECG[i] << " ";
    }
    cout << "...\n";

    // Convertir los datos a texto para el cifrado
    string ecgString = vectorToString(normalizedECG);
    cout << "ECG Data as String (for encryption): " << ecgString.substr(0, 50) << "...\n";

    // Configurar cifrado AES
    AutoSeededRandomPool prng;
    SecByteBlock key(AES::DEFAULT_KEYLENGTH);
    prng.GenerateBlock(key, key.size());
    CryptoPP::byte iv[AES::BLOCKSIZE];
    prng.GenerateBlock(iv, sizeof(iv));

    // Encriptar los datos de ECG normalizados
    string encryptedECG = encrypt(ecgString, key, iv);
    cout << "Encrypted ECG Data: " << encryptedECG.substr(0, 50) << "...\n";

    // Extraer características del ECG
    vector<double> features = extractECGFeatures(normalizedECG);
    cout << "Extracted ECG Features: ";
    for (const auto& feature : features) {
        cout << feature << " ";
    }
    cout << endl;

    // Crear plantilla biométrica
    vector<double> biometricTemplate = createBiometricTemplate(features);
    cout << "Biometric Template (before encryption): ";
    for (const auto& value : biometricTemplate) {
        cout << value << " ";
    }
    cout << endl;

    // Convertir la plantilla a texto para el cifrado
    string templateString = vectorToString(biometricTemplate);
    cout << "Biometric Template as String (for encryption): " << templateString.substr(0, 50) << "...\n";

    // Encriptar la plantilla biométrica
    string encryptedTemplate = encrypt(templateString, key, iv);
    cout << "Encrypted Biometric Template: " << encryptedTemplate.substr(0, 50) << "...\n";

    // Clasificación con SVM
    SVMClassifier classifier;
    classifier.train({features}, {1}); // Simulación de entrenamiento
    int predictedClass = classifier.predict(biometricTemplate);

    // Resultado de la clasificación
    cout << "Predicted Class: " << predictedClass << " (1 = Verified, 0 = Not Verified)" << endl;

    // Resumen final para el usuario
    cout << "\n--- Summary ---\n";
    cout << "1. ECG Data Loaded: " << ecgData.size() << " samples.\n";
    cout << "2. Normalized ECG Data (first 10 samples): ";
    for (int i = 0; i < 10 && i < normalizedECG.size(); ++i) {
        cout << normalizedECG[i] << " ";
    }
    cout << "\n3. Extracted Features (first 10 values): ";
    for (int i = 0; i < 10 && i < features.size(); ++i) {
        cout << features[i] << " ";
    }
    cout << "\n4. Biometric Template (before encryption): ";
    for (int i = 0; i < 10 && i < biometricTemplate.size(); ++i) {
        cout << biometricTemplate[i] << " ";
    }
    cout << "\n5. Encrypted ECG Data (sample): " << encryptedECG.substr(0, 50) << "...\n";
    cout << "6. Encrypted Biometric Template (sample): " << encryptedTemplate.substr(0, 50) << "...\n";
    cout << "7. Predicted Class: " << predictedClass << " (1 = Verified, 0 = Not Verified)\n";

    return 0;
}
