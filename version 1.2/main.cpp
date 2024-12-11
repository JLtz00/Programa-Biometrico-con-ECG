#include <iostream>
#include <vector>
#include <fstream>
#include <string>
#include <sstream>
#include <cryptopp/aes.h>
#include <cryptopp/filters.h>
#include <cryptopp/modes.h>
#include <cryptopp/osrng.h>
#include "ecg_feature_extraction.h"

using namespace CryptoPP;
using namespace std;

double SIMILARITY_THRESHOLD = 0.85;

// Función para cargar datos de ECG desde un archivo CSV
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

// Función para normalizar el ECG
vector<double> normalize(const vector<double>& ecgData) {
    double maxVal = *max_element(ecgData.begin(), ecgData.end());
    double minVal = *min_element(ecgData.begin(), ecgData.end());
    vector<double> normalized;

    for (double value : ecgData) {
        normalized.push_back((value - minVal) / (maxVal - minVal));
    }
    return normalized;
}

// Función para convertir un vector en una cadena de texto
string vectorToString(const vector<double>& data) {
    string result;
    for (double val : data) {
        result += to_string(val) + ",";
    }
    return result;
}

// Función para cifrar una plantilla de ECG
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

// Función para descifrar una plantilla de ECG
vector<double> decryptToVector(const string& cipherText, const SecByteBlock& key, const CryptoPP::byte iv[AES::BLOCKSIZE]) {
    string decryptedText;
    vector<double> data;

    try {
        CBC_Mode<AES>::Decryption decryption;
        decryption.SetKeyWithIV(key, key.size(), iv);

        StringSource(cipherText, true,
                     new StreamTransformationFilter(decryption,
                     new StringSink(decryptedText)));
    } catch (const Exception& e) {
        cerr << e.what() << endl;
        exit(1);
    }

    stringstream ss(decryptedText);
    string value;
    while (getline(ss, value, ',')) {
        data.push_back(stod(value));
    }
    return data;
}

// Calcular la similitud entre dos plantillas
double calculateSimilarity(const vector<double>& template1, const vector<double>& template2) {
    double score = 0;
    for (size_t i = 0; i < template1.size(); ++i) {
        score += std::abs(template1[i] - template2[i]);
    }
    return 1 - (score / template1.size());
}

int main() {
    string option;
    vector<vector<double>> storedTemplates;
    vector<string> encryptedTemplates;

    while (true) {
        cout << "\nSeleccione una opción:\n1. Crear plantilla\n2. Verificar archivo\n3. Configurar umbral de similitud\n4. Salir\n";
        cin >> option;

        if (option == "1") {
            // Crear una nueva plantilla
            string filename;
            cout << "Ingrese el nombre del archivo CSV para crear la plantilla: ";
            cin >> filename;

            vector<double> ecgData = loadECGData(filename);
            vector<double> normalizedECG = normalize(ecgData);
            vector<double> features = extractECGFeatures(normalizedECG);

            // Encriptar y almacenar la plantilla biométrica
            AutoSeededRandomPool prng;
            SecByteBlock key(AES::DEFAULT_KEYLENGTH);
            prng.GenerateBlock(key, key.size());
            CryptoPP::byte iv[AES::BLOCKSIZE];
            prng.GenerateBlock(iv, sizeof(iv));

            string templateString = vectorToString(features);
            string encryptedTemplate = encrypt(templateString, key, iv);

            storedTemplates.push_back(features);
            encryptedTemplates.push_back(encryptedTemplate);

            cout << "Plantilla creada y encriptada. Detalles:\n";
            cout << "ECG Media: " << features[0] << ", Max: " << features[1] << ", Min: " << features[2] << "\n";
            cout << "Amplitudes (P, QRS, T): " << features[3] << ", " << features[4] << ", " << features[5] << "\n";
            cout << "Desviación estándar: " << features[6] << "\n";
            cout << "Plantilla encriptada: " << encryptedTemplate << "\n";

        } else if (option == "2") {
            // Verificar un archivo contra la plantilla
            string filename;
            cout << "Ingrese el nombre del archivo CSV para verificar: ";
            cin >> filename;

            vector<double> ecgData = loadECGData(filename);
            vector<double> normalizedECG = normalize(ecgData);
            vector<double> features = extractECGFeatures(normalizedECG);

            bool verified = false;
            for (const auto& storedTemplate : storedTemplates) {
                double similarity = calculateSimilarity(storedTemplate, features);
                if (similarity >= SIMILARITY_THRESHOLD) {
                    verified = true;
                    cout << "Verificación exitosa. Similaridad: " << similarity << endl;
                    break;
                }
            }
            if (!verified) {
                cout << "Verificación fallida. Los datos no coinciden con ninguna plantilla." << endl;
            }

        } else if (option == "3") {
            // Configuración del umbral de similitud
            cout << "Ingrese un nuevo valor de umbral de similitud (0.0 a 1.0): ";
            cin >> SIMILARITY_THRESHOLD;
            cout << "Nuevo umbral de similitud establecido en " << SIMILARITY_THRESHOLD << endl;
        } else if (option == "4") {
            break;
        } else {
            cout << "Opción no válida" << endl;
        }
    }

    return 0;
}
