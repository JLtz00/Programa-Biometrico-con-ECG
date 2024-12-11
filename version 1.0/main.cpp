#include <iostream>
#include <vector>
#include <string>
#include <cryptopp/aes.h>
#include <cryptopp/filters.h>
#include <cryptopp/modes.h>
#include <cryptopp/osrng.h>
#include <cryptopp/base64.h>
#include "ecg_feature_extraction.h"
#include "svm_classifier.h"

using namespace std;
using namespace CryptoPP;

// Función para convertir un vector de `double` a una cadena de texto
string vectorToString(const vector<double>& data) {
    string result;
    for (double val : data) {
        result += to_string(val) + ",";
    }
    return result;
}

// Función de encriptación AES
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

// Función para convertir datos binarios a Base64
string toBase64(const string& binary) {
    string base64Encoded;
    StringSource(binary, true,
                 new Base64Encoder(
                 new StringSink(base64Encoded), false)); // "false" evita las nuevas líneas
    return base64Encoded;
}

int main() {
    // Simulación de datos capturados del ECG
    vector<double> ecgCapturedData = {0.02, 0.03, -0.01, 0.15, 0.85, 1.02, 0.75, 0.35, -0.15, -0.5};

    // Extracción de características (ondas P, QRS, T)
    vector<double> ecgFeatures = extractECGFeatures(ecgCapturedData);
    
    // Convertir las características del ECG a texto
    string ecgDataString = vectorToString(ecgFeatures);

    // Crear una plantilla biométrica simulada
    vector<double> biometricTemplate = {0.65, 0.9, 0.8};
    string templateString = vectorToString(biometricTemplate);

    // Configuración de la encriptación AES
    AutoSeededRandomPool prng;
    SecByteBlock key(AES::DEFAULT_KEYLENGTH);
    prng.GenerateBlock(key, key.size());

    CryptoPP::byte iv[AES::BLOCKSIZE];
    prng.GenerateBlock(iv, sizeof(iv));

    // Encriptar los datos del ECG
    string encryptedECG = encrypt(ecgDataString, key, iv);
    cout << "Encrypted ECG data (Base64): " << toBase64(encryptedECG) << endl;

    // Encriptar la plantilla biométrica
    string encryptedTemplate = encrypt(templateString, key, iv);
    cout << "Encrypted Biometric Template (Base64): " << toBase64(encryptedTemplate) << endl;

    // Clasificación
    SVMClassifier svm;
    svm.train({{0.2, 0.7, 0.5}, {0.1, 0.9, 0.6}, {-0.3, -0.8, -0.6}}, {1, 1, -1});
    int predictedClass = svm.predict(ecgFeatures);
    cout << "Predicted class: " << predictedClass << endl;

    return 0;
}

