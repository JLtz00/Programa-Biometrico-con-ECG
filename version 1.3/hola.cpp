if (option == "1") {
    std::string filename;
    std::cout << "Ingrese el nombre del archivo CSV para crear la plantilla: ";
    std::cin >> filename;

    auto ecgData = loadECGData(filename);
    ECGFeatures features;

    // Análisis morfológico
    features.morphology_waveforms = analyzeWaveforms(ecgData, samplingRate);
    features.rr_intervals = calculateRRIntervals(ecgData, samplingRate);
    features.amplitude_ratios = analyzeAmplitudeRatios(ecgData, samplingRate);

    // Análisis del dominio de la frecuencia
    features.frequency_domain = calculateFrequencyDomain(ecgData, samplingRate);
    analyzeFrequencyDomain(ecgData, samplingRate);
}
