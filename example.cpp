#include "dsp_speech_enhancement.h"
#include <vector>
#include <iostream>
#include <chrono>
#include <fstream>
#include <random>

// Utility function to generate synthetic noisy signal
std::vector<float> generateTestSignal(size_t size, float signalFreq, float noiseLevel) {
    std::vector<float> signal(size);
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> noise(0.0f, noiseLevel);
    
    for(size_t i = 0; i < size; i++) {
        // Generate pure sine wave
        float t = static_cast<float>(i) / 16000.0f;  // assuming 16kHz sample rate
        signal[i] = std::sin(2.0f * M_PI * signalFreq * t);
        // Add noise
        signal[i] += noise(gen);
    }
    
    return signal;
}

// Utility function to measure execution time
template<typename Func>
float measureExecutionTime(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / 1000.0f;
}

// Utility function to save signal to file
void saveSignal(const std::vector<float>& signal, const std::string& filename) {
    std::ofstream file(filename, std::ios::binary);
    file.write(reinterpret_cast<const char*>(signal.data()), signal.size() * sizeof(float));
}

int main() {
    // Parameters
    const size_t signalLength = 48000;  // 3 seconds at 16kHz
    const float signalFreq = 440.0f;    // 440Hz tone
    const float noiseLevel = 0.1f;      // Noise standard deviation
    
    // Generate test signal
    std::cout << "Generating test signal...\n";
    auto noisySignal = generateTestSignal(signalLength, signalFreq, noiseLevel);
    
    // Save original noisy signal
    saveSignal(noisySignal, "noisy_signal.raw");
    
    // Test different algorithms
    std::cout << "\nTesting different noise reduction algorithms:\n";
    
    // 1. Spectral Subtraction
    {
        dsp::NoiseReduction::SpectralSubtractionParams ssParams;
        ssParams.sampleRate = 16000;
        ssParams.frameSizeMs = 20;
        ssParams.overSubtractionFactor = 2.0f;
        ssParams.spectralFloor = 0.002f;
        
        float execTime = measureExecutionTime([&]() {
            auto enhanced = dsp::NoiseReduction::spectralSubtraction(noisySignal, ssParams);
            saveSignal(enhanced, "enhanced_spectral_subtraction.raw");
        });
        
        std::cout << "Spectral Subtraction completed in " << execTime << " seconds\n";
    }
    
    // 2. Wiener Filter
    {
        dsp::NoiseReduction::WienerFilterParams wfParams;
        wfParams.sampleRate = 16000;
        wfParams.frameSizeMs = 20;
        wfParams.smoothingFactor = 0.98f;
        
        float execTime = measureExecutionTime([&]() {
            auto enhanced = dsp::NoiseReduction::wienerFilter(noisySignal, wfParams);
            saveSignal(enhanced, "enhanced_wiener.raw");
        });
        
        std::cout << "Wiener Filter completed in " << execTime << " seconds\n";
    }
    
    // 3. MMSE STSA
    {
        dsp::NoiseReduction::MMSEParams mmseParams;
        mmseParams.sampleRate = 16000;
        mmseParams.frameSizeMs = 20;
        mmseParams.alpha = 0.98f;
        
        float execTime = measureExecutionTime([&]() {
            auto enhanced = dsp::NoiseReduction::mmseStsa(noisySignal, mmseParams);
            saveSignal(enhanced, "enhanced_mmse.raw");
        });
        
        std::cout << "MMSE STSA completed in " << execTime << " seconds\n";
    }
    
    // 4. Ephraim-Malah
    {
        dsp::NoiseReduction::EphraimMalahParams emParams;
        emParams.sampleRate = 16000;
        emParams.frameSizeMs = 20;
        
        float execTime = measureExecutionTime([&]() {
            auto enhanced = dsp::NoiseReduction::ephraimMalah(noisySignal, emParams);
            saveSignal(enhanced, "enhanced_ephraim_malah.raw");
        });
        
        std::cout << "Ephraim-Malah completed in " << execTime << " seconds\n";
    }
    
    // Compare results
    std::cout << "\nComparing results:\n";
    
    // Generate clean reference signal
    auto cleanSignal = generateTestSignal(signalLength, signalFreq, 0.0f);
    
    // Calculate SNR improvements for each method
    auto enhanced_ss = dsp::NoiseReduction::spectralSubtraction(noisySignal);
    auto enhanced_wiener = dsp::NoiseReduction::wienerFilter(noisySignal);
    auto enhanced_mmse = dsp::NoiseReduction::mmseStsa(noisySignal);
    auto enhanced_em = dsp::NoiseReduction::ephraimMalah(noisySignal);
    
    float initial_snr = dsp::DSPUtils::calculateSNR(cleanSignal, noisySignal);
    float ss_snr = dsp::DSPUtils::calculateSNR(cleanSignal, enhanced_ss);
    float wiener_snr = dsp::DSPUtils::calculateSNR(cleanSignal, enhanced_wiener);
    float mmse_snr = dsp::DSPUtils::calculateSNR(cleanSignal, enhanced_mmse);
    float em_snr = dsp::DSPUtils::calculateSNR(cleanSignal, enhanced_em);
    
    std::cout << "Initial SNR: " << initial_snr << " dB\n";
    std::cout << "Spectral Subtraction SNR: " << ss_snr << " dB (+" << ss_snr - initial_snr << " dB)\n";
    std::cout << "Wiener Filter SNR: " << wiener_snr << " dB (+" << wiener_snr - initial_snr << " dB)\n";
    std::cout << "MMSE STSA SNR: " << mmse_snr << " dB (+" << mmse_snr - initial_snr << " dB)\n";
    std::cout << "Ephraim-Malah SNR: " << em_snr << " dB (+" << em_snr - initial_snr << " dB)\n";
    
    // Calculate PESQ or other quality metrics if available
    // Note: PESQ implementation would require additional dependencies
    
    return 0;
}
