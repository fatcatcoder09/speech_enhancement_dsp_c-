#ifndef NOISE_REDUCTION_H
#define NOISE_REDUCTION_H

#include <vector>
#include <complex>
#include <cmath>
#include <algorithm>

namespace dsp {

// Forward declarations
class FFT;
class NoiseReduction;

/**
 * FFT Implementation class
 * Provides Fast Fourier Transform utilities
 */
class FFT {
public:
    static std::vector<std::complex<float>> fft(const std::vector<float>& input);
    static std::vector<float> ifft(const std::vector<std::complex<float>>& input);
    static std::vector<std::complex<float>> fftshift(const std::vector<std::complex<float>>& input);
    static std::vector<float> powerSpectrum(const std::vector<std::complex<float>>& input);
    
private:
    static size_t nextPowerOf2(size_t n);
    static void bitReverse(std::vector<std::complex<float>>& data);
    static size_t reverseBits(size_t x, size_t bits);
};

/**
 * DSP Utilities class
 * Provides common DSP operations and window functions
 */
class DSPUtils {
public:
    // Window functions
    static std::vector<float> hanningWindow(size_t size);
    static std::vector<float> hammingWindow(size_t size);
    static std::vector<float> blackmanWindow(size_t size);
    static std::vector<float> rectangularWindow(size_t size);
    static std::vector<float> kaiserWindow(size_t size, float beta);
    
    // Analysis functions
    static float calculateSNR(const std::vector<float>& signal, const std::vector<float>& noise);
    static float calculatePSNR(const std::vector<float>& original, const std::vector<float>& processed);
    static float calculateSegmentalSNR(const std::vector<float>& signal, const std::vector<float>& noise, 
                                     size_t frameSize);
    
    // Statistical functions
    static std::vector<float> movingAverage(const std::vector<float>& input, size_t windowSize);
    static float calculateRMS(const std::vector<float>& input);
    static std::pair<float, float> calculateMeanAndVariance(const std::vector<float>& input);
    
    // Audio processing utilities
    static std::vector<float> normalize(const std::vector<float>& input);
    static std::vector<float> preEmphasis(const std::vector<float>& input, float coefficient = 0.97f);
    static std::vector<float> deEmphasis(const std::vector<float>& input, float coefficient = 0.97f);
};

/**
 * Main Noise Reduction class
 * Contains all noise reduction algorithms
 */
class NoiseReduction {
public:
    // Configuration structs for each algorithm
    struct SpectralSubtractionParams {
        SpectralSubtractionParams() : 
            overSubtractionFactor(2.0f),
            spectralFloor(0.002f),
            frameSizeMs(20),
            sampleRate(16000),
            windowType("hanning") {}
            
        float overSubtractionFactor;
        float spectralFloor;
        int frameSizeMs;
        int sampleRate;
        std::string windowType;
    };

    struct WienerFilterParams {
        WienerFilterParams() :
            smoothingFactor(0.98f),
            frameSizeMs(20),
            sampleRate(16000),
            priorSNR(15.0f) {}
            
        float smoothingFactor;
        int frameSizeMs;
        int sampleRate;
        float priorSNR;
    };

    struct MMSEParams {
        MMSEParams() :
            alpha(0.98f),
            beta(0.8f),
            frameSizeMs(20),
            sampleRate(16000),
            noiseOverestimation(1.0f) {}
            
        float alpha;
        float beta;
        int frameSizeMs;
        int sampleRate;
        float noiseOverestimation;
    };

    struct LogMMSEParams {
        LogMMSEParams() :
            alpha(0.98f),
            beta(0.8f),
            gamma(0.2f),
            frameSizeMs(20),
            sampleRate(16000) {}
            
        float alpha;
        float beta;
        float gamma;
        int frameSizeMs;
        int sampleRate;
    };

    struct KalmanParams {
        KalmanParams() :
            processNoise(1e-4f),
            measurementNoise(1e-2f),
            order(2) {}
            
        float processNoise;
        float measurementNoise;
        int order;
    };

    struct EphraimMalahParams {
        EphraimMalahParams() :
            alpha(0.98f),
            beta(0.8f),
            frameSizeMs(20),
            sampleRate(16000),
            minGain(0.1f) {}
            
        float alpha;
        float beta;
        int frameSizeMs;
        int sampleRate;
        float minGain;
    };

    struct RegularizedSSParams {
        RegularizedSSParams() :
            lambda(0.1f),
            frameSizeMs(20),
            sampleRate(16000),
            overSubtractionFactor(1.5f) {}
            
        float lambda;
        int frameSizeMs;
        int sampleRate;
        float overSubtractionFactor;
    };

    struct PsychoacousticParams {
        PsychoacousticParams() :
            frameSizeMs(20),
            sampleRate(16000),
            spreadingFactor(0.2f),
            maskingThreshold(-60.0f) {}
            
        int frameSizeMs;
        int sampleRate;
        float spreadingFactor;
        float maskingThreshold;
    };

    struct SubspaceParams {
        SubspaceParams() :
            signalSubspaceDim(8),
            frameSizeMs(20),
            sampleRate(16000),
            svdThreshold(0.01f) {}
            
        int signalSubspaceDim;
        int frameSizeMs;
        int sampleRate;
        float svdThreshold;
    };

    // Main processing functions
    static std::vector<float> spectralSubtraction(const std::vector<float>& input, 
                                                const SpectralSubtractionParams& params = SpectralSubtractionParams());
    
    static std::vector<float> wienerFilter(const std::vector<float>& input,
                                         const WienerFilterParams& params = WienerFilterParams());
    
    static std::vector<float> mmseStsa(const std::vector<float>& input,
                                     const MMSEParams& params = MMSEParams());
    
    static std::vector<float> logMMSE(const std::vector<float>& input,
                                    const LogMMSEParams& params = LogMMSEParams());
    
    static std::vector<float> kalmanFilter(const std::vector<float>& input,
                                         const KalmanParams& params = KalmanParams());

    static std::vector<float> ephraimMalah(const std::vector<float>& input,
                                         const EphraimMalahParams& params = EphraimMalahParams());
    
    static std::vector<float> regularizedSS(const std::vector<float>& input,
                                          const RegularizedSSParams& params = RegularizedSSParams());
    
    static std::vector<float> psychoacousticFilter(const std::vector<float>& input,
                                                 const PsychoacousticParams& params = PsychoacousticParams());
    
    static std::vector<float> subspaceEnhancement(const std::vector<float>& input,
                                                const SubspaceParams& params = SubspaceParams());

    // Make modifiedBessel public since it's used by DSPUtils
    static float modifiedBessel(float x);

private:
    // Helper functions
    static std::vector<float> estimateNoisePSD(const std::vector<std::complex<float>>& spectrum);
    static std::vector<float> estimateSignalPSD(const std::vector<std::complex<float>>& spectrum,
                                               const std::vector<float>& previousPSD,
                                               float smoothingFactor);
    static float expint(float x);

    // New helper functions
    static std::vector<float> computeMaskingThresholds(const std::vector<std::complex<float>>& spectrum);
    static std::vector<std::vector<float>> computeHankelMatrix(const std::vector<float>& frame);
    static std::pair<std::vector<std::vector<float>>, std::vector<float>> computeSVD(
        const std::vector<std::vector<float>>& matrix);
};

} // namespace dsp

#endif // NOISE_REDUCTION_H