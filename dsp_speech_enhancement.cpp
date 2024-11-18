#include "dsp_speech_enhancement.h"
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace dsp {

// FFT Implementation
std::vector<std::complex<float>> FFT::fft(const std::vector<float>& input) {
    size_t n = nextPowerOf2(input.size());
    std::vector<std::complex<float>> data(n);
    
    // Copy and pad input
    for(size_t i = 0; i < input.size(); i++) {
        data[i] = std::complex<float>(input[i], 0.0f);
    }
    for(size_t i = input.size(); i < n; i++) {
        data[i] = std::complex<float>(0.0f, 0.0f);
    }
    
    bitReverse(data);
    
    // Butterfly computation
    for(size_t s = 1; s <= std::log2(n); s++) {
        size_t m = 1 << s;
        std::complex<float> wm = std::polar(1.0f, -2.0f * float(M_PI) / float(m));
        
        for(size_t k = 0; k < n; k += m) {
            std::complex<float> w = 1.0f;
            for(size_t j = 0; j < m/2; j++) {
                std::complex<float> t = w * data[k + j + m/2];
                std::complex<float> u = data[k + j];
                data[k + j] = u + t;
                data[k + j + m/2] = u - t;
                w *= wm;
            }
        }
    }
    
    return data;
}

std::vector<float> FFT::ifft(const std::vector<std::complex<float>>& input) {
    std::vector<std::complex<float>> data = input;
    for(auto& x : data) x = std::conj(x);
    
    data = fft(std::vector<float>(data.size()));
    
    std::vector<float> result(input.size());
    float scale = 1.0f / input.size();
    for(size_t i = 0; i < input.size(); i++) {
        result[i] = std::real(std::conj(data[i])) * scale;
    }
    
    return result;
}

std::vector<std::complex<float>> FFT::fftshift(const std::vector<std::complex<float>>& input) {
    std::vector<std::complex<float>> output(input.size());
    size_t half = input.size() / 2;
    
    for(size_t i = 0; i < half; i++) {
        output[i] = input[i + half];
        output[i + half] = input[i];
    }
    
    return output;
}

std::vector<float> FFT::powerSpectrum(const std::vector<std::complex<float>>& input) {
    std::vector<float> power(input.size());
    for(size_t i = 0; i < input.size(); i++) {
        power[i] = std::norm(input[i]);
    }
    return power;
}

// DSP Utilities Implementation
std::vector<float> DSPUtils::hanningWindow(size_t size) {
    std::vector<float> window(size);
    for(size_t i = 0; i < size; i++) {
        window[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (size - 1)));
    }
    return window;
}

std::vector<float> DSPUtils::hammingWindow(size_t size) {
    std::vector<float> window(size);
    for(size_t i = 0; i < size; i++) {
        window[i] = 0.54f - 0.46f * std::cos(2.0f * M_PI * i / (size - 1));
    }
    return window;
}

std::vector<float> DSPUtils::blackmanWindow(size_t size) {
    std::vector<float> window(size);
    for(size_t i = 0; i < size; i++) {
        window[i] = 0.42f - 0.5f * std::cos(2.0f * M_PI * i / (size - 1)) 
                    + 0.08f * std::cos(4.0f * M_PI * i / (size - 1));
    }
    return window;
}

std::vector<float> DSPUtils::rectangularWindow(size_t size) {
    return std::vector<float>(size, 1.0f);
}

std::vector<float> DSPUtils::kaiserWindow(size_t size, float beta) {
    std::vector<float> window(size);
    float denom = NoiseReduction::modifiedBessel(beta);
    for(size_t i = 0; i < size; i++) {
        float x = beta * std::sqrt(1.0f - std::pow(2.0f * i / (size - 1) - 1.0f, 2));
        window[i] = NoiseReduction::modifiedBessel(x) / denom;
    }
    return window;
}

float DSPUtils::calculateSNR(const std::vector<float>& signal, const std::vector<float>& noise) {
    if(signal.size() != noise.size()) {
        throw std::invalid_argument("Signal and noise must have same length");
    }
    
    float signalPower = calculateRMS(signal);
    float noisePower = calculateRMS(noise);
    
    return 20.0f * std::log10(signalPower / noisePower);
}

float DSPUtils::calculatePSNR(const std::vector<float>& original, const std::vector<float>& processed) {
    if(original.size() != processed.size()) {
        throw std::invalid_argument("Signals must have same length");
    }
    
    float maxVal = *std::max_element(original.begin(), original.end());
    float mse = 0.0f;
    
    for(size_t i = 0; i < original.size(); i++) {
        float diff = original[i] - processed[i];
        mse += diff * diff;
    }
    
    mse /= original.size();
    return 20.0f * std::log10(maxVal / std::sqrt(mse));
}

std::vector<float> DSPUtils::movingAverage(const std::vector<float>& input, size_t windowSize) {
    std::vector<float> output(input.size());
    float sum = 0.0f;
    
    // Initial window
    for(size_t i = 0; i < windowSize && i < input.size(); i++) {
        sum += input[i];
        output[i] = sum / (i + 1);
    }
    
    // Moving window
    for(size_t i = windowSize; i < input.size(); i++) {
        sum = sum - input[i - windowSize] + input[i];
        output[i] = sum / windowSize;
    }
    
    return output;
}

float DSPUtils::calculateRMS(const std::vector<float>& input) {
    float sum = 0.0f;
    for(float x : input) {
        sum += x * x;
    }
    return std::sqrt(sum / input.size());
}

std::pair<float, float> DSPUtils::calculateMeanAndVariance(const std::vector<float>& input) {
    float mean = std::accumulate(input.begin(), input.end(), 0.0f) / input.size();
    
    float variance = 0.0f;
    for(float x : input) {
        float diff = x - mean;
        variance += diff * diff;
    }
    variance /= input.size();
    
    return {mean, variance};
}

std::vector<float> DSPUtils::normalize(const std::vector<float>& input) {
    float maxAbs = 0.0f;
    for(float x : input) {
        maxAbs = std::max(maxAbs, std::abs(x));
    }
    
    if(maxAbs == 0.0f) return input;
    
    std::vector<float> output(input.size());
    for(size_t i = 0; i < input.size(); i++) {
        output[i] = input[i] / maxAbs;
    }
    
    return output;
}

std::vector<float> DSPUtils::preEmphasis(const std::vector<float>& input, float coefficient) {
    std::vector<float> output(input.size());
    output[0] = input[0];
    
    for(size_t i = 1; i < input.size(); i++) {
        output[i] = input[i] - coefficient * input[i-1];
    }
    
    return output;
}

std::vector<float> DSPUtils::deEmphasis(const std::vector<float>& input, float coefficient) {
    std::vector<float> output(input.size());
    output[0] = input[0];
    
    for(size_t i = 1; i < input.size(); i++) {
        output[i] = input[i] + coefficient * output[i-1];
    }
    
    return output;
}

// Noise Reduction Implementation
std::vector<float> NoiseReduction::spectralSubtraction(const std::vector<float>& input, 
                                                      const SpectralSubtractionParams& params) {
    int frameSize = (params.frameSizeMs * params.sampleRate) / 1000;
    std::vector<float> output(input.size());
    
// Get window function
    std::vector<float> window;
    if(params.windowType == "hanning") {
        window = DSPUtils::hanningWindow(frameSize);
    } else if(params.windowType == "hamming") {
        window = DSPUtils::hammingWindow(frameSize);
    } else if(params.windowType == "blackman") {
        window = DSPUtils::blackmanWindow(frameSize);
    } else {
        window = DSPUtils::rectangularWindow(frameSize);
    }

    // Process frame by frame with 50% overlap
    int hopSize = frameSize / 2;
    std::vector<float> noisePSD(frameSize/2 + 1, 0.0f);
    bool firstFrames = true;
    
    for(size_t i = 0; i < input.size(); i += hopSize) {
        // Extract frame
        std::vector<float> frame(frameSize, 0.0f);
        size_t remainingSamples = std::min(frameSize, (int)(input.size() - i));
        for(size_t j = 0; j < remainingSamples; j++) {
            frame[j] = input[i + j] * window[j];
        }
        
        // Compute FFT
        auto spectrum = FFT::fft(frame);
        
        // Estimate noise power during first few frames
        if(firstFrames && i < frameSize * 5) {
            for(size_t j = 0; j < noisePSD.size(); j++) {
                float power = std::norm(spectrum[j]);
                noisePSD[j] = (i == 0) ? power : 
                    0.9f * noisePSD[j] + 0.1f * power;
            }
        }
        
        // Apply spectral subtraction
        for(size_t j = 0; j < spectrum.size(); j++) {
            float magnitude = std::abs(spectrum[j]);
            float phase = std::arg(spectrum[j]);
            
            float newMagnitude = magnitude * magnitude - 
                params.overSubtractionFactor * noisePSD[j];
            newMagnitude = std::max(newMagnitude, 
                params.spectralFloor * magnitude * magnitude);
            newMagnitude = std::sqrt(newMagnitude);
            
            spectrum[j] = std::polar(newMagnitude, phase);
        }
        
        // Inverse FFT and overlap-add
        auto cleanFrame = FFT::ifft(spectrum);
        for(size_t j = 0; j < remainingSamples; j++) {
            output[i + j] += cleanFrame[j] * window[j];
        }
        
        if(i >= frameSize * 5) firstFrames = false;
    }
    
    // Normalize for overlap-add
    for(size_t i = 0; i < output.size(); i++) {
        output[i] /= 2.0f;  // Due to 50% overlap
    }
    
    return output;
}

std::vector<float> NoiseReduction::wienerFilter(const std::vector<float>& input,
                                               const WienerFilterParams& params) {
    int frameSize = (params.frameSizeMs * params.sampleRate) / 1000;
    std::vector<float> output(input.size());
    std::vector<float> previousPSD(frameSize/2 + 1, 0.0f);
    
    auto window = DSPUtils::hanningWindow(frameSize);
    int hopSize = frameSize / 2;
    
    for(size_t i = 0; i < input.size(); i += hopSize) {
        // Extract frame
        std::vector<float> frame(frameSize, 0.0f);
        size_t remainingSamples = std::min(frameSize, (int)(input.size() - i));
        for(size_t j = 0; j < remainingSamples; j++) {
            frame[j] = input[i + j] * window[j];
        }
        
        auto spectrum = FFT::fft(frame);
        
        // Estimate noise and signal PSDs
        auto noisePSD = estimateNoisePSD(spectrum);
        auto signalPSD = estimateSignalPSD(spectrum, previousPSD, params.smoothingFactor);
        
        // Apply Wiener filter
        for(size_t j = 0; j < spectrum.size(); j++) {
            float gain = signalPSD[j] / (signalPSD[j] + noisePSD[j]);
            spectrum[j] *= gain;
        }
        
        // Store current PSD
        previousPSD = signalPSD;
        
        // Inverse FFT and overlap-add
        auto cleanFrame = FFT::ifft(spectrum);
        for(size_t j = 0; j < remainingSamples; j++) {
            output[i + j] += cleanFrame[j] * window[j];
        }
    }
    
    // Normalize for overlap-add
    for(size_t i = 0; i < output.size(); i++) {
        output[i] /= 2.0f;
    }
    
    return output;
}

std::vector<float> NoiseReduction::mmseStsa(const std::vector<float>& input,
                                           const MMSEParams& params) {
    int frameSize = (params.frameSizeMs * params.sampleRate) / 1000;
    std::vector<float> output(input.size());
    std::vector<float> noisePSD(frameSize/2 + 1, 0.0f);
    
    auto window = DSPUtils::hanningWindow(frameSize);
    int hopSize = frameSize / 2;
    
    for(size_t i = 0; i < input.size(); i += hopSize) {
        // Extract frame
        std::vector<float> frame(frameSize, 0.0f);
        size_t remainingSamples = std::min(frameSize, (int)(input.size() - i));
        for(size_t j = 0; j < remainingSamples; j++) {
            frame[j] = input[i + j] * window[j];
        }
        
        auto spectrum = FFT::fft(frame);
        
        // Update noise estimate
        if(i < frameSize * 5) {
            for(size_t j = 0; j < noisePSD.size(); j++) {
                float power = std::norm(spectrum[j]);
                noisePSD[j] = (i == 0) ? power : 
                    params.alpha * noisePSD[j] + (1 - params.alpha) * power;
            }
        }
        
        // Compute a posteriori SNR
        std::vector<float> snrPost(spectrum.size());
        for(size_t j = 0; j < spectrum.size(); j++) {
            float power = std::norm(spectrum[j]);
            snrPost[j] = power / (params.noiseOverestimation * noisePSD[j]);
        }
        
        // Apply MMSE gain
        for(size_t j = 0; j < spectrum.size(); j++) {
            float v = snrPost[j] / (1 + snrPost[j]);
            float gain = std::sqrt(M_PI) / 2 * std::sqrt(v) * 
                std::exp(-v/2) * (1 + v) * modifiedBessel(v/2);
            spectrum[j] *= gain;
        }
        
        // Inverse FFT and overlap-add
        auto cleanFrame = FFT::ifft(spectrum);
        for(size_t j = 0; j < remainingSamples; j++) {
            output[i + j] += cleanFrame[j] * window[j];
        }
    }
    
    // Normalize for overlap-add
    for(size_t i = 0; i < output.size(); i++) {
        output[i] /= 2.0f;
    }
    
    return output;
}

float NoiseReduction::modifiedBessel(float x) {
    float sum = 1.0f;
    float term = 0.25f * x * x;
    
    for(int k = 1; k < 10; k++) {
        sum += term;
        term *= 0.25f * x * x / (k * k);
    }
    
    return sum;
}

std::vector<float> NoiseReduction::estimateNoisePSD(
    const std::vector<std::complex<float>>& spectrum) {
    
    std::vector<float> psd(spectrum.size());
    for(size_t i = 0; i < spectrum.size(); i++) {
        psd[i] = std::norm(spectrum[i]);
    }
    return psd;
}

std::vector<float> NoiseReduction::estimateSignalPSD(
    const std::vector<std::complex<float>>& spectrum,
    const std::vector<float>& previousPSD,
    float smoothingFactor) {
    
    std::vector<float> psd(spectrum.size());
    for(size_t i = 0; i < spectrum.size(); i++) {
        float currentPSD = std::norm(spectrum[i]);
        psd[i] = smoothingFactor * previousPSD[i] + 
            (1 - smoothingFactor) * currentPSD;
    }
    return psd;
}

// Helper function for LogMMSE
float NoiseReduction::expint(float x) {
    if(x <= 0) return std::numeric_limits<float>::infinity();
    if(x <= 1) {
        return -std::log(x) - 0.57721566f + x - 
            std::pow(x, 2)/4 + std::pow(x, 3)/18;
    }
    float result = (std::exp(-x)/x) * 
        (1 + 1/x + 2/std::pow(x,2) + 6/std::pow(x,3));
    return result;
}

std::vector<float> NoiseReduction::ephraimMalah(const std::vector<float>& input,
                                               const EphraimMalahParams& params) {
    int frameSize = (params.frameSizeMs * params.sampleRate) / 1000;
    std::vector<float> output(input.size());
    auto window = DSPUtils::hanningWindow(frameSize);
    int hopSize = frameSize / 2;

    std::vector<float> previousPSD(frameSize/2 + 1, 0.0f);
    std::vector<float> previousGain(frameSize/2 + 1, 1.0f);

    for(size_t i = 0; i < input.size(); i += hopSize) {
        // Extract and window frame
        std::vector<float> frame(frameSize, 0.0f);
        size_t remainingSamples = std::min(frameSize, (int)(input.size() - i));
        for(size_t j = 0; j < remainingSamples; j++) {
            frame[j] = input[i + j] * window[j];
        }

        auto spectrum = FFT::fft(frame);
        auto noisePSD = estimateNoisePSD(spectrum);

        // Apply Ephraim-Malah gain
        for(size_t j = 0; j < spectrum.size(); j++) {
            float postSNR = std::norm(spectrum[j]) / noisePSD[j];
            float priorSNR = params.alpha * std::pow(previousGain[j], 2) * previousPSD[j] / noisePSD[j] +
                            (1 - params.alpha) * std::max(postSNR - 1.0f, 0.0f);
            
            float v = priorSNR * postSNR / (1.0f + priorSNR);
            float gain = (std::sqrt(v) / (1.0f + v)) * expint(v/2);
            gain = std::max(gain, params.minGain);
            
            spectrum[j] *= gain;
            previousGain[j] = gain;
            previousPSD[j] = std::norm(spectrum[j]);
        }

        // IFFT and overlap-add
        auto cleanFrame = FFT::ifft(spectrum);
        for(size_t j = 0; j < remainingSamples; j++) {
            output[i + j] += cleanFrame[j] * window[j];
        }
    }

    // Normalize
    for(auto& sample : output) sample /= 2.0f;
    
    return output;
}

std::vector<float> NoiseReduction::regularizedSS(const std::vector<float>& input,
                                               const RegularizedSSParams& params) {
    int frameSize = (params.frameSizeMs * params.sampleRate) / 1000;
    std::vector<float> output(input.size());
    auto window = DSPUtils::hanningWindow(frameSize);
    int hopSize = frameSize / 2;

    for(size_t i = 0; i < input.size(); i += hopSize) {
        std::vector<float> frame(frameSize, 0.0f);
        size_t remainingSamples = std::min(frameSize, (int)(input.size() - i));
        for(size_t j = 0; j < remainingSamples; j++) {
            frame[j] = input[i + j] * window[j];
        }

        auto spectrum = FFT::fft(frame);
        auto noisePSD = estimateNoisePSD(spectrum);

        // Apply regularized spectral subtraction
        for(size_t j = 0; j < spectrum.size(); j++) {
            float magnitude = std::abs(spectrum[j]);
            float phase = std::arg(spectrum[j]);
            
            float power = std::pow(magnitude, 2);
            float noisePower = params.overSubtractionFactor * noisePSD[j];
            float regularizedPower = std::max(power - noisePower, 
                params.lambda * power) / (1 + params.lambda);
            
            spectrum[j] = std::polar(std::sqrt(regularizedPower), phase);
        }

        auto cleanFrame = FFT::ifft(spectrum);
        for(size_t j = 0; j < remainingSamples; j++) {
            output[i + j] += cleanFrame[j] * window[j];
        }
    }

    for(auto& sample : output) sample /= 2.0f;
    return output;
}

std::vector<float> NoiseReduction::psychoacousticFilter(const std::vector<float>& input,
                                                      const PsychoacousticParams& params) {
    int frameSize = (params.frameSizeMs * params.sampleRate) / 1000;
    std::vector<float> output(input.size());
    auto window = DSPUtils::hanningWindow(frameSize);
    int hopSize = frameSize / 2;

    for(size_t i = 0; i < input.size(); i += hopSize) {
        std::vector<float> frame(frameSize, 0.0f);
        size_t remainingSamples = std::min(frameSize, (int)(input.size() - i));
        for(size_t j = 0; j < remainingSamples; j++) {
            frame[j] = input[i + j] * window[j];
        }

        auto spectrum = FFT::fft(frame);
        auto maskingThresholds = computeMaskingThresholds(spectrum);
        auto noisePSD = estimateNoisePSD(spectrum);

        // Apply psychoacoustic filtering
        for(size_t j = 0; j < spectrum.size(); j++) {
            float magnitude = std::abs(spectrum[j]);
            float phase = std::arg(spectrum[j]);
            
            if(magnitude < maskingThresholds[j]) {
                magnitude = maskingThresholds[j];
            } else {
                float snr = std::pow(magnitude, 2) / noisePSD[j];
                magnitude *= (1.0f - std::exp(-snr * params.spreadingFactor));
            }
            
            spectrum[j] = std::polar(magnitude, phase);
        }

        auto cleanFrame = FFT::ifft(spectrum);
        for(size_t j = 0; j < remainingSamples; j++) {
            output[i + j] += cleanFrame[j] * window[j];
        }
    }

    for(auto& sample : output) sample /= 2.0f;
    return output;
}

std::vector<float> NoiseReduction::subspaceEnhancement(const std::vector<float>& input,
                                                     const SubspaceParams& params) {
    int frameSize = (params.frameSizeMs * params.sampleRate) / 1000;
    std::vector<float> output(input.size());
    auto window = DSPUtils::hanningWindow(frameSize);
    int hopSize = frameSize / 2;

    for(size_t i = 0; i < input.size(); i += hopSize) {
        std::vector<float> frame(frameSize, 0.0f);
        size_t remainingSamples = std::min(frameSize, (int)(input.size() - i));
        for(size_t j = 0; j < remainingSamples; j++) {
            frame[j] = input[i + j] * window[j];
        }

        // Construct Hankel matrix
        auto hankelMatrix = computeHankelMatrix(frame);
        
        // Perform SVD
        std::pair<std::vector<std::vector<float>>, std::vector<float>> svdResult = computeSVD(hankelMatrix);
        const auto& U = svdResult.first;
        auto& S = svdResult.second;
        
        // Truncate to signal subspace
        for(size_t j = params.signalSubspaceDim; j < S.size(); j++) {
            if(S[j] < params.svdThreshold * S[0]) {
                S[j] = 0.0f;
            }
        }
        
        // Reconstruct frame
        std::vector<float> cleanFrame(frameSize, 0.0f);
        for(size_t j = 0; j < frameSize; j++) {
            for(size_t k = 0; k < params.signalSubspaceDim; k++) {
                cleanFrame[j] += U[j][k] * S[k];
            }
        }

        // Overlap-add
        for(size_t j = 0; j < remainingSamples; j++) {
            output[i + j] += cleanFrame[j] * window[j];
        }
    }

    for(auto& sample : output) sample /= 2.0f;
    return output;
}

} // namespace dsp