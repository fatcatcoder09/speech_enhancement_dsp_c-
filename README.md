
```markdown
# Speech Enhancement DSP Library in C++

A modern C++ library implementing various speech enhancement and noise reduction algorithms for digital signal processing applications.

## Features

### Core Audio Processing
- Fast Fourier Transform (FFT/IFFT)
- Real-time frame processing
- Overlap-add synthesis
- Various window functions support

### Noise Reduction Algorithms
- **Spectral Subtraction**
  - Classic algorithm with oversubtraction factor
  - Spectral floor control
  - Multiple window function options

- **Wiener Filtering**
  - Adaptive noise estimation
  - Smoothing factor control
  - Prior SNR estimation

- **MMSE-based Methods**
  - MMSE STSA (Short-Time Spectral Amplitude)
  - Log-MMSE
  - Ephraim-Malah algorithm

- **Advanced Techniques**
  - Regularized Spectral Subtraction
  - Psychoacoustic Filtering
  - Subspace Enhancement
  - Kalman Filtering

### DSP Utilities
- Window Functions
  - Hanning
  - Hamming
  - Blackman
  - Kaiser
  - Rectangular

- Signal Analysis
  - SNR Calculation
  - PSNR Measurement
  - RMS Energy
  - Moving Average
  - Statistical Analysis

## Getting Started

### Prerequisites
```cpp
- C++17 compatible compiler
- CMake 3.14 or higher
- No external dependencies required
```

### Installation
```bash
git clone https://github.com/yourusername/speech_enhancement_dsp_cpp.git
cd speech_enhancement_dsp_cpp
mkdir build && cd build
cmake ..
make
```

### Basic Usage Example
```cpp
#include "dsp_speech_enhancement.h"
#include <vector>

// Initialize parameters
dsp::NoiseReduction::SpectralSubtractionParams params;
params.sampleRate = 16000;
params.frameSizeMs = 20;
params.overSubtractionFactor = 2.0f;

// Process audio
std::vector<float> noisySignal = loadAudioFile("noisy.wav");
auto enhancedSignal = dsp::NoiseReduction::spectralSubtraction(noisySignal, params);
```

## Performance Optimization

- Frame-based processing for memory efficiency
- SIMD-friendly data structures
- Optimized FFT implementation
- Efficient overlap-add synthesis

## API Documentation

### Main Classes
- `dsp::FFT` - FFT operations
- `dsp::DSPUtils` - Signal processing utilities
- `dsp::NoiseReduction` - Noise reduction algorithms

### Parameter Structures
```cpp
SpectralSubtractionParams
WienerFilterParams
MMSEParams
EphraimMalahParams
RegularizedSSParams
PsychoacousticParams
SubspaceParams
```

## Testing

Run the example file to test different algorithms:
```bash
./build/speech_enhancement_example
```

The example includes:
- Synthetic signal generation
- Multiple algorithm comparison
- Performance measurements
- SNR improvement analysis

## Contributing

1. Fork the repository
2. Create your feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Based on classic speech enhancement papers
- Inspired by MATLAB Audio Processing Toolbox
- Community contributions and feedback

## References

1. Ephraim, Y., & Malah, D. (1984). Speech enhancement using a minimum mean-square error short-time spectral amplitude estimator
2. Boll, S. (1979). Suppression of acoustic noise in speech using spectral subtraction
3. Wiener, N. (1949). Extrapolation, Interpolation, and Smoothing of Stationary Time Series
```