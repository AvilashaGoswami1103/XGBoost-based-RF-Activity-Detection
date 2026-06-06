# XGBoost-based-RF-Activity-Detection
## 1. preprocess.ipynp: IQ Signal Preprocessing Pipeline
Implements a high-performance preprocessing pipeline for Radio Frequency (RF) In-phase and Quadrature (IQ) data. Prepares raw RF recordings for downstream machine learning and signal analysis tasks.
### Key Features:
● Automatic detection of binary and CSV IQ file formats.
● Support for large-scale RF datasets through chunk-based processing.
● GPU-accelerated DC offset removal.
● 5th-order Butterworth bandpass filtering.
● Stateful SOS filtering for continuous signal processing across chunks.
● RMS-based signal power normalization.
● Multi-threaded batch processing of multiple files.
● Automatic GPU detection and memory management.
● Conversion of raw IQ recordings into NumPy-ready datasets for machine learning.
### Processing Pipeline:
Each IQ recording undergoes the folloqing preprocessing steps:
1.1 Data Loading: pipeline loads RF Recordings from supported files formats: .iq/.bin/.dat/.csv -> Binary files are interpreted as interleaved IQ samples: I0, Q0, I1, Q1, I2, Q2, ... -> CSV files are automatically parsed into separate I and Q channels.
1.2 IQ De-interleaving: Raw RF samples are separated into In-phase(I) and Quadrature(Q) components for independent signal processing.
1.3 DC Offset Removal: A DC offset removal stage is applied independently to the I and Q channels -> removes receiver bias and centers the signal around zero. Implemented using GPU acceleration via CuPy.
1.4 Butterworth Bandpass Filtering: Butterworth bandpass filter designed using Scipy -> suppresses low-frequency drift and out-of-band noise while preserving the useful RF spectrum.
Filter Configuration:
| Parameter    | Value   |
| ------------ | ------- |
| Sample Rate  | 100 MHz |
| Low Cutoff   | 100 kHz |
| High Cutoff  | 45 MHz  |
| Filter Order | 5       |

1.5 Stateful SOS Filtering: Filtering performed using Second-order Sections(SOS): scipy.signal.sosfilt()
The filter state is preserved between chunks, ensuring Continuous filtering, No boundary discontinuities and Stable processing of large recordings.
1.6 RMS Power Normalization: Signal amplitudes are normalized using Root Mean Square (RMS) power -> Normalised Inorm and Qnorm computed -> standardizes signal energy across recordings and improves machine learning performance.
1.7 Processed Data Generation: The processed IQ data is stored as NumPy arrays: shape = (2, N) -> row 0: I channel, row 1: Q channel -> Output files are saved as: *_processed.npy. 
