import librosa
import numpy as np

NUM_MFCC = 40

def extract_features_MFCC(audio_file):
    """Extract MFCC features from an audio file, shape=(TIME, MFCC)."""
    waveform, sample_rate = librosa.load(audio_file, sr=None)

    # Convert to mono channel
    if len(waveform.shape) == 2:
        waveform = librosa.to_mono(waveform.transpose())

    # Resample to 16kHz
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, sample_rate, 16000)

    # Mel-frequency cepstral coefficients (MFCCs) are robust to noise bcoz of logarithmic compression
    features = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=NUM_MFCC)
    # the shape of features will be 40 X 441, where 40 represent featues where as 441 represent frames

    return features.transpose()