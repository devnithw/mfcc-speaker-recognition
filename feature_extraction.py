import librosa
import numpy as np

NUM_MFCC = 40

def extract_features_MFCC(audio_file):
    """
    Extract MFCC features from an audio file and return numpy array with output shape=(TIME, MFCC).
    """
    # Load audio using Librosa
    waveform, sample_rate = librosa.load(audio_file, sr=None)

    # Check whether audio is mono channel otherwise convert to mono
    if len(waveform.shape) == 2:
        waveform = librosa.to_mono(waveform.transpose())

    # Resample to 16kHz
    if sample_rate != 16000:
        waveform = librosa.resample(waveform, sample_rate, 16000)

    # Get Mel Frequency Cepstral Coefficients
    features = librosa.feature.mfcc(y=waveform, sr=sample_rate, n_mfcc=NUM_MFCC)

    return features.transpose()