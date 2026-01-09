import numpy as np
import librosa

class FeatureExtractor:
    def __init__(self, rate=16000):
        self.rate = rate
        
    def extract(self, audio_buffer):
        """
        Extracts key biomarkers from a raw audio buffer.
        :param audio_buffer: 1D numpy array of float32 audio samples.
        :return: Discrete feature vector (dict) for dashboard and model.
        """
        if len(audio_buffer) == 0:
            return None
        
        # 1. RMS Energy (Loudness)
        rms = np.sqrt(np.mean(audio_buffer**2))
        
        # 2. Zero Crossing Rate (Roughness/Noise)
        zcr = ((audio_buffer[:-1] * audio_buffer[1:]) < 0).sum() / len(audio_buffer)
        
        # 3. MFCCs (Timbre/Spectral Envelope)
        # We need to reshape slightly for librosa or just pass 1D
        mfccs = librosa.feature.mfcc(y=audio_buffer, sr=self.rate, n_mfcc=13)
        mfcc_mean = np.mean(mfccs, axis=1) # Average over time -> (13,)
        
        # 4. Pitch / Fundamental Frequency (F0)
        # Piptrack is heavy, we use a lighter heuristic or just simple correlation if needed
        # For speed: simple FFT peak? No, speech is complex. We stick to ZCR for now as proxy for high freq.
        # But let's try a quick harmonic pitch estimate if possible.
        f0 = 0.0
        try:
            pitches, magnitudes = librosa.piptrack(y=audio_buffer, sr=self.rate, fmin=50, fmax=400)
            # Get max mag pitch
            idx = magnitudes.argmax()
            f0 = pitches.flat[idx]
        except:
            pass
            
        return {
            "rms": float(rms),
            "zcr": float(zcr),
            "f0": float(f0),
            "mfcc": mfcc_mean.tolist()
        }
