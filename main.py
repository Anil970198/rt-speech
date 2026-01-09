import time
import sys
import os
from src.stream import AudioStreamer
from src.features import FeatureExtractor
from src.predict import StressClassifier
import numpy as np

def draw_dashboard(stress_score, features):
    """
    Renders a live ASCII dashboard.
    """
    # clear screen
    os.system('cls' if os.name == 'nt' else 'clear')
    
    print("=== REAL-TIME SPEECH ASSESSMENT (rt-speech) ===")
    print("Press Ctrl+C to Exit\n")
    
    # Stress Bar
    bar_len = 20
    filled = int(stress_score * bar_len)
    bar = "|" * filled + "." * (bar_len - filled)
    
    status = "CALM"
    color = "\033[92m" # Green
    if stress_score > 0.4: 
        status = "ALERT"
        color = "\033[93m" # Yellow
    if stress_score > 0.7:
        status = "STRESS"
        color = "\033[91m" # Red
    reset = "\033[0m"
    
    print(f"Stress Level: [{color}{bar}{reset}] {stress_score*100:.1f}%  {color}{status}{reset}")
    print("-" * 40)
    
    # Feature Stats
    if features:
        print(f"RMS Energy:  {features['rms']:.4f}")
        print(f"Pitch (F0):  {features['f0']:.1f} Hz")
        print(f"ZeroCross:   {features['zcr']:.4f}")
    
def main():
    rate = 16000
    chunk = int(rate * 0.5) # 0.5s chunks for faster updates
    
    streamer = AudioStreamer(rate=rate, chunk=chunk)
    extractor = FeatureExtractor(rate=rate)
    classifier = StressClassifier() # Uses Heuristic default
    
    print("Initializing Stream...")
    streamer.start()
    
    try:
        while True:
            audio_frame = streamer.get_frame()
            if audio_frame is not None:
                # 1. Extract
                if np.max(np.abs(audio_frame)) < 0.01:
                    # Silence detection
                    features = None
                    score = 0.0
                else:
                    features = extractor.extract(audio_frame)
                    # 2. Predict
                    score = classifier.predict(features)
                    
                # 3. Visual
                draw_dashboard(score, features)
                
            time.sleep(0.1) # UI Refresh rate
            
    except KeyboardInterrupt:
        print("\nStopping...")
    finally:
        streamer.stop()

if __name__ == "__main__":
    main()
