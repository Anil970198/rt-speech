import numpy as np
import pickle
import os

class StressClassifier:
    def __init__(self, model_path="models/svm_stress.pkl"):
        self.model_path = model_path
        self.model = None
        self.scaler = None
        
        # Try loading model
        if os.path.exists(model_path):
            try:
                with open(model_path, 'rb') as f:
                    data = pickle.load(f)
                    self.model = data['model']
                    self.scaler = data['scaler']
                print("Loaded SVM Model.")
            except Exception as e:
                print(f"Failed to load model: {e}")
        else:
            print("No pre-trained model found. Using Heuristic Mode.")
            
    def predict(self, features):
        """
        Predicts stress level (0.0 to 1.0) from feature dict.
        """
        if self.model and self.scaler:
            # Vectorize: MFCCs (13) + RMS + ZCR + F0 = 16 features
            vec = np.array(features['mfcc'] + [features['rms'], features['zcr'], features['f0']]).reshape(1, -1)
            vec_scaled = self.scaler.transform(vec)
            prob = self.model.predict_proba(vec_scaled)[0][1] # Probability of Class 1 (Stress)
            return prob
        else:
            # HEURISTIC FALLBACK (For testing without training)
            # High Pitch + High Energy = Stress
            score = 0.0
            
            # RMS (0.0 to 1.0 generally)
            if features['rms'] > 0.1: score += 0.3
            if features['rms'] > 0.3: score += 0.3
            
            # F0 (Pitch) Human speech 85-255Hz
            if features['f0'] > 200: score += 0.2
            if features['f0'] > 300: score += 0.2
            
            return min(score, 1.0)
