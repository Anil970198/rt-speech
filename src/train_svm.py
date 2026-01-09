import os
import glob
import numpy as np
import pickle
import librosa
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report
from tqdm import tqdm
from features import FeatureExtractor # Reuse our extractor

# Config
DATA_DIRS = [
    "../digital-wellbeing/data", 
    "../data", 
    "data"
]
MODEL_OUT = "models/svm_stress.pkl"

def load_data():
    """
    Scans dictionaries for RAVDESS data.
    Returns X (features) and y (labels: 0=Calm, 1=Stress).
    """
    valid_files = []
    for d in DATA_DIRS:
        if os.path.exists(d):
            print(f"Scanning {d}...")
            # RAVDESS structure: Actor_*/01-01-*-*-*-*-*.mp4
            files = glob.glob(os.path.join(d, "**/*.mp4"), recursive=True)
            files += glob.glob(os.path.join(d, "**/*.wav"), recursive=True)
            valid_files.extend(files)
            
    if not valid_files:
        print("ERROR: No data found. Please place RAVDESS data in 'data/' folder.")
        return None, None

    print(f"Found {len(valid_files)} files. Extracting features...")
    
    extractor = FeatureExtractor(rate=16000)
    X = []
    y = []
    
    for f in tqdm(valid_files):
        try:
            # Parse Filename
            # 03-01-01-01-01-01-01.mp4
            # Emotion is 3rd integer:
            # 01 = neutral, 02 = calm, 03 = happy, 04 = sad, 05 = angry, 06 = fearful, 07 = disgust, 08 = surprised
            basename = os.path.basename(f)
            parts = basename.split("-")
            if len(parts) < 3: continue
            
            emotion = int(parts[2])
            
            # Labeling Strategy
            # Calm/Happy/Neutral -> 0 (CALM)
            # Angry/Fear/Disgust/Sad -> 1 (STRESS)
            if emotion in [1, 2, 3]:
                label = 0
            elif emotion in [4, 5, 6, 7]:
                label = 1
            else:
                continue # Skip surprised
                
            # Extract Audio from Video/Audio file
            # We use librosa load which handles both
            y_audio, sr = librosa.load(f, sr=16000)
            
            # Extract Features
            # We treat the whole file as one "chunk" for training average
            feats = extractor.extract(y_audio)
            if feats is None: continue
            
            # Vectorize
            vec = feats['mfcc'] + [feats['rms'], feats['zcr'], feats['f0']]
            X.append(vec)
            y.append(label)
            
        except Exception as e:
            # print(f"Error processing {f}: {e}")
            pass
            
    return np.array(X), np.array(y)

def train():
    X, y = load_data()
    if X is None: return
    
    print(f"Dataset Size: {len(X)} samples.")
    print(f"Class Distribution: {np.bincount(y)}")
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Pipeline: Scale -> SVM
    # SVM is fast and effective for small feature vectors (16-dim)
    model = make_pipeline(StandardScaler(), SVC(probability=True, kernel='rbf'))
    
    print("Training SVM...")
    model.fit(X_train, y_train)
    
    print("Evaluating...")
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=["Calm", "Stress"]))
    
    # Save
    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, 'wb') as f:
        pickle.dump({
            'model': model['svc'],
            'scaler': model['standardscaler']
        }, f)
        
    print(f"Model saved to {MODEL_OUT}")

if __name__ == "__main__":
    train()
