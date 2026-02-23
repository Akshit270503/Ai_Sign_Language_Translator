"""
Model Training Script
======================
Train a Neural Network classifier on collected gesture data.
Usage: python train_model.py
Requires: dataset/gesture_data.csv (run collect_data.py first)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import os

def train_model():
    # ── Load Data ──────────────────────────────────────────────────────────────
    data_path = "dataset/gesture_data.csv"
    if not os.path.exists(data_path):
        print("❌ No dataset found! Run collect_data.py first.")
        return

    print("📂 Loading dataset...")
    df = pd.read_csv(data_path)
    print(f"   Total samples: {len(df)}")
    print(f"   Classes: {sorted(df['label'].unique())}")
    print(f"   Samples per class:\n{df['label'].value_counts().to_string()}\n")

    X = df.drop('label', axis=1).values
    y = df['label'].values

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y_enc, test_size=0.2, random_state=42, stratify=y_enc)

    # ── Try TensorFlow first, fallback to sklearn ───────────────────────────────
    try:
        import tensorflow as tf
        from tensorflow import keras

        print("🧠 Training Neural Network (TensorFlow)...")
        model = keras.Sequential([
            keras.layers.Input(shape=(63,)),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(128, activation='relu'),
            keras.layers.BatchNormalization(),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(64, activation='relu'),
            keras.layers.Dense(len(le.classes_), activation='softmax')
        ])

        model.compile(
            optimizer='adam',
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )

        history = model.fit(
            X_train, y_train,
            epochs=50,
            batch_size=32,
            validation_data=(X_test, y_test),
            verbose=1,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5)
            ]
        )

        _, acc = model.evaluate(X_test, y_test, verbose=0)
        print(f"\n✅ Test Accuracy: {acc*100:.2f}%")

        # Save model
        os.makedirs("model", exist_ok=True)
        model.save("model/gesture_model.h5")
        np.save("model/label_classes.npy", le.classes_)
        print("💾 Model saved to model/gesture_model.h5")
        print("💾 Labels saved to model/label_classes.npy")

    except ImportError:
        # Fallback: Random Forest with sklearn
        from sklearn.ensemble import RandomForestClassifier
        import pickle

        print("🌲 Training Random Forest (sklearn)...")
        clf = RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42, n_jobs=-1)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        acc = (y_pred == y_test).mean()
        print(f"\n✅ Test Accuracy: {acc*100:.2f}%")
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, target_names=le.classes_))

        os.makedirs("model", exist_ok=True)
        with open("model/gesture_model.pkl", 'wb') as f:
            pickle.dump(clf, f)
        np.save("model/label_classes.npy", le.classes_)
        print("💾 Model saved to model/gesture_model.pkl")

    print("\n🚀 Training complete! Your model is ready.")
    print("   The app.py will automatically use this model if present.")

if __name__ == '__main__':
    train_model()
