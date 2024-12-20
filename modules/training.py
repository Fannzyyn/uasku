from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from modules.preprocessing import load_dataset
import joblib
import numpy as np

def flatten_images(X):
    return X.reshape(X.shape[0], -1)  # Flatten gambar menjadi vektor 1D

def train_model(X_train, y_train):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def main():
    train_path = 'data/train'  # Ganti sesuai path folder dataset Anda
    validation_path = 'data/validation'  # Ganti sesuai path folder validasi

    # Memuat dataset
    print("Memuat dataset training...")
    X_train, y_train = load_dataset(train_path)
    print("Memuat dataset validasi...")
    X_valid, y_valid = load_dataset(validation_path)

    # Flatten data untuk digunakan oleh Random Forest
    X_train_flat = flatten_images(X_train)
    X_valid_flat = flatten_images(X_valid)

    # Melatih model
    print("Melatih model Random Forest...")
    model = train_model(X_train_flat, y_train)

    # Mengukur performa model pada data validasi
    y_pred = model.predict(X_valid_flat)
    accuracy = accuracy_score(y_valid, y_pred)
    print(f"Akurasi pada data validasi: {accuracy * 100:.2f}%")

    print("\nLaporan Klasifikasi:")
    print(classification_report(y_valid, y_pred))

    # Menyimpan model
    joblib.dump(model, 'random_forest_signature_model.pkl')
    print("Model berhasil disimpan sebagai 'random_forest_signature_model.pkl'")

if __name__ == "__main__":
    main()
