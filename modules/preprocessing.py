import os
import cv2
import numpy as np

def preprocess_image(image_path, size=(128, 128)):
    # Membaca gambar dalam format grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        return None  # Mengembalikan None jika gambar gagal dibaca
    # Mengubah ukuran gambar
    image = cv2.resize(image, size)
    # Normalisasi gambar
    image = image / 255.0
    return image

def load_dataset(folder_path, size=(128, 128)):
    X, y = [], []
    
    for label, subdir in enumerate(os.listdir(folder_path)):
        subdir_path = os.path.join(folder_path, subdir)
        for file in os.listdir(subdir_path):
            file_path = os.path.join(subdir_path, file)
            image = preprocess_image(file_path, size)
            if image is not None:
                X.append(image)
                y.append(label)

    return np.array(X), np.array(y)

def main():
    train_path = 'data/train'  # Ganti sesuai path folder dataset Anda
    print("Memuat dataset training...")
    X_train, y_train = load_dataset(train_path)
    
    print(f"Jumlah gambar training: {len(X_train)}")
    print(f"Jumlah label training: {len(y_train)}")

    # Anda dapat menambahkan lebih banyak verifikasi atau pengecekan di sini
    # Misalnya: Cek gambar pertama
    print(f"Ukuran gambar pertama: {X_train[0].shape}")
    
if __name__ == "__main__":
    main()
