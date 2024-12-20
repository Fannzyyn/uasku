import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
import sys
import os

# Menambahkan direktori utama proyek ke sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Mengimpor fungsi main dari training.py
from modules.training import main  

def plot_confusion_matrix(y_valid, y_pred, class_names):
    # Membuat confusion matrix
    cm = confusion_matrix(y_valid, y_pred)
    
    # Menampilkan confusion matrix dalam bentuk heatmap
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prediksi')
    plt.ylabel('Aktual')
    plt.title('Confusion Matrix')
    plt.show()

def main_gabung():
    print("Proses Pelatihan dan Evaluasi Model...")
    # Jalankan fungsi dari training.py
    main()

    # Asumsikan kita mendapatkan hasil y_pred dan y_valid dari training.py
    # Pastikan Anda mendapatkan y_pred dan y_valid dari model yang dilatih
    # Simulasi hasil prediksi dan data validasi
    y_pred = [0, 1, 2]  # Gantilah dengan hasil prediksi yang benar
    y_valid = [0, 1, 2]  # Gantilah dengan data validasi yang benar

    class_names = [f'class{i+1}' for i in range(20)]  # Menentukan nama kelas

    # Plot confusion matrix
    plot_confusion_matrix(y_valid, y_pred, class_names)

if __name__ == "__main__":
    main_gabung()
