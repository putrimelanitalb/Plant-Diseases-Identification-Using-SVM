### IMPORT LIBRARY ###
import os
import cv2
import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

### FUNGSI BACA DATA GAMBAR ###
def read_data(data_path, categories):
    # List kosong untuk menyimpan data gambar
    data = []
    for category in categories:
        # Buat path ke setiap folder categories
        path = os.path.join(data_path, category)
        # Simpan category sebagai label untuk masing-masing categories
        label = category
        for img in os.listdir(path):
            # Buat path ke masing-masing gambar
            img_path = os.path.join(path, img)
            # Baca data gambar
            leaf_img = cv2.imread(img_path)
            # Resize gambar menjadi 100x100
            leaf_img = cv2.resize(leaf_img, (100,100))
            # Konversi gambar menjadi grayscale
            leaf_img = cv2.cvtColor(leaf_img, cv2.COLOR_BGR2GRAY)
            # Ubah gambar menjadi flatten array
            image = np.array(leaf_img).flatten()
            # Masukkan gambar dan labelnya dalam variabel data
            data.append([image, label])
    # Acak urutan data
    random.shuffle(data)
    # Buat list kosong untuk menyimpan gambar dan label secara terpisah
    images = []
    labels  = []
    # Pisahkan gambar dan label dari data
    for image, label in data:
        images.append(image)
        labels.append(label)
    return images, labels

### PATH FOLDER DATA ###
train_path = 'grape_datasets/train'
test_path  = 'grape_datasets/valid'
categories = ['black_rot', 'esca', 'healthy', 'leaf_blight']

### BACA DATA GAMBAR ###
print('[STATUS] Baca data train dan data test')
# Membaca data train
trainData, trainLabel = read_data(train_path, categories)
print('Data Train :', np.shape(trainData))
# Membaca data test
testData, testLabel = read_data(test_path, categories)
print('Data Test  :', np.shape(testData))

### TRAINING MODEL SVM ###
# Membuat model SVM
model = SVC(C=1, kernel='rbf', gamma='scale')
# Sesuaikan model SVM dengan data train
svm = model.fit(trainData, trainLabel)
print('\n[STATUS] Training model SVM selesai')

### PREDIKSI DATA TEST MENGGUNAKAN MODEL SVM ###
testPredict = svm.predict(testData)
print('\n[STATUS] Prediksi data test selesai')

### CLASSIFICATION REPORT ###
print('\nClassification Report :')
print(classification_report(testLabel, testPredict))

### CONFUSION MATRIX ###
# Tanpa normalisasi
print('\nConfusion Matrix tanpa Normalisasi :')
print(confusion_matrix(testLabel, testPredict, normalize = None))
# Dengan normalisasi
print('\nConfusion Matrix dengan Normalisasi :')
print(confusion_matrix(testLabel, testPredict, normalize = 'true'))

### SKOR TINGKAT AKURASI ###
print('\nAccuracy   :', accuracy_score(testLabel, testPredict))

### OUTPUT SALAH SATU DATA TEST ###
# Prediksi nama kelas
if categories[testPredict[0]] == 'black_rot':
    print('Prediction : Black Rot')
elif categories[testPredict[0]] == 'esca':
    print('Prediction : Esca (Black Measles)')
elif categories[testPredict[0]] == 'healthy':
    print('Prediction : Healthy')
elif categories[testPredict[0]] == 'leaf_blight':
    print('Prediction : Leaf Blight (Isariopsis Leaf Spot)')
# Tampilkan gambar
leaf = testData[0].reshape(100,100)
plt.imshow(leaf, cmap = 'gray')
plt.show()
