# Fashion MNIST Veri Seti ile Görüntü Sınıflandırma
# Öğrenci Adı: Oğuzhan Kuzlukluoğlu
# Öğrenci Numarası: 20253022


import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# Fashion MNIST veri setini yükleme
(x_train, y_train), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Veri setini inceleme
print("Eğitim verisi boyutu:", x_train.shape)
print("Test verisi boyutu:", x_test.shape)

# Sınıf isimlerini tanımlama
class_names = ['Tişört/Üst', 'Pantolon', 'Kazak', 'Elbise', 'Ceket',
               'Sandalet', 'Gömlek', 'Spor Ayakkabı', 'Çanta', 'Ayak Bileği Botu']

# Örnek görüntüleri gösterme
plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_train[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[y_train[i]])
plt.savefig('fashion_mnist_ornekler.png')
plt.close()

# Veriyi önişleme
# Piksel değerlerini [0-1] aralığına normalize etme
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# CNN için veri boyutunu yeniden şekillendirme (28x28x1)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Etiketleri kategorik formata dönüştürme (one-hot encoding)
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)

# CNN modelini oluşturma
model = Sequential([
    # İlk evrişim katmanı
    Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    MaxPooling2D((2, 2)),
    
    # İkinci evrişim katmanı
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    
    # Üçüncü evrişim katmanı
    Conv2D(64, (3, 3), activation='relu'),
    
    # Düzleştirme
    Flatten(),
    
    # Tam bağlantılı katmanlar
    Dense(128, activation='relu'),
    Dropout(0.5),  # Aşırı uyumu önlemek için dropout
    Dense(10, activation='softmax')  # 10 sınıf için çıkış katmanı
])

# Modeli derleme
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Model özetini görüntüleme
model.summary()

# Modeli eğitme
history = model.fit(x_train, y_train, 
                    epochs=10, 
                    batch_size=64,
                    validation_split=0.2)

# Eğitim sürecini görselleştirme
plt.figure(figsize=(12, 4))

# Doğruluk grafiği
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Eğitim Doğruluğu')
plt.plot(history.history['val_accuracy'], label='Doğrulama Doğruluğu')
plt.xlabel('Epoch')
plt.ylabel('Doğruluk')
plt.ylim([0.5, 1])
plt.legend(loc='lower right')
plt.title('Eğitim ve Doğrulama Doğruluğu')

# Kayıp grafiği
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Eğitim Kaybı')
plt.plot(history.history['val_loss'], label='Doğrulama Kaybı')
plt.xlabel('Epoch')
plt.ylabel('Kayıp')
plt.legend(loc='upper right')
plt.title('Eğitim ve Doğrulama Kaybı')

plt.savefig('egitim_sureci.png')
plt.close()

# Test veri seti üzerinde değerlendirme
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print('\nTest doğruluğu:', test_acc)

# Sınıflandırma sonuçlarını inceleme
y_pred = model.predict(x_test)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_test, axis=1)

# Confusion Matrix
conf_mat = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(12, 10))
sns.heatmap(conf_mat, annot=True, fmt='d', cmap=plt.cm.Blues,
            xticklabels=class_names, yticklabels=class_names)
plt.xlabel('Tahmin Edilen Sınıf')
plt.ylabel('Gerçek Sınıf')
plt.title('Karışıklık Matrisi')
plt.savefig('karisiklik_matrisi.png')
plt.close()

# Sınıflandırma Raporu
print("Sınıflandırma Raporu:\n")
print(classification_report(y_true, y_pred_classes, target_names=class_names))

# Hatalı tahminleri görselleştirme
misclassified_idx = np.where(y_pred_classes != y_true)[0]

n_to_show = min(25, len(misclassified_idx))
plt.figure(figsize=(12, 12))

for i, idx in enumerate(np.random.choice(misclassified_idx, n_to_show, replace=False)):
    plt.subplot(5, 5, i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(x_test[idx].reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f'Gerçek: {class_names[y_true[idx]]}\nTahmin: {class_names[y_pred_classes[idx]]}',
              color='red', fontsize=8)
plt.tight_layout()
plt.savefig('hatali_tahminler.png')
plt.close()

# Modeli kaydetme
model.save('fashion_mnist_model.h5')
print("Model kaydedildi: fashion_mnist_model.h5")

# Örnek tahmin yapma
def make_prediction(image_index):
    # Görüntüyü al
    img = x_test[image_index].reshape(1, 28, 28, 1)
    
    # Tahmin yap
    prediction = model.predict(img)
    predicted_class = np.argmax(prediction)
    
    # Gerçek sınıfı al
    true_class = y_true[image_index]
    
    # Görüntüyü göster
    plt.figure(figsize=(6, 3))
    plt.subplot(1, 2, 1)
    plt.imshow(x_test[image_index].reshape(28, 28), cmap=plt.cm.binary)
    plt.title(f'Gerçek: {class_names[true_class]}')
    
    # Tahmin olasılıklarını göster
    plt.subplot(1, 2, 2)
    plt.barh(range(10), prediction[0])
    plt.yticks(range(10), class_names)
    plt.title(f'Tahmin: {class_names[predicted_class]}')
    
    plt.tight_layout()
    return plt

# Rastgele bir görüntü üzerinde tahmin yapma
random_image_index = np.random.randint(0, len(x_test))
prediction_plot = make_prediction(random_image_index)
prediction_plot.savefig('ornek_tahmin.png')
prediction_plot.close()

print("Tüm işlemler tamamlandı. Oluşturulan dosyalar:")
print("- fashion_mnist_ornekler.png")
print("- egitim_sureci.png")
print("- karisiklik_matrisi.png")
print("- hatali_tahminler.png")
print("- ornek_tahmin.png")
print("- fashion_mnist_model.h5")