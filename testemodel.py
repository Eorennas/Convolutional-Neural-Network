import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

#  Forçar uso da GPU (se disponível)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
        print(" GPU detectada e configurada.")
    except RuntimeError as e:
        print(f"Erro ao configurar GPU: {e}")
else:
    print(" Nenhuma GPU detectada. Usando CPU.")

#  Função para carregar batches do CIFAR-10 local
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(batch[b'labels'])
        return images, labels

#  Caminho do dataset local já extraído
data_path = "cifar-10-python/cifar-10-batches-py"

# Carregar dados de treino
train_images, train_labels = [], []
for i in range(1, 6):
    images, labels = load_cifar_batch(os.path.join(data_path, f'data_batch_{i}'))
    train_images.append(images)
    train_labels.append(labels)
train_images = np.concatenate(train_images)
train_labels = np.concatenate(train_labels)

# Carregar dados de teste
test_images, test_labels = load_cifar_batch(os.path.join(data_path, 'test_batch'))

# Normalizar imagens
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Dividir treino em treino/validação
from sklearn.model_selection import train_test_split
train_images, val_images, train_labels, val_labels = train_test_split(
    train_images, train_labels, test_size=0.2, random_state=42
)

# Flatten labels
train_labels = train_labels.flatten()
val_labels = val_labels.flatten()
test_labels = test_labels.flatten()

#  Aumento de dados
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
    layers.RandomContrast(0.1),
])

# Construir modelo
def build_model():
    model = models.Sequential([
        layers.Input(shape=(32, 32, 3)),
        data_augmentation,

        layers.Conv2D(64, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.2),

        layers.Conv2D(128, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.3),

        layers.Conv2D(256, (3,3), activation='relu', padding='same',
                      kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Conv2D(256, (3,3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2,2)),
        layers.Dropout(0.4),

        layers.Flatten(),
        layers.Dense(256, activation='relu', kernel_regularizer=regularizers.l2(1e-4)),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(10, activation='softmax')
    ])
    return model

model = build_model()

#  Compilar modelo
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

#  Callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)

#  Treinar modelo
history = model.fit(train_images, train_labels,
                    epochs=50,  # Pode parar antes com early stopping
                    batch_size=64,
                    validation_data=(val_images, val_labels),
                    callbacks=[early_stopping, reduce_lr])

#  Avaliação final
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=0)
print(f"\n Acurácia final no conjunto de teste: {test_acc:.4f}")

#  Gráficos de treino
plt.figure(figsize=(12, 5))

# Acurácia
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia por Época')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.grid(True)

# Perda
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Loss por Época')
plt.xlabel('Época')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()

#  Mostrar resumo
model.summary()
