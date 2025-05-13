import os
import pickle
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

# Função para carregar um batch do CIFAR-10
def load_cifar_batch(filename):
    with open(filename, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')
        images = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        labels = np.array(batch[b'labels'])
        return images, labels

# Caminho correto para os dados já extraídos
data_path = "cifar-10-python/cifar-10-batches-py"
train_images, train_labels = [], []

# Carregando os 5 batches de treino
for i in range(1, 6):
    images, labels = load_cifar_batch(os.path.join(data_path, f'data_batch_{i}'))
    train_images.append(images)
    train_labels.append(labels)

train_images = np.concatenate(train_images)
train_labels = np.concatenate(train_labels)

# Carregando o batch de teste
test_images, test_labels = load_cifar_batch(os.path.join(data_path, 'test_batch'))

# Normalização dos dados
train_images = train_images.astype("float32") / 255.0
test_images = test_images.astype("float32") / 255.0

# Construção do modelo CNN
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compilação do modelo
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Treinamento do modelo
history = model.fit(train_images, train_labels, epochs=30,
                    validation_data=(test_images, test_labels))


# Avaliação do modelo
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
print(f"Acurácia no teste: {test_acc:.2f}")
