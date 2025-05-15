import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# Parâmetros
epochs_val = 30
batch_size_val = 64
imageDimensions = (32, 32, 3)
num_classes = 10

# Carregar dados
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Normalizar
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# One-hot encoding
y_train_cat = to_categorical(y_train, num_classes)
y_test_cat = to_categorical(y_test, num_classes)

# Dividir validação
from sklearn.model_selection import train_test_split
x_train, x_val, y_train_cat, y_val_cat = train_test_split(x_train, y_train_cat, test_size=0.2)

# Aumentar dados
dataGen = ImageDataGenerator(width_shift_range=0.1,
                             height_shift_range=0.1,
                             zoom_range=0.2,
                             shear_range=0.1,
                             rotation_range=10)
dataGen.fit(x_train)

# Modelo CNN mais robusto
def myModel():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=imageDimensions))
    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Treinar modelo
model = myModel()
print(model.summary())

start_train = time.time()
history = model.fit(dataGen.flow(x_train, y_train_cat, batch_size=batch_size_val),
                    validation_data=(x_val, y_val_cat),
                    epochs=epochs_val,
                    steps_per_epoch=len(x_train)//batch_size_val)
train_time = time.time() - start_train

# Avaliar
start_pred = time.time()
y_pred_probs = model.predict(x_test)
pred_time = time.time() - start_pred
y_pred = np.argmax(y_pred_probs, axis=1)

# Resultados
acc = np.mean(y_pred == y_test.flatten())
print(f"CNN Acurácia: {acc:.4f}")
print(f"Tempo treino: {train_time:.2f}s")
print(f"Tempo predição: {pred_time:.2f}s")
print(classification_report(y_test, y_pred))

# Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Greens')
plt.title("Matriz de Confusão - CNN")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.savefig("matriz_confusao_cnn.png", dpi=300, bbox_inches='tight')
plt.close()

# Gráficos de treino
plt.figure()
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia por Época - CNN')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.savefig("acuracia_cnn.png", dpi=300, bbox_inches='tight')
plt.close()

plt.figure()
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda por Época - CNN')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()
plt.savefig("perda_cnn.png", dpi=300, bbox_inches='tight')
plt.close()

# Salvar modelo
model.save('modelo_cifar10_cnn.h5')
print("Modelo salvo!")
