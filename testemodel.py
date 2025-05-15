import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import time
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import classification_report, confusion_matrix

# 1. Carregar e preparar os dados
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype("float32") / 255.0
x_test = x_test.astype("float32") / 255.0
y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# 2. Definir o modelo CNN
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(32,32,3)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 3. Treinamento com histórico
start_train = time.time()
history = model.fit(x_train, y_train_cat, epochs=30, batch_size=64, validation_split=0.1)
train_time = time.time() - start_train

# 4. Avaliação
start_pred = time.time()
y_pred_probs = model.predict(x_test)
pred_time = time.time() - start_pred
y_pred = np.argmax(y_pred_probs, axis=1)

# 5. Métricas
print("\n=== CNN ===")
print(f"Acurácia: {np.mean(y_pred == y_test.flatten()):.4f}")
print(f"Tempo de treino: {train_time:.2f}s")
print(f"Tempo de predição: {pred_time:.2f}s")
print(classification_report(y_test, y_pred))

# 6. Matriz de confusão
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10,8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Matriz de Confusão - CNN")
plt.xlabel("Previsto")
plt.ylabel("Real")
plt.savefig("matriz_confusao_cnn.png", dpi=300, bbox_inches='tight')
plt.show()

# 7. Gráfico de acurácia
plt.figure()
plt.plot(history.history['accuracy'], label='Treino')
plt.plot(history.history['val_accuracy'], label='Validação')
plt.title('Acurácia por Época')
plt.xlabel('Época')
plt.ylabel('Acurácia')
plt.legend()
plt.savefig("acuracia_por_epoca.png", dpi=300, bbox_inches='tight')
plt.show()

# 8. Gráfico de perda
plt.figure()
plt.plot(history.history['loss'], label='Treino')
plt.plot(history.history['val_loss'], label='Validação')
plt.title('Perda por Época')
plt.xlabel('Época')
plt.ylabel('Perda')
plt.legend()
plt.savefig("perda_por_epoca.png", dpi=300, bbox_inches='tight')
plt.show()