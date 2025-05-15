import os
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
from collections import Counter

(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

x_train, x_test = x_train / 255.0, x_test / 255.0

class_names = ['avião', 'automóvel', 'pássaro', 'gato', 'cervo',
               'cachorro', 'sapo', 'cavalo', 'navio', 'caminhão']

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

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

history = model.fit(x_train, y_train, epochs=20,
                    validation_data=(x_test, y_test),
                    verbose=2)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f'\n✅ Acurácia final no conjunto de teste: {test_acc:.4f}')

def predict_and_save(img_path, idx):
    img = cv2.imread(img_path)
    if img is None:
        print(f"❌ Erro ao carregar imagem: {img_path}")
        return None

    img_resized = cv2.resize(img, (32, 32))
    img_normalized = img_resized / 255.0
    img_batch = np.expand_dims(img_normalized, axis=0)

    predictions = model.predict(img_batch, verbose=0)
    class_index = np.argmax(predictions)
    confidence = np.max(predictions)
    if confidence > 0.9 and class_index:
        resultado = class_names[class_index]
    else:
        resultado = "não identificado"

    print(f"\n📷 {os.path.basename(img_path)}: {resultado} ({confidence * 100:.2f}%)")

    # Salvar imagem com título
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(f"{resultado} ({confidence*100:.1f}%)")
    plt.axis('off')
    plt.savefig(f"resultados/resultado_{idx}.png")
    plt.clf()

    return class_index

os.makedirs("resultados", exist_ok=True)

pasta = "data"
resultados = []

if os.path.exists(pasta):
    arquivos = sorted([f for f in os.listdir(pasta) if f.lower().endswith(('.jpg', '.png', '.jpeg'))])
    for idx, nome in enumerate(arquivos):
        caminho = os.path.join(pasta, nome)
        pred = predict_and_save(caminho, idx)
        if pred is not None:
            resultados.append(pred)
else:
    print(f"⚠️ Pasta '{pasta}' não encontrada.")

# ---------- Análise ----------
if resultados:
    total = len(resultados)
    contagem = Counter(resultados)
    print("\n📊 Resultados finais:")
    for classe_idx, qtd in contagem.items():
        print(f"🔹 {class_names[classe_idx]}: {qtd} imagem(ns) ({(qtd / total) * 100:.1f}%)")
    print(f"\nTotal: {total} imagens classificadas.")
else:
    print("⚠️ Nenhuma imagem foi classificada.")
