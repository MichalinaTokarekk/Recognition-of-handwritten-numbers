import tkinter as tk
from tkinter import Canvas
from keras import models, layers
from keras.models import load_model
from keras.datasets import mnist
from keras.utils import to_categorical
import numpy as np
from PIL import Image, ImageOps, ImageGrab
import matplotlib.pyplot as plt


# Załaduj i przygotuj dane MNIST
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28)).astype('float32') / 255
test_images = test_images.reshape((10000, 28 * 28)).astype('float32') / 255
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Zbuduj i skompiluj model sieci neuronowej
network = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(28 * 28,)),
    layers.Dense(10, activation='softmax')
])
network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

# Trenuj model
x_val = train_images[:10000]
partial_x_train = train_images[10000:]
y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]
history = network.fit(partial_x_train, partial_y_train, epochs=5, batch_size=128, validation_data=(x_val, y_val))
network.save('mnist_network_saved.h5')

# Funkcja do przewidywania cyfry z rysunku
def predict_digit(img):
    img = img.resize((28, 28), Image.LANCZOS)
    img = ImageOps.grayscale(img)
    img_array = np.array(img).reshape(1, 784).astype('float32') / 255
    result = network.predict(img_array)
    return np.argmax(result), np.max(result)

# Aplikacja tkinter do rysowania i klasyfikacji
class ImageDraw(tk.Frame):
    def __init__(self, parent=None):
        tk.Frame.__init__(self, parent)
        self.parent = parent
        self.setUI()

    def setUI(self):
        self.parent.title("Rysuj cyfrę")
        self.pack(fill=tk.BOTH, expand=1)
        self.canvas = Canvas(self, bg='black')
        self.canvas.pack(fill=tk.BOTH, expand=1)
        self.canvas.bind("<B1-Motion>", self.draw)

        clear_button = tk.Button(self, text="Wyczyść", command=self.clear)
        classify_button = tk.Button(self, text="Klasyfikuj", command=self.classify)
        clear_button.pack(side=tk.RIGHT, padx=5, pady=5)
        classify_button.pack(side=tk.RIGHT, padx=5, pady=5)

    def draw(self, event):
        self.canvas.create_oval(event.x - 8, event.y - 8, event.x + 8, event.y + 8, fill='white', outline='white')

    def clear(self):
        self.canvas.delete("all")

    def classify(self):
        self.canvas.update()
        x = self.parent.winfo_rootx() + self.canvas.winfo_x()
        y = self.parent.winfo_rooty() + self.canvas.winfo_y()
        x1 = x + self.canvas.winfo_width()
        y1 = y + self.canvas.winfo_height()
        img = ImageGrab.grab().crop((x, y, x1, y1))
        digit, confidence = predict_digit(img)
        print(f"Predicted Digit: {digit}, Confidence: {confidence:.2%}")

# Uruchomienie głównej aplikacji
def main():
    root = tk.Tk()
    app = ImageDraw(parent=root)
    root.geometry("280x330")
    root.mainloop()

# Wyświetlanie wyników trenowania i walidacji na wykresach
plt.figure()
plt.plot(history.history['loss'], 'bo', label='Strata trenowania')
plt.plot(history.history['val_loss'], 'b', label='Strata walidacji')
plt.title('Strata trenowania i walidacji')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()

plt.figure()
plt.plot(history.history['accuracy'], 'bo', label='Dokładność trenowania')
plt.plot(history.history['val_accuracy'], 'b', label='Dokładność walidacji')
plt.title('Dokładność trenowania i walidacji')
plt.xlabel('Epoki')
plt.ylabel('Dokładność')
plt.show()

#wyswietlammy nasze predykcje
def display_text_predictions(images, labels, predictions):
    num_images = len(predictions)
    for i in range(num_images):
        true_label = np.argmax(labels[i])

        predicted_label = np.argmax(predictions[i])
        print(f'True: {true_label}\tPred: {predicted_label}')
        print(predictions[i])
predictions = network.predict(test_images[:100])
display_text_predictions(test_images, test_labels, predictions)

#wyswietlanie kilka obrazkow oraz sprawdzenie ich poprawnosci
test_images_reshaped = test_images[:100].reshape((100, 28, 28)) #do wyswietlenia musimy przywrocic forme macierzy 28*27
plt.figure(figsize=(10, 10))  #ustawiamy rozmiar wyswietlanego obrazka
for i in range(25):  #dobor "i" musi odpowiadac subplotowi ponizej
    plt.subplot(5, 5, i + 1)  # 5x5 = 25 = i
    plt.imshow(test_images_reshaped[i], cmap='gray') #wyswietlanie obrazka
    plt.text(0, 31, f'Real: {np.argmax(test_labels[i])}, Pred: {np.argmax(predictions[i])}', color='red')
    plt.xticks([]) #schowanie podzialki poziomej
    plt.yticks([])

plt.show()
plt.close()

if __name__ == '__main__':
    main()