import keras
from keras.datasets import mnist
import numpy as np
from keras import models
from keras import layers
from keras.utils import to_categorical
import matplotlib.pyplot as plt

(train_images, train_labels), (test_images, test_labels) = mnist.load_data() #ładujemy nasze cyferki do rozpoznania

print(train_images.shape)    #tutaj sprawdzamy wymiary zbioru
print(len(train_labels))

print(test_images.shape)
print(len(test_labels))

train_images = train_images.reshape((60000, 28 * 28)) #tutaj przerabiamy sobie nasze dane aby byly latwiej przyjete przez siec, po prostu listujemy sobie wartosic macierz (28,28) jako liste (28*28)
train_images = train_images.astype('float32') / 255 # tutaj wartosci zawarte w macierzy mają wartość 0-255, aby lepiej si3c to przetworzyla to bierzemy sobie te wartosci od 0 do 1
test_images = test_images.reshape((10000, 28 * 28)) #powtarzamy proces
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels) #tutaj była funkcja one_hot ale znalazłem to szybsze rozwiązanie
test_labels = to_categorical(test_labels)

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

network.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

x_val = train_images[:10000]
partial_x_train = train_images[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]


history = network.fit(partial_x_train, partial_y_train, epochs=5, batch_size=128, validation_data=(x_val, y_val))


#network.save('mnist_network_saved.h5')

test_loss, test_acc = network.evaluate(test_images, test_labels)

print('test_acc:', test_acc)
print('test_loss:', test_loss)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1, len(loss) + 1)

plt.figure()
plt.plot(epochs, loss, 'bo', label='Strata trenowania')
plt.plot(epochs, val_loss, 'b', label='Strata walidacji')
plt.title('Strata trenowania i walidacji')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()


plt.figure()
plt.plot(epochs, acc, 'bo', label='Dokladnosc trenowania')
plt.plot(epochs, val_acc, 'b', label='Dokladnosc walidacji')
plt.title('Dokladnosc trenowania i walidacji')
plt.xlabel('Epoki')
plt.ylabel('Strata')
plt.legend()




#wyswietlammy nasze predykcje
def display_text_predictions(images, labels, predictions):
    num_images = len(predictions)
    for i in range(num_images):
        true_label = np.argmax(labels[i])
        predicted_label = np.argmax(predictions[i])
        print(f'True: {true_label}\tPred: {predicted_label}')

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







