import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
import seaborn as sns

(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Vrijednost izmed 0 i 1
x_train = x_train.astype('float32') / 255
x_test = x_test.astype('float32') / 255

x_train = x_train.reshape((x_train.shape[0], 784))
x_test = x_test.reshape((x_test.shape[0], 784))

y_train_cat = to_categorical(y_train, 10)
y_test_cat = to_categorical(y_test, 10)

# printanje nekih primjera iz mnist dataseta
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i].reshape(28, 28), cmap='gray')
    plt.title(f"Label: {y_train[i]}")
    plt.axis('off')
plt.suptitle('Sample images from MNIST dataset')
plt.tight_layout()
plt.show()

# inicijalizacija modela
model = Sequential()
model.add(Dense(units=100, activation='relu', input_shape=(784,)))  
model.add(Dense(units=50, activation='relu'))  
model.add(Dense(units=10, activation='softmax'))  

model.summary()

model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

history = model.fit(x_train, y_train_cat,
                    epochs=20,
                    batch_size=32,
                    validation_split=0.2)

# Evaluacija
train_loss, train_acc = model.evaluate(x_train, y_train_cat, verbose=0)
test_loss, test_acc = model.evaluate(x_test, y_test_cat, verbose=0)

print(f"\nTraining accuracy: {train_acc:.4f}")
print(f"Test accuracy: {test_acc:.4f}")

# matrica zabune
def plot_confusion_matrix(y_true, y_pred, title):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(title)
    plt.xlabel('Predicted label')
    plt.ylabel('True label')
    plt.show()

# testiranje
y_train_pred = model.predict(x_train).argmax(axis=1)
y_test_pred = model.predict(x_test).argmax(axis=1)

plot_confusion_matrix(y_train, y_train_pred, 'Confusion Matrix - Training Set')
plot_confusion_matrix(y_test, y_test_pred, 'Confusion Matrix - Test Set')

# printanje statisike
# print("\nResults Analysis:")
# print("The neural network achieved good accuracy on both training and test sets.")
# print("The confusion matrices show that most digits are classified correctly.")
# print("The similar performance on training and test sets suggests the model is not overfitting.")
# print("Some common misclassifications might occur between similar-looking digits (e.g., 4 vs 9, 3 vs 8).")