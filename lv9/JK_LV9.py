import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Input
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping
from tensorflow.keras.preprocessing import image
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import datetime
import os
import seaborn as sns

# Parameters
batch_size = 32
img_size = (48, 48)
epochs = 15

# Load datasets
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'gtsrb/Train',
    labels='inferred',
    label_mode='categorical',
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset='training',
    seed=123
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'gtsrb/Train',
    labels='inferred',
    label_mode='categorical',
    image_size=img_size,
    batch_size=batch_size,
    validation_split=0.2,
    subset='validation',
    seed=123
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    'gtsrb/Test',
    labels='inferred',
    label_mode='categorical',
    image_size=img_size,
    batch_size=batch_size
)

# Normalizacija podataka
normalization_layer = tf.keras.layers.Rescaling(1./255)
train_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
val_ds = val_ds.map(lambda x, y: (normalization_layer(x), y))
test_ds = test_ds.map(lambda x, y: (normalization_layer(x), y))

# augmentacija podataka
data_augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.1),
    tf.keras.layers.RandomZoom(0.1),
])
train_ds = train_ds.map(lambda x, y: (data_augmentation(x, training=True), y))

# definicija  modela
model = Sequential([
    Input(shape=(48, 48, 3)),
    Conv2D(32, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.4),
    Dense(43, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# logovi
log_dir = "logs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
checkpoint_cb = ModelCheckpoint('best_model.keras', save_best_only=True)
tensorboard_cb = TensorBoard(log_dir=log_dir, histogram_freq=1)
early_stop_cb = EarlyStopping(patience=3, restore_best_weights=True)

# treniranje model
# model.fit(
#     train_ds,
#     validation_data=val_ds,
#     epochs=epochs,
#     callbacks=[checkpoint_cb, tensorboard_cb, early_stop_cb]
# )

# evaulacija i matrica zabune
model.load_weights('best_model.keras')
loss, acc = model.evaluate(test_ds)
print(f"[Test Accuracy: {acc:.2f}]")

# predikcija
# y_pred = best_model.predict(x_test_s)
# y_pred_classes = np.argmax(y_pred, axis=1)
# y_true = np.argmax(y_test_s, axis=1)

# cm = confusion_matrix(y_true, y_pred_classes)
# plt.figure(figsize=(10, 8))
# sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
# plt.title('Matrica zabune - Testni skup')
# plt.xlabel('Predviđeno')
# plt.ylabel('Stvarno')
# plt.show()

# y_test_s = to_categorical(y_test, num_classes=43)

# y_pred_probs = model.predict(test_ds)
# y_pred = np.argmax(y_pred_probs, axis=1)
# y_true_labels = np.argmax(y_true_s, axis=1)

predictions = np.array([])
labels =  np.array([])
for x, y in test_ds:
  predictions = np.concatenate([predictions, np.argmax(model.predict(x), axis = -1)])
  labels = np.concatenate([labels, np.argmax(y.numpy(), axis=-1)])

cm = confusion_matrix(labels, predictions)

sns.heatmap(cm, annot=True)
# disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=test_ds.class_names)
# disp.plot(cmap=plt.cm.Blues, xticks_rotation=90)
plt.title("Matrica zabune")
plt.tight_layout()
plt.show()

# Single image classification
try:
    img_path = 'my_sign.jpg'
    img = image.load_img(img_path, target_size=img_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    print(f"Predviđena klasa: {predicted_class} ({test_ds.class_names[predicted_class]})")
except FileNotFoundError:
    print("Slika nije pronađena.")
