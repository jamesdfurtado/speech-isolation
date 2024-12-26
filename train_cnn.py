import os
import numpy as np
from keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

# Set directories
spectrograms_dir = r"C:\Users\james\Desktop\AudioSet\spectrogram\new"  # Update with your path
npy_dir = r"C:\Users\james\Desktop\AudioSet\array\new"  # Update with your path

# Load and prepare dataset
def load_data(spectrograms_dir, npy_dir):

    spectrograms = []   # Feature variable
    labels = []         # Target variable

    for file_name in os.listdir(spectrograms_dir):
        if file_name.endswith(".png"):
            spectrogram_path = os.path.join(spectrograms_dir, file_name)
            label_path = os.path.join(npy_dir, file_name.replace(".png", ".npy"))

            # Load the spectrogram image
            img = cv2.imread(spectrogram_path, cv2.IMREAD_GRAYSCALE)
            spectrograms.append(img)

            # Load the corresponding numpy label array
            labels.append(np.load(label_path))

    # Convert lists to numpy arrays
    spectrograms = np.array(spectrograms)
    labels = np.array(labels)

    # Normalize the images
    spectrograms = spectrograms.astype('float32') / 255.0

    return spectrograms, labels


# Load data
X, y = load_data(spectrograms_dir, npy_dir)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print('Train-test split done!')

# CNN Model
model = models.Sequential([
    layers.InputLayer(shape=(128, 128, 1)),  # Input layer (128x128 spectrogram)
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='sigmoid')  # Output layer (10 elements for speech detection)
])

print('Model initialized!')

# Compile the model
model.compile(optimizer='adam',
              loss='binary_crossentropy',  # Binary crossentropy loss for multi-label classification
              metrics=['accuracy'])

print('Model compiled!')

# FULL DATASET vvv -------------------------------------------------

print('Beginning model training...')

# Train the model on the entire dataset
history = model.fit(
    X_train,
    y_train,
    epochs=100,  # starting point - 50 epochs, hopefully not too many!
    validation_data=(X_test, y_test),
    batch_size=32
)

print('model training done!')

# FULL DATASET ^^^ -------------------------------------------------

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc}")

# Save the model
try:
  model.save(r"C:\Users\james\Desktop\Audio Denoising\model\real_model_v0.h5")
  print('Model saved successfully.')
except:
  print('Model was not saved.')

# QUICK RUNTIME------------------------------------------------------



