import numpy as np
import pandas as pd
import os
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.applications import VGG16
import matplotlib.pyplot as plt

# Function to load data
def load_data(image_directory, csv_file, image_extension='.jpg', image_size=(150, 150)):
    df = pd.read_csv(csv_file)
    images = []
    labels = []
    for idx, row in df.iterrows():
        img_filename = os.path.join(image_directory, str(row['Image']) + image_extension)
        if os.path.exists(img_filename):
            img = Image.open(img_filename).convert('RGB')
            img = img.resize(image_size)
            img_array = np.array(img) / 255.0
            images.append(img_array)
            labels.append(row['Label'])
        else:
            print(f"Image {img_filename} not found.")
    images = np.array(images)
    labels = LabelEncoder().fit_transform(np.array(labels))
    return images, labels

# Update these paths to where your actual files are located
image_dir = 'C:/Users/Asus/OneDrive/Desktop/CNNProject/CNNProject/'
csv_file = 'C:/Users/Asus/OneDrive/Desktop/CNNProject/sorted_cat_data.csv'
images, labels = load_data(image_dir, csv_file)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

train_datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

base_model = VGG16(weights='imagenet', include_top=False, input_shape=X_train.shape[1:])

# Freeze the base model
base_model.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')  # For binary classification
])

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Note: Use train_datagen.flow() if your data is not pre-augmented
history = model.fit(train_datagen.flow(X_train, y_train, batch_size=32),
                    steps_per_epoch=len(X_train) / 32, epochs=50,
                    validation_data=(X_test, y_test))

# Plot training results
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc*100:.2f}%")

# Visualization function
def plot_predictions(images, true_labels, model, n=10):
    """
    Plots n randomly selected images from the test set along with
    the predicted and actual labels.
    """
    fig, axes = plt.subplots(nrows=n, ncols=1, figsize=(5, n*3))
    if n == 1:
        axes = [axes]
    for i, ax in enumerate(axes):
        # Select a random image
        idx = np.random.randint(0, images.shape[0])
        img = images[idx]
        true_label = true_labels[idx]

        # Predict the label
        img_array = np.expand_dims(img, axis=0)  # Model expects a batch of images
        prediction = model.predict(img_array)
        predicted_label = (prediction > 0.5).astype("int32")[0, 0]

        ax.imshow(img)
        ax.axis('off')
        ax.set_title(f"Actual: {true_label}, Predicted: {predicted_label}")

    plt.tight_layout()
    plt.show()

# Call the function with a subset of your test images and their true labels
plot_predictions(X_test, y_test, model, n=7)
