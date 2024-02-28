
# CNN Project: Cat Breed Classification

This project focuses on building a Convolutional Neural Network (CNN) to classify cat breeds using TensorFlow and Keras. The dataset required for this project can be found at the link: 'x'.

## Project Overview

- **Objective**: To classify images of cats into specific breeds.
- **Tools Used**: TensorFlow, Keras, Pandas, NumPy, scikit-learn, Matplotlib, PIL.

## Data Preparation

The data is loaded and prepared using a custom function `load_data`, which reads images and labels from a specified directory and CSV file, respectively. The images are resized to 150x150 pixels and normalized.

### Data Augmentation

To enhance the dataset and prevent overfitting, data augmentation techniques such as rotation, width and height shift, shear, zoom, and horizontal flip are applied.

## Model Architecture

The model uses the VGG16 architecture as a base model with the following modifications:
- The base model's top layers are excluded, and it is set to non-trainable to utilize its pre-trained weights for feature extraction.
- A global average pooling layer is added to reduce the dimensionality.
- A fully connected dense layer with 256 units and ReLU activation is included.
- A dropout layer with a rate of 0.5 is used to reduce overfitting.
- The output layer consists of a single neuron with a sigmoid activation function for binary classification.

## Training

The model is compiled with the Adam optimizer and binary cross-entropy loss function. It is trained using the augmented data for 50 epochs, with a batch size of 32.

## Evaluation and Visualization

The model's performance is evaluated on a test set, and accuracy is reported. Additionally, a custom function `plot_predictions` is provided to visualize the model's predictions alongside the actual labels for a selection of test images.

## Usage

1. Ensure the dataset is downloaded from the provided link and stored in the specified directory.
2. Update the `image_dir` and `csv_file` paths to match the location of your dataset.
3. Run the script to train the model and evaluate its performance.

## Dependencies

- Python
- TensorFlow
- Keras
- Pandas
- NumPy
- scikit-learn
- Matplotlib
- PIL

This project demonstrates the application of transfer learning and data augmentation in building an effective CNN for image classification tasks.
