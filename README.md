# Convolutional-Neural-Network

Convolutional Neural Network

# Project Overview

In this project, we will build a convolutional neural network (CNN) to classify images of cats and dogs. The project includes the following steps:

1. **Training the Model**:
   - The CNN will be trained on a dataset consisting of:
     - 4000 images of dogs
     - 4000 images of cats

2. **Testing the Model**:
   - After training, the model will be evaluated on a test dataset that includes:
     - 1000 images of dogs
     - 1000 images of cats

3. **Making Predictions**:
   - The trained model will then be used to classify new images located in the `single_prediction` folder as either a dog or a cat.

# Technology Used

We will use **TensorFlow**, integrated with **Keras**, to build and train our CNN model. TensorFlow provides a robust backend for Keras, allowing us to perform efficient computations and leverage features like distributed training. Keras, as TensorFlow's high-level API, simplifies the process of creating, training, and evaluating deep learning models.

# Why TensorFlow and Keras?

- **TensorFlow**: A powerful and scalable platform for building machine learning models, offering support for both CPU and GPU.
- **Keras**: Simplifies the design and implementation of neural networks, making it accessible to both beginners and experts.

Keras was originally developed as a standalone API but has been fully integrated into TensorFlow to leverage its advanced features and improve usability. This integration allows us to write concise code while benefiting from TensorFlow's performance and scalability.

# Steps to Follow

1. **Prepare the Data**:
   - Organize the dataset into training and test directories.
   - Apply preprocessing techniques such as normalization and augmentation to improve model performance.

2. **Build the CNN Model**:
   - Use TensorFlow/Keras to create the CNN architecture.
   - Include layers such as convolutional, pooling, and fully connected layers.

3. **Train the Model**:
   - Train the model on the training dataset.
   - Monitor training performance using metrics like accuracy and loss.

4. **Evaluate the Model**:
   - Test the model using the test dataset to assess its performance.

5. **Make Predictions**:
   - Use the model to predict the class (dog or cat) of images in the `single_prediction` folder.

# Folder Structure

```
project/
|
|-- training_dataset/
|   |-- dogs/
|   |-- cats/
|
|-- test_dataset/
|   |-- dogs/
|   |-- cats/
|
|-- single_prediction/
|   |-- <new images to classify>
|
|-- ReadMe
|-- <other project files>
```

# Getting Started

1. Install the required libraries:
   ```bash
   pip install tensorflow
   ```

2. Run the script to train and evaluate the CNN model.

3. Use the `single_prediction` folder to test the model with your own images.

# Notes

- Ensure your dataset is properly organized and preprocessed before training.
- The `single_prediction` folder should contain images in a format supported by TensorFlow (e.g., JPEG, PNG).
- Use a GPU-enabled machine for faster training if possible.

# Medium Post Link
https://medium.com/@mchowdhury6/convolutional-neural-network-4b5e7e2319f2
