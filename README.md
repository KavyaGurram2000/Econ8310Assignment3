# Fashion MNIST Classification Project

This project focuses on classifying images from the Fashion MNIST dataset using a Convolutional Neural Network (CNN) implemented in PyTorch.

## Project Files

- **`Untitled8.ipynb`**: Jupyter notebook containing the code for data loading, model definition, training, and saving model weights.
- **`fashion_mnist_weights.pth`**: File storing the trained model's weights.
- **`train_and_save`**: Function to train the model and save its weights.
- **`evaluate_model`**: Function to load the saved weights and evaluate the model's accuracy on test data.

## Model Overview

The CNN model (`FastFashionNet`) includes:

1. **Convolutional Layers**: Extract features from input images.
2. **Activation Functions**: Introduce non-linearity to the model.
3. **Pooling Layers**: Reduce the spatial dimensions of the data.
4. **Fully Connected Layers**: Perform classification based on extracted features.

## Training and Evaluation

1. **Training**: The model is trained for 3 epochs using the Adam optimizer and Cross-Entropy Loss function. The training process includes periodic loss reporting to monitor progress.

2. **Evaluation**: After training, the model's performance is evaluated on the test dataset, achieving an accuracy of approximately 91.17%.

## Usage Instructions

1. **Training the Model**:
   - Run the `train_and_save` function to train the model and save the weights:
     ```python
     model = train_and_save()
     ```

2. **Evaluating the Model**:
   - Use the `evaluate_model` function to load the saved weights and assess the model's accuracy:
     ```python
     accuracy = evaluate_model()
     print(f"Test Accuracy: {accuracy:.2f}%")
     ```

## Results Summary

The model achieves an accuracy of approximately 91.17% on the test dataset. Training for more epochs or adjusting hyperparameters may improve performance.

---

This project provides a straightforward implementation of a CNN for image classification using PyTorch, suitable for educational purposes and as a foundation for further experimentation.
