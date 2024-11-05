# Dog-Breed-classification
This project is a deep learning-based dog breed classifier using TensorFlow and Keras. The model leverages the DenseNet201 architecture, pre-trained on ImageNet, to classify images of dogs into one of 14 predefined breeds. Here’s a high-level overview of the project’s workflow:

Data Preparation: Images are loaded from a local directory, resized, and normalized. A subset of images is set aside for testing, while the rest forms the training dataset. The labels are one-hot encoded using Keras utilities.

Data Augmentation: An ImageDataGenerator applies random transformations (like flipping, rotation, zooming, and shifting) to the training images to make the model more robust against variations in the input data.

Model Architecture: DenseNet201, a powerful convolutional neural network, is used as the base model with pre-trained weights from ImageNet. A custom dense layer with 128 units and a softmax output layer (with 14 classes) is added for dog breed classification.

Training: The model is compiled with the Adam optimizer and categorical cross-entropy loss. It’s trained for 30 epochs with validation data to prevent overfitting.

Prediction: Once trained, the model can predict the breed of a given dog image. The prediction function uses a mapper function to return the corresponding breed name from the 14 classes based on the model’s output.

The project is ideal for those interested in transfer learning, fine-tuning models, and image classification applications in animal identification.
