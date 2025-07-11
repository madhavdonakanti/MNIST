# Neural Net from scratch

This was a project to develop a neural network from scratch to be trained on the MNIST data set of 28x28 images of handwritten digits. 

A feedforward neural network implementation built with NumPy for multi-class digit classification.

## Architecture

- **Input**: 784 features (28×28 images)
- **Hidden Layers**: 3 layers × 280 neurons each (why? saw somewhere that no.of neurons=root(m*n) where m and n are no. of input and output neurons), can easily be changed during fine tuning
- **Output**: 10 classes (digits 0-9)  (having one output and predicting the number with regression would’ve been more inaccurate)
- **Activation**: Sigmoid/ReLu/Tanh
- **Loss**: Categorical Cross-Entropy with Softmax

## Key Features

- Hand-coded backpropagation algorithm
- Support for multiple activation functions
- Training/validation monitoring with cost visualization
- Can adjust no. of samples for training and validation from train set
- Code for normal gradient descent and Mini batch gradient descent is available

## Files

- train.csv  - Training data with labels
- test.csv - Test data for predictions
- Submissions - Generated predictions with ImageID and Label as columns for different hyperparameters
- Cost graphs - model cost and validation cost graphs for different hyperparameters

## How to Run

1. Load the code along with the train.csv and test.csv files
2. Select the activation function by typing it in in the codefile
3. Put the correct file paths required in the codefile
4. RUN

## Notation in file names

GD : Gradient descent

MGD : Mini-batch gradient descent

sub : submission file for test.csv

sig : sigmoid

number in the image name : number of epochs

## Inferences

- Mini-batch gradient descent massively outperformed normal gradient descent. Convergence was not that noisy and was much faster(almost twice as fast) due to more frequently updated weights and biases. Resulted in a much lower cost in a lot less time.
- Sigmoid and tanh performed slightly better than ReLu in normal gradient descent but performed significantly better than ReLu in mini-batch gradient descent. Costs at end of training:
    - Normal Gradient descent :
        - Sigmoid - 2.1ish
        - ReLu - 2.3ish
        - Tanh - 3ish
    - Mini-Batch Gradient descent :
        - Sigmoid - 0.5ish
        - ReLu- 2.3ish
        - Tanh - 0.8ish
- I believe as it was a relatively small dataset ReLu didnt have much of an advantage as the training process wasnt as computationally intensive/expensive. ReLu’s simplicity would shine in terms of cutting training time and expense on a much larger training task.
- With normal gradient descent ReLu was the activation function of choice as the training time was almost half of that of sigmoid/tanh with similar end costs.
- With mini-batch gradient descent sigmoid, tanh, and ReLu all took about the same training time but sigmoid and tanh performed significantly better.
- Mini-batch converged in far less epochs due to the larger no. of steps.

## More that could’ve been done

- Fine tuning hyperparameters like batch size, no.of epochs, and learning rate.
- Different weights and biases initialization
- Trying out other activation functions like Leaky ReLu etc.
- Trying out single output with mean squared error as the loss function.

inspired from :

 [https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc](https://medium.com/@waadlingaadil/learn-to-build-a-neural-network-from-scratch-yes-really-cac4ca457efc)   and   Stanford CS231n