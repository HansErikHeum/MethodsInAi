# Use Python 3.8 or newer (https://www.python.org/downloads/)
import unittest
# Remember to install numpy (https://numpy.org/install/)!
import numpy as np
import pickle
import os
import math
"""
@author: Hans Erik Heum

I use the pseudocode described in the book, combined with formulas for matrix-multiplication to solve this task.
Even though some might prefer to run the algorithm with the bias-vectors combined with the weigth-matrices, i chose to calculate the bias's weigth seperateley.

I sometimes classify variables as 'inputX and outputX', for instance: inputWeigths and outputWeigths. The input-part, means it is between the input-nodes and the hidden layers. 
The output-prefix tells us it is between the hidden layer and the output node.

When the 'perceptron'-prefix is used, we are refering to task 1.
'nn' stands for Neural Network, and we are refering to task 2.

"""


class NeuralNetwork:
    """Implement/make changes to places in the code that contains #TODO."""

    def __init__(self, input_dim: int, hidden_layer: bool) -> None:
        """
        Initialize the feed-forward neural network with the given arguments.
        :param input_dim: Number of features in the dataset.
        :param hidden_layer: Whether or not to include a hidden layer.
        :return: None.
        """

        # Number of hidden units if hidden_layer = True.
        self.hidden_units = 25

        # This parameter is called the step size, also known as the learning rate (lr).
        self.lr = 1e-3

        # This is the number of times we are going to repeat. This is often known as epochs.
        self.epochs = 400

        # We are going to store the data here.
        # Since you are only asked to implement training for the feed-forward neural network,
        # only self.x_train and self.y_train need to be used. You will need to use them to implement train().
        # The self.x_test and self.y_test is used by the unit tests. Do not change anything in it.
        self.x_train, self.y_train = None, None
        self.x_test, self.y_test = None, None

        # TODO: Make necessary changes here. For example, assigning the arguments "input_dim" and "hidden_layer" to
        # variables and so forth.
        self.input_dim = input_dim
        self.hidden_layer = hidden_layer
        self.perceptronWeigths = None
        self.nnInputWeigths = None
        self.nnOutputWeigths = None
        self.initializeWeights()

        self.perceptronBias = None
        self.nnHiddenBias = None
        self.nnOutputBias = None
        self.initializeBias()

    def initializeWeights(self):
        # makes all the weights with numbers between -0.5 and 0.5
        self.perceptronWeigths = np.random.uniform(
            -0.5, 0.5, size=(1, self.input_dim))

        self.nnInputWeigths = np.random.uniform(
            -0.5, 0.5, size=(self.hidden_units, self.input_dim))
        self.nnOutputWeigths = np.random.uniform(
            -0.5, 0.5, size=(1, self.hidden_units))

    def initializeBias(self):
        # Makes bias-matrices with numbers between -0.5 and 0.5
        self.perceptronBias = np.random.uniform(-0.5, 0.5, size=(1, 1))

        self.nnHiddenBias = np.random.uniform(-0.5,
                                              0.5, size=(self.hidden_units, 1))
        self.nnOutputBias = np.random.uniform(-0.5, 0.5, size=(1, 1))

    def load_data(self, file_path: str = os.path.join(os.getcwd(), 'data_breast_cancer.p')) -> None:
        """
        Do not change anything in this method.

        Load data for training and testing the model.
        :param file_path: Path to the file 'data_breast_cancer.p' downloaded from Blackboard. If no arguments is given,
        the method assumes that the file is in the current working directory.

        The data have the following format.
                   (row, column)
        x: shape = (number of examples, number of features)
        y: shape = (number of examples)
        """
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            self.x_train, self.y_train = data['x_train'], data['y_train']
            self.x_test, self.y_test = data['x_test'], data['y_test']

    def train(self) -> None:
        """Run the backpropagation algorithm to train this neural network"""

        if self.hidden_layer:
            self.trainNeuralNetwork()
        else:
            self.trainPerceptron()

    def trainPerceptron(self):
        for i in range(self.epochs):
            print(i)
            # for each examples
            for i in range(len(self.x_train)):
                self.updatePerceptronWeigths(
                    self.x_train[i], self.y_train[i])

    def updatePerceptronWeigths(self, x, y):
        output = self.perceptronFeedForward(x)
        outputError = y-output
        gradientOutput = output*(1-output)

        x = np.asarray(np.asmatrix(x))
        deltaPerceptronWeigths = np.dot(self.lr*outputError*gradientOutput, x)
        self.perceptronWeigths += deltaPerceptronWeigths

        self.perceptronBias += self.lr*outputError*gradientOutput

    def trainNeuralNetwork(self):
        for i in range(self.epochs):
            print(i)
            for i in range(len(self.x_train)):
                self.updateNeuralNetworkWeights(
                    self.x_train[i], self.y_train[i])

    def updateNeuralNetworkWeights(self, x, y):

        output, hiddenOutput = self.nnFeedForward(x)
        outputError = y-output
        # we implement the formula lr*x*error, to find the new weigths.
        gradientOutput = output*(1-output)

        newOutputWeights = np.dot(
            self.lr*outputError*gradientOutput, hiddenOutput.transpose())
        self.nnOutputWeigths += newOutputWeights
        # find new bias
        self.nnOutputBias += self.lr*outputError*gradientOutput

        # finds hidden error
        hiddenError = np.dot(self.nnOutputWeigths.transpose(), outputError)

        gradientInput = hiddenOutput*(1-hiddenOutput)

        x = np.asarray(np.asmatrix(x))
        newInputWeigths = np.dot(
            self.lr*hiddenError*gradientInput, x)
        self.nnInputWeigths += newInputWeigths
        # find new bias
        self.nnHiddenBias += self.lr*hiddenError*gradientInput

    def predict(self, x: np.ndarray) -> float:
        """
        Given an example x we want to predict its class probability.
        For example, for the breast cancer dataset we want to get the probability for cancer given the example x.
        :param x: A single example (vector) with shape = (number of features)
        :return: A float specifying probability which is bounded [0, 1].
        """

        # print("X verdier er ", x)  # 30 jævler bortover

        # TODO: Implement the forward pass.

        if self.hidden_layer:
            output, hiddenOutput = self.nnFeedForward(x)
            return float(output[0][0])
        else:  # perceptron
            return float(self.perceptronFeedForward(x)[0][0])

    def perceptronFeedForward(self, x):
        x = np.asarray(np.asmatrix(x).transpose())
        output = np.dot(self.perceptronWeigths, x)
        output = output + self.perceptronBias

        vfunc = np.vectorize(self.sigmoid)
        output = vfunc(output)
        return output

    def nnFeedForward(self, x):
        # ____________________________hiddenLayer________________________
        #  Weigths * X
        x = np.asarray(np.asmatrix(x).transpose())
        hidden = np.dot(self.nnInputWeigths, x)
        # add bias
        hidden = hidden + self.nnHiddenBias
        # apply sigmoid function
        vfunc = np.vectorize(self.sigmoid)
        hiddenOutput = vfunc(hidden)

        # ____________________________outputLayer_________________________
        #   weigths * hiddenn output
        output = np.dot(self.nnOutputWeigths, hiddenOutput)
        # add bias
        output = output + self.nnOutputBias
        # apply sigmoid function
        output = vfunc(output)
        return output, hiddenOutput

    def sigmoid(self, x):
        return (1 / (1 + np.exp(-x)))


class TestAssignment5(unittest.TestCase):
    """
    Do not change anything in this test class.

    --- PLEASE READ ---
    Run the unit tests to test the correctness of your implementation.
    This unit test is provided for you to check whether this delivery adheres to the assignment instructions
    and whether the implementation is likely correct or not.
    If the unit tests fail, then the assignment is not correctly implemented.
    """

    def setUp(self) -> None:
        self.threshold = 0.8
        self.nn_class = NeuralNetwork
        self.n_features = 30

    def get_accuracy(self) -> float:
        """Calculate classification accuracy on the test dataset."""
        self.network.load_data()
        self.network.train()

        n = len(self.network.y_test)
        correct = 0
        for i in range(n):
            # Predict by running forward pass through the neural network
            pred = self.network.predict(self.network.x_test[i])
            # Sanity check of the prediction
            assert 0 <= pred <= 1, 'The prediction needs to be in [0, 1] range.'
            # Check if right class is predicted
            correct += self.network.y_test[i] == round(float(pred))
        print("accuracy : ", round(correct / n, 3))
        return round(correct / n, 3)

    def test_perceptron(self) -> None:
        """Run this method to see if Part 1 is implemented correctly."""

        self.network = self.nn_class(self.n_features, False)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')

    def test_one_hidden(self) -> None:
        """Run this method to see if Part 2 is implemented correctly."""

        self.network = self.nn_class(self.n_features, True)
        accuracy = self.get_accuracy()
        self.assertTrue(accuracy > self.threshold,
                        'This implementation is most likely wrong since '
                        f'the accuracy ({accuracy}) is less than {self.threshold}.')


if __name__ == '__main__':
    unittest.main()
