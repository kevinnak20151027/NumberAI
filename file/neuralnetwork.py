import numpy
import scipy.special
import pickle
def activation_function(x):
    return scipy.special.expit(x)

class neuralnetwork:

    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate):

        self.inodes = inputnodes
        self.hnodes = hiddennodes
        self.onodes = outputnodes

        self.lr = learningrate

        self.wih = numpy.random.normal(
            0.0,
            pow(self.hnodes, -0.5),
            (self.hnodes, self.inodes)
        )

        self.who = numpy.random.normal(
            0.0,
            pow(self.onodes, -0.5),
            (self.onodes, self.hnodes)
        )

        self.activation_function = activation_function


    def train(self, inputs_list, targets_list):

        inputs = numpy.array(inputs_list, ndmin=2).T
        targets = numpy.array(targets_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)


        output_errors = targets - final_outputs

        hidden_errors = numpy.dot(self.who.T, output_errors)


        self.who += self.lr * numpy.dot(
            (output_errors * final_outputs * (1-final_outputs)),
            hidden_outputs.T
        )


        self.wih += self.lr * numpy.dot(
            (hidden_errors * hidden_outputs * (1-hidden_outputs)),
            inputs.T
        )


    def query(self, inputs_list):

        inputs = numpy.array(inputs_list, ndmin=2).T

        hidden_inputs = numpy.dot(self.wih, inputs)
        hidden_outputs = self.activation_function(hidden_inputs)

        final_inputs = numpy.dot(self.who, hidden_outputs)
        final_outputs = self.activation_function(final_inputs)

        return final_outputs


    # 保存模型
    def save(self, filename):

        with open(filename, "wb") as f:
            pickle.dump(self, f)


    # 读取模型
    @staticmethod
    def load(filename):

        with open(filename, "rb") as f:
            return pickle.load(f)