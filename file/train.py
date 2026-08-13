import numpy
from neuralnetwork import neuralnetwork
import os

input_nodes = 784
hidden_nodes = 100
output_nodes = 10

learning_rate = 0.1


n = neuralnetwork(
    input_nodes,
    hidden_nodes,
    output_nodes,
    learning_rate
)




BASE = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

TRAIN_FILE = os.path.join(
    os.path.dirname(__file__),
    "data",
    "mnist_train.csv"
)


data_file = open(
    TRAIN_FILE,
    "r"
)

training_data = data_file.readlines()

data_file.close()
epochs = 7;
for e in range(epochs):


    for record in training_data:

        all_values = record.split(',')


        inputs = (
            numpy.asfarray(all_values[1:])
            /255.0*0.99
        )+0.01


        targets = numpy.zeros(output_nodes)+0.01

        targets[int(all_values[0])] = 0.99


        n.train(inputs, targets)
        pass
    pass



print("训练完成")

MODEL_FILE = os.path.join(
    BASE,
    "model",
    "mnist.model"
)


n.save(
    MODEL_FILE
)
print("模型保存成功")