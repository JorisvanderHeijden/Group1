# %load basic_graph.py
'''
Implementations of nodes for a computation graph. Each node
has a forward pass and a backward pass function, allowing
for the evaluation and backpropagation of data.
'''

from abc import ABC, abstractmethod
import math
import time
import numpy as np


class Node(object):

    def __init__(self, inputs):
        self.inputs = inputs

    @abstractmethod
    def forward(self):
        ''' Feed-forward the result '''
        raise NotImplementedError("Missing forward-propagation method.")

    @abstractmethod
    def backward(self, d):
        ''' Back-propagate the error
            d is the delta of the subsequent node in the network '''
        raise NotImplementedError("Missing back-propagation method.")


class ConstantNode(Node):

    def __init__(self, value):
        self.output = value

    def forward(self):
        return self.output

    def backward(self, d):
        pass


class VariableNode(Node):

    def __init__(self, value):
        self.output = value

    def forward(self):
        return self.output

    def backward(self, d):
        self.output -= 0.1 * d # Gradient Descent


class AdditionNode(Node):

    def forward(self):
        self.output = sum([i.forward() for i in self.inputs])
        return self.output

    def backward(self, d):
        for i in self.inputs:
            i.backward(d)


class MultiplicationNode(Node):

    def forward(self):
        self.output = self.inputs[0].forward() * self.inputs[1].forward()
        return self.output

    def backward(self, d):
        self.inputs[0].backward(d * self.inputs[1].output)
        self.inputs[1].backward(d * self.inputs[0].output)


class MSENode(Node):

    def forward(self):
        self.output = 0.5 * (
            self.inputs[0].forward() - self.inputs[1].forward())**2
        return self.output

    def backward(self, d):
        self.inputs[0].backward(d * (self.inputs[0].output - self.inputs[1].output))
        self.inputs[1].backward(d * (self.inputs[1].output - self.inputs[0].output))


class SigmoidNode(Node):

    def forward(self):
        self.output = 1.0 / (1.0 + math.exp(-self.inputs[0].forward()))
        return self.output

    def backward(self, d):
        self.inputs[0].backward(d * self.output * (1.0 - self.output))

class ReLUNode(object):

    def forward(self):
        raise NotImplementedError("Forward pass for ReLU activation node has not been implemented yet.")

    def backward(self, d):
        raise NotImplementedError("Backward pass for ReLU activation node has not been implemented yet.")

class TanhNode(object):

    def forward(self):
        raise NotImplementedError("Forward pass for tanh activation node has not been implemented yet.")

    def backward(self, d):
        raise NotImplementedError("Backward pass for tanh activation node has not been implemented yet.")
        
# Example graph as shown in MLP lecture slides
class MyGraph(object):

    def __init__(self, x, y, weigths):
        ''' x: input
            y: expected output
            w: initial weight
            b: initial bias '''
                       
        self.weights = [VariableNode(weight) for weight in weigths]

        self.z1 = MultiplicationNode([ConstantNode(x),self.weights[0]])
        self.n1 = SigmoidNode([self.z1])
        self.z2 = MultiplicationNode([self.n1, self.weights[1]])
        self.n2 = SigmoidNode([self.z2])
        self.z3 = MultiplicationNode([self.n1, self.weights[2]])
        self.n3 = SigmoidNode([self.z3])
        self.z4 = AdditionNode([MultiplicationNode([
                             self.n2,
                             self.weights[3]]),
                             MultiplicationNode([
                             self.n3,
                             self.weights[4]])])
        self.n4 = SigmoidNode([self.z4])
        self.graph = MSENode([self.n4,ConstantNode(y)])

    def forward(self):
        return self.graph.forward()

    def backward(self, d):
        self.graph.backward(d)
    def set_weights(self, new_weights):
        for i in len(new_weights):
            self.weights[i].output = new_weights[i]

    def get_weights(self):
        return [weight.output for weight in self.weights]

x=2
y=3
w1=2
w2=1
w3=2
w4=4
w5=1


sg = MyGraph(x, y,[w1,w2,w3,w4,w5])

Error1 = sg.forward()

print(sg.z1.output)
print(sg.n1.output)
print(sg.n2.output)
print(sg.n3.output)
print(sg.n4.output)
print("Graph",sg.graph.output)

print("Initial error prediction is", prediction)


sg.backward(sg.n4.output)

new_weights=sg.get_weights()
print("w has new value", np.transpose(new_weights))


'''
# Example graph as shown in MLP lecture slides
class SampleGraph(object):

    def __init__(self, x, y, w, b):
        '' x: input
            y: expected output
            w: initial weight
            b: initial bias ''
        self.w = VariableNode(w)
        self.b = VariableNode(b)
        self.graph = MSENode([
            AdditionNode([MultiplicationNode([ConstantNode(x),self.w]),
                          MultiplicationNode([self.b,ConstantNode(1)])
            ]),
            ConstantNode(y)
        ])

    def forward(self):
        return self.graph.forward()

    def backward(self, d):
        self.graph.backward(d)


class Neuron(Node):

    def __init__(self, inputs, weights, activation):
        '' weights: list of initial weights, same length as inputs ''
        self.inputs = inputs
        # Initialize a weight for each input
        self.weights = [VariableNode(weight) for weight in weights]
        # Neurons normally have a bias, ignore for this assignment
        #self.bias = VariableNode(bias, "b")

        # Multiplication node for each pair of inputs and weights
        mults = [MultiplicationNode([i, w]) for i, w, in zip(self.inputs, self.weights)]
        # Neurons normally have a bias, ignore for this assignment
        #mults.append(MultiplicationNode([self.bias, ConstantNode(1)]))

        # Sum all multiplication results
        added = AdditionNode(mults)

        # Apply activation function
        if activation == 'sigmoid':
            self.graph = SigmoidNode([added])
        elif activation == 'relu':
            self.graph = ReLUNode([added])
        elif activation == 'tanh':
            self.graph = TanhNode([added])
        else:
            raise ValueError("Unknown activation function.")

    def forward(self):
        return self.graph.forward()

    def backward(self, d):
        self.graph.backward(d)

    def set_weights(self, new_weights):
        for i in len(new_weights):
            self.weights[i].output = new_weights[i]

    def get_weights(self):
        return [weight.output for weight in self.weights]

if __name__ == '__main__':
    print("Loaded simple graph nodes")

# Example network
sg = SampleGraph(2, 2, 2, 1)
prediction = sg.forward()
print("Initial prediction is", prediction)
sg.backward(1)
print("w has new value", sg.w.output)
print("b has new value", sg.b.output)

class MyGraph(object):

    def __init__(self, x, y, w, b):
        '' x: input
            y: expected output
            w: initial weight
            b: initial bias ''
        self.w = VariableNode(w)
        self.b = VariableNode(b)
        self.n1 =Neuron([ConstantNode(x)], [w[0]],'sigmoid')
        
        self.n2 =Neuron([self.n1],[w[1]],'sigmoid') 
    
        self.n3 = Neuron([self.n1],[w[2]],'sigmoid')
    
        self.n4=Neuron([self.n2, self.n3],[w[3], w[4]],'sigmoid')
        
        self.graph = self.n4

    def forward(self):
        return self.graph.forward()

    def backward(self, d):
        self.graph.backward(d)
        



#n4=z4.forward()
sg.forward()


print("n4", Neuron.get_weights(sg))

prediction=sg.forward()

sg.backward(y)

print('n4',prediction)

print("n4", sg.get_weights())

#sg.backward(1)
#print("w has new value", sg.w.output)
#print("b has new value", sg.b.output)

# Example graph as shown in MLP lecture slides

'''   
