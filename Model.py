import numpy as np
import matplotlib.pyplot as plt
import h5py
import sklearn
import sklearn.datasets
import math

def sigmoid(Z):
    """
    Implements the sigmoid activation in numpy
    
    Arguments:
    Z -- numpy array of any shape
    
    Returns:
    A -- output of sigmoid(z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    
    A = 1/(1+np.exp(-Z))
    cache = Z
    
    return A, cache

def relu(Z):
    """
    Implement the RELU function.

    Arguments:
    Z -- Output of the linear layer, of any shape

    Returns:
    A -- Post-activation parameter, of the same shape as Z
    cache -- a python dictionary containing "A" ; stored for computing the backward pass efficiently
    """
    
    A = np.maximum(0,Z)
    
    assert(A.shape == Z.shape)
    
    cache = Z 
    return A, cache


def relu_backward(dA, cache):
    """
    Implement the backward propagation for a single RELU unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    dZ = np.array(dA, copy=True) # just converting dz to a correct object.
    
    # When z <= 0, you should set dz to 0 as well. 
    dZ[Z <= 0] = 0
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def sigmoid_backward(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ -- Gradient of the cost with respect to Z
    """
    
    Z = cache
    
    s = 1/(1+np.exp(-Z))
    dZ = dA * s * (1-s)
    
    assert (dZ.shape == Z.shape)
    
    return dZ

def predict(X, y, parameters):
    """
    This function is used to predict the results of a  n-layer neural network.
    
    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    
    Returns:
    p -- predictions for the given dataset X
    """
    
    m = X.shape[1]
    p = np.zeros((1, m), dtype=np.int32)  # Use np.int32 or np.int64 if specific precision is needed


    
    # Forward propagation
    a3, caches = forward_propagation(X, parameters)
    
    # convert probas to 0/1 predictions
    for i in range(0, a3.shape[1]):
        if a3[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    # print results
    print("Accuracy: "  + str(np.mean((p[0,:] == y[0,:]))))
    
    return p

def plot_decision_boundary(model, X, y):
    # Set min and max values and give it some padding
    x_min, x_max = X[0, :].min() - 1, X[0, :].max() + 1
    y_min, y_max = X[1, :].min() - 1, X[1, :].max() + 1
    h = 0.01
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole grid
    Z = model(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral)
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.scatter(X[0, :], X[1, :], c=y, cmap=plt.cm.Spectral)
    plt.show()
    
def predict_dec(parameters, X):
    """
    Used for plotting decision boundary.
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (m, K)
    
    Returns
    predictions -- vector of predictions of our model (red: 0 / blue: 1)
    """
    
    # Predict using forward propagation and a classification threshold of 0.5
    a3, cache = forward_propagation(X, parameters)
    predictions = (a3>0.5)
    return predictions

def load_dataset():
    np.random.seed(1)
    train_X, train_Y = sklearn.datasets.make_circles(n_samples=300, noise=.05)
    np.random.seed(2)
    test_X, test_Y = sklearn.datasets.make_circles(n_samples=100, noise=.05)
    # Visualize the data
    plt.scatter(train_X[:, 0], train_X[:, 1], c=train_Y, s=40, cmap=plt.cm.Spectral);
    train_X = train_X.T
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_X = test_X.T
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    return train_X, train_Y, test_X, test_Y

############ Parameters Initialization ############

def initialize_parameters_he(layer_dims):

    parameters = {} # save initailized parameters
    L = len(layer_dims) # calculate how many layers in a network

    for l in range(1,L): 
        
        parameters["W" +str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * math.sqrt(2./layer_dims[l-1]) #using he method
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1)) * math.sqrt(2./layer_dims[l-1]) #using he method

    return parameters

def initialize_parameters_random(layer_dims):

    parameters = {} # save initailized parameters
    L = len(layer_dims) # calculate how many layers in a network

    for l in range(1,L): 
        
        parameters["W" +str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01 # random initialization
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1))  # using random initialization

    return parameters

def initialize_parameters_zeros(layer_dims):

    parameters = {} # save initailized parameters
    L = len(layer_dims) # calculate how many layers in a network

    for l in range(1,L): 
        
        parameters["W" +str(l)] = np.zeros((layer_dims[l], layer_dims[l-1]))  # 0 initialization
        parameters["b" + str(l)] = np.zeros((layer_dims[l], 1)) # 0 initialization

    return parameters


############ Foward Propagation ############

def linear_forward(W, A_prev, b):
    #computing Z
    Z = np.dot(W,A_prev) + b
    cache = (W, A_prev, b) 
    
    return Z, cache

def linear_activation_forward(W, A_prev, b, activation):

    if activation == "relu":
        
        Z, linear_cache = linear_forward(W, A_prev, b)
        A, activation_cache = relu(Z)

    elif activation == "sigmoid":
        
        Z, linear_cache = linear_forward(W, A_prev, b)
        A, activation_cache = sigmoid(Z)

    cache = (linear_cache, activation_cache)

    return A, cache

def forward_propagation(X, parameters):
    
    # storing A0
    A_prev = X
    # storing parameters for backward propagation
    caches = []
    L = len(parameters) // 2 #gives us how many layers our network has

    for l in range(1, L):

        A, cache = linear_activation_forward(parameters["W" + str(l)], A_prev, parameters["b" + str(l)], activation = "relu")
        caches.append(cache)
        A_prev = A

    AL, cache = linear_activation_forward(parameters["W" + str(L)], A_prev, parameters["b" + str(L)], activation = "sigmoid")
    caches.append(cache)
    
    
    return AL, caches

############ Computing cost ############

def compute_cost(AL, Y):
    """
    Implement the cost function defined by equation (7).

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    
    m = Y.shape[1]

    # Compute loss from aL and y.
    ### START CODE HERE ### (≈ 1 lines of code)
    cost = (-1/m) * (np.dot(Y, np.log(AL).T) + np.dot((1-Y), np.log(1-AL).T))
    ### END CODE HERE ###
    
    cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    assert(cost.shape == ())
    
    return cost

############ propagating backward #############


def linear_backward(dZ, linear_cache):
    W, A_prev, b = linear_cache
    m = A_prev.shape[1]
    
    # Calculate gradients
    dA_prev =  np.dot(W.T, dZ)
    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True) 

    return dA_prev, dW, db

def linear_activation_backward(dA, cache, activation):
    linear_cache, activation_cache = cache

    if activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
    elif activation == "relu":
        dZ = relu_backward(dA, activation_cache)
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db

def backward_propagation(AL, caches, Y):
    grads = {}
    L = len(caches)
    Y = Y.reshape(AL.shape) 
    current_cache = caches[L-1]
    
    # Sigmoid -> Linear
    dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, "sigmoid")

    # Loop from l=L-2 to l=0
    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 1)], current_cache, activation="relu")
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads

def update_parameters(parameters, grads, learning_rate):
    
    L = len(parameters) // 2

    for l in range(1, L+1):
        
        parameters["W" +str(l)] -= learning_rate * grads["dW" + str(l)] 
        parameters["b" +str(l)] -= learning_rate * grads["db" + str(l)] 
        
    return parameters
