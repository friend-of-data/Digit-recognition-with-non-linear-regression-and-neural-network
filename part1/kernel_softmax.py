import numpy as np
from scipy import sparse
from utils import *


def kernel_func(X, Y, c, p):
    """
    Compute the polynomial kernel between two matrices X and Y::
        K(x, y) = (<x, y> + c)^p
    for each pair of rows x in X and y in Y.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (m, d) NumPy array (m datapoints each with d features)
        c - a coefficient to trade off high-order and low-order terms (scalar)
        p - the degree of the polynomial kernel

    Returns:
        kernel_matrix - (n, m) NumPy array containing the kernel matrix
    """
    dot_product = np.dot(X, Y.T)
    kernel_matrix = np.power(dot_product + c, p)
    return kernel_matrix

def compute_kernel_probabilities(X, theta, temp_parameter, kernel_func, **kernel_params):
    """
    Computes, for each datapoint X[i], the probability that X[i] is labeled as j
    for j = 0, 1, ..., k-1 using kernelized features.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        theta - (k, n) NumPy array, where row j represents the parameters of our model for label j
                in the transformed kernel space
        temp_parameter - the temperature parameter of softmax function (scalar)
        kernel_func - a function that computes the kernel matrix
        **kernel_params - additional parameters for the kernel function

    Returns:
        H - (k, n) NumPy array, where each entry H[j][i] is the probability that X[i] is labeled as j
    """
    kernel_matrix = kernel_func(theta, X, **kernel_params)
    logits = kernel_matrix / temp_parameter
    logits -= np.max(logits, axis=0, keepdims=True)
    exp_logits = np.exp(logits)
    H = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)
    return H

def compute_kernel_cost_function(X, Y, theta, lambda_factor, temp_parameter, kernel_func, **kernel_params):
    """
    Computes the total cost over every datapoint using kernelized features.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, n) NumPy array, where row j represents the parameters of our
                model for label j in the transformed kernel space
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)
        kernel_func - a function that computes the kernel matrix
        **kernel_params - additional parameters for the kernel function

    Returns
        c - the cost value (scalar)
    """
    n = X.shape[0]
    k = theta.shape[0]
    
    probabilities = compute_kernel_probabilities(X, theta, temp_parameter, kernel_func, **kernel_params)
    one_hot_Y = sparse.coo_matrix((np.ones(n), (Y, np.arange(n))), shape=(k, n)).toarray()
    epsilon = 1e-10
    log_probabilities = np.log(probabilities + epsilon)
    cross_entropy_loss = -np.sum(one_hot_Y * log_probabilities) / n
    l2_regularization = (lambda_factor / 2) * np.sum(theta**2)
    cost = cross_entropy_loss + l2_regularization
    
    return cost

def run_kernel_gradient_descent_iteration(X, Y, theta, alpha, lambda_factor, temp_parameter, kernel_func, **kernel_params):
    """
    Runs one step of batch gradient descent using kernelized features.

    Args:
        X - (n, d) NumPy array (n datapoints each with d features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        theta - (k, n) NumPy array, where row j represents the parameters of our
                model for label j in the transformed kernel space
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        temp_parameter - the temperature parameter of softmax function (scalar)
        kernel_func - a function that computes the kernel matrix
        **kernel_params - additional parameters for the kernel function

    Returns:
        theta - (k, n) NumPy array that is the final value of parameters theta
    """
    n = X.shape[0]
    k = theta.shape[0]

    kernel_matrix = kernel_func(theta, X, **kernel_params)
    logits = kernel_matrix / temp_parameter
    logits -= np.max(logits, axis=0, keepdims=True)
    exp_logits = np.exp(logits)
    probabilities = exp_logits / np.sum(exp_logits, axis=0, keepdims=True)

    one_hot_Y = sparse.coo_matrix((np.ones(n), (Y, np.arange(n))), shape=(k, n)).toarray()
    K = kernel_func(X, X, **kernel_params)
    grad = (-1 / (temp_parameter * n)) * ((one_hot_Y - probabilities) @ K) + lambda_factor * theta

    theta -= alpha * grad

    return theta

def softmax_regression(X, Y, temp_parameter, alpha, lambda_factor, k, num_iterations, kernel_func, **kernel_params):
    """
    Runs batch gradient descent for a specified number of iterations on a dataset
    with theta initialized to the all-zeros array using kernelized features.

    Args:
        X - (n, d - 1) NumPy array (n data points, each with d-1 features)
        Y - (n, ) NumPy array containing the labels (a number from 0-9) for each
            data point
        temp_parameter - the temperature parameter of softmax function (scalar)
        alpha - the learning rate (scalar)
        lambda_factor - the regularization constant (scalar)
        k - the number of labels (scalar)
        num_iterations - the number of iterations to run gradient descent (scalar)
        kernel_func - a function that computes the kernel matrix
        **kernel_params - additional parameters for the kernel function

    Returns:
        theta - (k, n) NumPy array that is the final value of parameters theta in kernel space
        cost_function_progression - a Python list containing the cost calculated at each step of gradient descent
    """
    # Compute the kernel matrix between X and itself
    K = kernel_func(X, X, **kernel_params)
    theta = np.zeros([k, X.shape[0]])  # Initialize theta for k classes and n data points in kernel space
    cost_function_progression = []
    
    for i in range(num_iterations):
        cost_function_progression.append(
            compute_kernel_cost_function(X, Y, theta, lambda_factor, temp_parameter, kernel_func, **kernel_params)
        )
        theta = run_kernel_gradient_descent_iteration(
            X, Y, theta, alpha, lambda_factor, temp_parameter, kernel_func, **kernel_params
        )
        
    return theta, cost_function_progression

# Example usage with polynomial kernel
# Example data
train_x, train_y, test_x, test_y = get_MNIST_data()

temp_parameter = 1.0
alpha = 0.3
lambda_factor = 1.0e-4
k = 10  # Number of classes
num_iterations = 150

# Define the polynomial kernel parameters
kernel_params = {'c': 1, 'p': 3}

# Run softmax regression with polynomial kernel
theta, cost_function_progression = softmax_regression(
    train_x, train_y, temp_parameter, alpha, lambda_factor, k, num_iterations, kernel_func, **kernel_params
)



