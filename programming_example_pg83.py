import numpy as np

# Set a random seed for reproducibility of results
np.random.seed(3)

# Define the learning rate for the model
LEARNING_RATE = 0.1

# List of indices to access training data in random order
index_list = [0, 1, 2, 3]

# Training data: 4 examples with 3 features each (including bias as the first feature)
x_train = [
    np.array([1.0, -1.0, -1.0]),  # Input vector 1
    np.array([1.0, -1.0, 1.0]),   # Input vector 2
    np.array([1.0, 1.0, -1.0]),   # Input vector 3
    np.array([1.0, 1.0, 1.0])     # Input vector 4
]

# Corresponding labels (ground truth) for the training data
y_train = [0.0, 1.0, 1.0, 0.0]

# Function to initialize weights for a neuron
def neuron_w(input_count):
    # Initialize weights: zero for bias, random values for other inputs
    weights = np.zeros(input_count + 1)
    for i in range(1, input_count + 1):
        weights[i] = np.random.uniform(-1.0, 1.0)
    return weights

# Initialize weights for 3 neurons
n_w = [neuron_w(2), neuron_w(2), neuron_w(2)]

# Initialize output values for each neuron
n_y = [0, 0, 0]

# Initialize error values for each neuron
n_error = [0, 0, 0]

# Function to display current weights of neurons
def show_learning():
    print('Current weights:')
    for i, w in enumerate(n_w):
        print(f'neuron {i}: w0 = {w[0]:5.2f}, w1 = {w[1]:5.2f}, w2 = {w[2]:5.2f}')
        print('----------------')

# Function for the forward pass
def forward_pass(x):
    global n_y
    # Compute the output of the first two neurons using tanh activation function
    n_y[0] = np.tanh(np.dot(n_w[0], x))  # Neuron 0
    n_y[1] = np.tanh(np.dot(n_w[1], x))  # Neuron 1

    # Prepare inputs for the third neuron (including bias)
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])

    # Compute the output of the third neuron using logistic sigmoid activation function
    z2 = np.dot(n_w[2], n2_inputs)
    n_y[2] = 1.0 / (1.0 + np.exp(-z2))

# Function for the backward pass (backpropagation)
def backward_pass(y_truth):
    global n_error
    # Calculate the derivative of the loss function
    error_prime = -(y_truth - n_y[2])

    # Derivative of the logistic sigmoid function
    derivative = n_y[2] * (1.0 - n_y[2])
    n_error[2] = error_prime * derivative

    # Derivative of the tanh function for the first neuron
    derivative = 1.0 - n_y[0]**2
    n_error[0] = n_w[2][1] * n_error[2] * derivative

    # Derivative of the tanh function for the second neuron
    derivative = 1.0 - n_y[1]**2
    n_error[1] = n_w[2][2] * n_error[2] * derivative

# Function to adjust weights of neurons
def adjust_weights(x):
    global n_w
    # Adjust weights for the first two neurons
    n_w[0] -= (x * LEARNING_RATE * n_error[0])
    n_w[1] -= (x * LEARNING_RATE * n_error[1])

    # Prepare inputs for the third neuron (including bias)
    n2_inputs = np.array([1.0, n_y[0], n_y[1]])

    # Adjust weights for the third neuron
    n_w[2] -= (n2_inputs * LEARNING_RATE * n_error[2])

# Main function to run the neural network
def main():
    all_correct = False
    while not all_correct:
        all_correct = True
        # Randomize the order of training examples
        np.random.shuffle(index_list)
        for i in index_list:
            # Perform forward and backward passes and adjust weights
            forward_pass(x_train[i])
            backward_pass(y_train[i])
            adjust_weights(x_train[i])
            # Show updated weights after each adjustment
            show_learning()

        # Check if the predictions are correct for all training examples
        for i in range(len(x_train)):
            forward_pass(x_train[i])
            print(f'x1 = {x_train[i][1]:4.1f}, x2 = {x_train[i][2]:4.1f}, y = {n_y[2]:.4f}')
            # Determine if the prediction is correct (binary classification threshold at 0.5)
            if ((y_train[i] < 0.5 and n_y[2] >= 0.5) or (y_train[i] >= 0.5 and n_y[2] < 0.5)):
                all_correct = False

# Run the script
if __name__ == '__main__':
    main()
