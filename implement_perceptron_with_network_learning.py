'''
Code from pages 9 and 10 of the book, "Learning Deep Learning"

Here we implement a network of weights, inputs, desired outputs and allow the model to train on the inputs and outputs
'''

import random

def show_learning(w):
    # This function prints the current weights of the perceptron in a formatted way.
    print('w0 =', '%5.2f' % w[0], ', w1 =', '%5.2f' % w[1], ', w2 =', '%5.2f' % w[2])

# Set a seed for the random number generator to make the results repeatable.
random.seed(7)

# Define the learning rate, a hyperparameter that controls how much the weights are adjusted.
LEARNING_RATE = 0.1

# Create a list of indices to access training examples in a randomized order.
index_list = [0, 1, 2, 3]

# Define the training inputs (x_train) and the expected outputs (y_train).
x_train = [(1.0, -1.0, -1.0), (1.0, -1.0, 1.0), (1.0, 1.0, -1.0), (1.0, 1.0, 1.0)]
y_train = [1.0, 1.0, 1.0, -1.0]

# Initialize the perceptron weights.
w = [0.2, -0.6, 0.25]

# Display the initial weights.
show_learning(w)

def compute_output(w, x):
    # This function computes the perceptron's output for a given input x and weights w.
    z = 0.0
    for i in range(len(w)):
        # Sum the product of each input value and its corresponding weight.
        z += x[i] * w[i]
    # Apply a sign function: if the sum is less than 0, return -1; otherwise, return 1.
    return -1 if z < 0 else 1

# Flag to check if all predictions are correct.
all_correct = False

# Training loop: continues until all predictions match the expected outputs.
while not all_correct:
    # Assume all predictions are correct initially.
    all_correct = True
    # Randomize the order of training examples.
    random.shuffle(index_list)
    for i in index_list:
        # Get the current training example.
        x = x_train[i]
        y = y_train[i]
        # Compute the perceptron's output.
        p_out = compute_output(w, x)

        # Check if the output matches the expected output.
        if y != p_out:
            # If not, update each weight.
            for j in range(len(w)):
                # Print current y and x[j] for understanding.
                print(f'y: {y}  x[j]: {x[j]}')
                # Adjust the weight based on the error, learning rate, and input value.
                w[j] += (y * LEARNING_RATE * x[j])
                # Print updated weight for understanding.
                print(f'w[j]: {w[j]}')
            # Since there was an incorrect prediction, set all_correct to False.
            all_correct = False
            # Show updated weights after each adjustment.
            show_learning(w)
