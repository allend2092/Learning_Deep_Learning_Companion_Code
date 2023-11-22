import  random

def show_learning(w):
    print('w0 =', '%5.2f' % w[0], ', w1 =', '%5.2f' % w[1], ', w2 =', '%5.2f' % w[2])

# define variables needed to control training process
random.seed(7) # To make repeatable
LEARNING_RATE = 0.1
index_list = [0,1,2,3] # Used to randomize order

# Define training examples
x_train = [(1.0,-1.0,-1.0),(1.0,-1.0,1.0),(1.0,1.0,-1.0),(1.0,1.0,1.0)] # inputs

# Ground truth
y_train = [1.0,1.0,1.0,-1.0]

# Define preceptron weights
w = [0.2, -0.6, 0.25]

# print initial weights
show_learning(w)

# Length of w and x must be n+1 for neuron with n inputs
def compute_output(w, x):
    z = 0.0
    for i in range(len(w)):
        z += x[i] * w[i] # Compute sum of weighted inputs
    if z < 0: # Apply sign function
        return -1
    else:
        return 1

# Perceptron training loop
all_correct = False
while not all_correct:
    all_correct = True
    random.shuffle(index_list) # randomize order
    for i in index_list:
        x = x_train[i]
        y = y_train[i]
        p_out = compute_output(w, x) # Perceptron function

        if y != p_out: # Update weights when wrong
            for j in range(0, len(w)):
                print(f'y: {y}  x[j]: {x[j]}')
                w[j] += (y * LEARNING_RATE * x[j])
                print(f'w[j]: {w[j]}')
            all_correct = False
            show_learning(w) # show updated weights




