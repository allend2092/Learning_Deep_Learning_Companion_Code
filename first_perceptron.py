'''
Code from page 4, chapter 1: The Rosenblatt perceptron
'''

# Import necessary libraries (if any)

# Define your classes here (if needed)
class SampleClass:
    def __init__(self, attribute):
        self.attribute = attribute

    def sample_method(self):
        return f"This is a sample method returning: {self.attribute}"

# Define other functions here
def sample_function():
    return "This is a sample function."

# deep learning page 4: First element in vector x must be 1
# Length of w and x must be n+1 for neuron with n inputs
def compute_output(w, x):
    z = 0.0
    for i in range(len(w)):
        z += x[i] * w[i] # Compute sum of weighted inputs
    if z < 0: # Apply sign function
        return -1
    else:
        return 1

# Main function to drive the program
def main():
    # Example of using the SampleClass
    # sample_obj = SampleClass("Sample Attribute")
    # print(sample_obj.sample_method())

    # Example of using the sample_function
    # print(sample_function())
    output1 = compute_output([0.9, -0.6, -0.5], [1.0, -1.0, -1.0])
    output2 = compute_output([0.9, -0.6, -0.5], [1.0, -1.0, 1.0])
    output3 = compute_output([0.9, -0.6, -0.5], [1.0, 1.0, -1.0])
    output4 = compute_output([0.9, -0.6, -0.5], [1.0, 1.0, 1.0])

    print(output1)
    print(output2)
    print(output3)
    print(output4)


# This is a standard boilerplate that calls the main() function.
if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"An error occurred: {e}")
