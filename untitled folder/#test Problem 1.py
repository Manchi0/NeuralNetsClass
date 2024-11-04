# Problem 1

# Import necessary libraries
import random
from mat4py import loadmat  # To load MATLAB .mat files
import matplotlib.pyplot as plt  # For plotting data and decision boundaries

# Load datasets from .mat files
dt1_1 = loadmat('/Users/manasmunjial/Downloads/Two_moons_no_overlap.mat')
dt1_2 = loadmat('/Users/manasmunjial/Downloads/Two_moons_overlap2.mat')

dt2_1 = loadmat('/Users/manasmunjial/Downloads/Two_moons_no_overlap2.mat')
dt2_2 = loadmat('/Users/manasmunjial/Downloads/Two_moons_overlap3.mat')

def TrainTest_split_pro(x, y, ratio=0.3):
    """
    Splits the dataset into training and testing sets based on the given ratio.
    
    Parameters:
    x (list): Feature vectors.
    y (list): Corresponding labels.
    ratio (float): Proportion of data to be used as the test set.
    
    Returns:
    Tuple containing training and testing feature vectors and labels.
    """
    # Combine features and labels into a single list of tuples
    joined = list(zip(x, y))
    # Shuffle the combined list to ensure random distribution
    random.shuffle(joined)
    # Calculate the number of training samples
    t = len(joined) - int(len(joined) * ratio)

    # Split into training and testing data
    train = joined[:t]
    test = joined[t:]

    # Unzip the training and testing data back into separate feature and label lists
    x_train, y_train = zip(*train)
    x_test, y_test = zip(*test)

    return list(x_train), list(y_train), list(x_test), list(y_test)

# Extract features and labels from the loaded datasets
x11 = dt1_1['X']
y11 = dt1_1['Y']

x12 = dt1_2['X']
y12 = dt1_2['Y']

# Split the first dataset into training and testing sets
x11_train, y11_train, x11_test, y11_test = TrainTest_split_pro(x11, y11)
# Split the second dataset into training and testing sets
x12_train, y12_train, x12_test, y12_test = TrainTest_split_pro(x12, y12)

# Set the learning rate for weight updates
learning_rate = 0.01

def model(input, hidden):
    """
    Initializes the weights and biases for the neural network.
    
    Parameters:
    input (int): Number of input features.
    hidden (int): Number of hidden neurons.
    
    Returns:
    Tuple containing the initialized weights and biases.
    """
    weights = []
    bias = []
    # Initialize weights with small random values
    for i in range(input):
        by_in = []
        for j in range(hidden):
            # Random weight between 0.0001 and 0.0099
            by_in.append(random.randrange(1, 100) * 0.0001)
        weights.append(by_in)
    # Initialize biases with small random values, one extra for the bias term
    for k in range(hidden + 1):
        bias.append(random.randrange(1, 100) * 0.0001)

    return weights, bias

def activation(v):
    """
    Activation function that maps input to binary output.
    
    Parameters:
    v (float): Input value.
    
    Returns:
    int: 1 if v >= 0, else -1.
    """
    if v >= 0:
        return 1
    elif v < 0:
        return -1

def andNN(z_o):
    """
    Simulates an AND gate using the outputs from neurons.
    
    Parameters:
    z_o (list): List of neuron outputs.
    
    Returns:
    int: 1 if all neuron outputs are active, else -1.
    """
    b1 = 1  # Bias term
    # Sum the bias and neuron outputs
    y_in = b1 + z_o[0] + z_o[1] + z_o[2]
    # If the sum is 4 (indicating all neurons are active), output 1, else -1
    if y_in == 4:
        y_f = 1
    else:
        y_f = -1
    return y_f

def delta_update_weight1(learning_rate, yT, weights, z_in, neuron_index, x_val, bias):
    """
    Updates the weights and bias for a single neuron using the delta rule.
    
    Parameters:
    learning_rate (float): Learning rate for weight updates.
    yT (int): Target output.
    weights (list): Current weights.
    z_in (list): Input to neurons before activation.
    neuron_index (int): Index of the neuron to update.
    x_val (list): Input feature vector.
    bias (float): Current bias value for the neuron.
    
    Returns:
    Tuple containing updated weights and bias.
    """
    for i in range(len(weights)):
        # Update each weight connected to the neuron
        weights[i][neuron_index] = weights[i][neuron_index] + learning_rate * (yT - z_in[neuron_index]) * x_val[i]
    # Update the bias for the neuron
    bias = bias + learning_rate * (yT - z_in[neuron_index])
    return weights, bias

def delta_update_weightall(learning_rate, yT, weights, z_in, x_val, bias):
    """
    Updates the weights and biases for all neurons using the delta rule.
    
    Parameters:
    learning_rate (float): Learning rate for weight updates.
    yT (int): Target output.
    weights (list): Current weights.
    z_in (list): Input to neurons before activation.
    x_val (list): Input feature vector.
    bias (list): Current biases for all neurons.
    
    Returns:
    Tuple containing updated weights and biases.
    """
    for neuron_index in range(len(z_in)):
        for i in range(len(weights)):
            # Update each weight connected to the neuron
            weights[i][neuron_index] = weights[i][neuron_index] + learning_rate * (yT - z_in[neuron_index]) * x_val[i]
        # Update the bias for the neuron
        bias[neuron_index] = bias[neuron_index] + learning_rate * (yT - z_in[neuron_index])
    return weights, bias

def testing(wt, x_test, y_test, bias):
    """
    Evaluates the model on the test dataset and counts the number of errors.
    
    Parameters:
    wt (list): Current weights of the network.
    x_test (list): Test feature vectors.
    y_test (list): True labels for the test set.
    bias (list): Current biases for all neurons.
    
    Returns:
    int: Total number of classification errors on the test set.
    """
    errors = 0
    z_i = [None, None, None]  # Inputs to hidden neurons
    z_o = [None, None, None]  # Outputs from hidden neurons

    for i in range(len(x_test)):
        # Compute the input to each hidden neuron
        for j in range(len(z_i)):
            z_i[j] = bias[j] + x_test[i][0] * wt[0][j] + x_test[i][1] * wt[1][j]
            z_o[j] = activation(z_i[j])  # Apply activation function

        # Compute the final output using the AND gate
        y_f = andNN(z_o)

        # Compare the predicted label with the true label
        if y_f != y_test[i][0]:
            errors += 1

    return errors

def training(x, y, x_test, y_test, epochs, wt, bias):
    """
    Trains the neural network over a specified number of epochs.
    
    Parameters:
    x (list): Training feature vectors.
    y (list): Training labels.
    x_test (list): Test feature vectors.
    y_test (list): Test labels.
    epochs (int): Number of training epochs.
    wt (list): Initial weights.
    bias (list): Initial biases.
    
    Returns:
    Tuple containing a list of error counts per epoch and the updated weights.
    """
    all_error = []  # To store the number of errors at each epoch
    z_i = [None, None, None]  # Inputs to hidden neurons
    z_o = [None, None, None]  # Outputs from hidden neurons

    for epoch in range(epochs):
        for i in range(len(x)):
            # Compute the input to each hidden neuron for the current training example
            for j in range(len(z_i)):
                z_i[j] = bias[j] + x[i][0] * wt[0][j] + x[i][1] * wt[1][j]
                z_o[j] = activation(z_i[j])  # Apply activation function

            # Compute the final output using the AND gate
            y_f = andNN(z_o)

            # If the prediction is incorrect, update the weights and biases
            if y_f != y[i][0]:
                # Check if any neuron input is within a certain threshold
                if min(z_i) < 0.25 and min(z_i) > -0.25:
                    # Find the neuron with the smallest absolute input value
                    z_abs = [abs(ele) for ele in z_i]
                    sorted_abs = sorted(z_abs)
                    for lo in range(len(sorted_abs)):
                        l = z_abs.index(sorted_abs[lo])
                        if z_i[l] < 0.25 and z_i[l] > -0.25:
                            # Flip the neuron's output and check if it corrects the prediction
                            z_o[l] *= -1
                            y_fu = andNN(z_o)
                            if y_fu == y[i][0]:
                                # If correct, update weights and bias for that neuron
                                wt, bias[l] = delta_update_weight1(learning_rate, y[i][0], wt, z_i, l, x[i], bias[l])
                            else:
                                # If not correct, revert the change
                                z_o[l] *= -1
                else:
                    # If no neuron input is within the threshold, update all weights and biases
                    wt, bias = delta_update_weightall(learning_rate, y[i][0], wt, z_i, x[i], bias)

        # After each epoch, evaluate the model on the test set
        sum_error = testing(wt, x_test, y_test, bias)
        all_error.append(sum_error)

    return all_error, wt

def decisionL(x, y, wt):
    """
    Plots the decision boundaries and data points.
    
    Parameters:
    x (list): Feature vectors.
    y (list): Corresponding labels.
    wt (list): Weights of the network.
    """
    # Extract labels as a flat list
    ny = [yi[0] for yi in y]

    # Separate the data points based on their class labels
    group1 = [xi for xi, yi in zip(x, ny) if yi == 1]
    groupN1 = [xi for xi, yi in zip(x, ny) if yi == -1]

    # Create a new figure for plotting
    plt.figure(figsize=(8, 6))

    # Extract the first feature for plotting decision boundaries
    x_feature = [xi[0] for xi in x]
    min_x0 = min(x_feature)
    max_x0 = max(x_feature)

    num_points = 100  # Number of points to plot the decision boundary
    if num_points == 1:
        x_values = [min_x0]
    else:
        step = (max_x0 - min_x0) / (num_points - 1)
        x_values = [min_x0 + step * i for i in range(num_points)]

    num_weights = len(wt[0])  # Number of hidden neurons

    # Plot the decision boundary for each hidden neuron
    for n in range(num_weights):
        w0 = wt[0][n]  # Weight for the first feature
        w1 = wt[1][n]  # Weight for the second feature

        if w1 != 0:
            # Compute corresponding y values for the decision boundary
            y_values = [-(w0 * xv) / w1 for xv in x_values]
        else:
            # Handle the case where w1 is zero to avoid division by zero
            y_values = [-(w0 * xv) / 1e-8 for xv in x_values]

        # Plot the decision boundary line
        plt.plot(x_values, y_values, label=f'Line {n+1}')

    # Extract x and y coordinates for class 1
    group1_x = [xi[0] for xi in group1]
    group1_y = [xi[1] for xi in group1]

    # Extract x and y coordinates for class -1
    groupN1_x = [xi[0] for xi in groupN1]
    groupN1_y = [xi[1] for xi in groupN1]

    # Plot the data points for each class
    plt.scatter(group1_x, group1_y, color='blue', label='Class 1', edgecolor='k')
    plt.scatter(groupN1_x, groupN1_y, color='red', label='Class -1', edgecolor='k')
    
    # Set plot labels and title
    plt.xlabel('F1')
    plt.ylabel('F2')
    plt.title('Decision Boundaries')
    plt.legend()
    plt.grid(True)
    plt.show()

def Graphs(x_train, y_train, x_test, y_test, wt, bias):
    """
    Trains the model, plots error rates, and visualizes decision boundaries.
    
    Parameters:
    x_train (list): Training feature vectors.
    y_train (list): Training labels.
    x_test (list): Test feature vectors.
    y_test (list): Test labels.
    wt (list): Initial weights.
    bias (list): Initial biases.
    """
    # Train the model for 20 epochs and get the error rates
    all_error, wts = training(x_train, y_train, x_test, y_test, 20, wt, bias)
    
    # Print the final weights and average error rate
    print('Weights: ', wts)
    print('Error Rate: ', (sum(all_error) / len(all_error)))

    # Plot the error rates over epochs
    plt.figure()
    plt.plot(range(len(all_error)), all_error)
    plt.xlabel("Epochs")
    plt.ylabel("Errors")
    plt.title("Error Rate over Epochs")
    plt.show()

    # Plot the decision boundaries using the trained weights
    decisionL(x_test, y_test, wts)

# Initialize weights and biases for a network with 2 input features and 3 hidden neurons
wts, bi = model(2, 3)

# Train and visualize the model on the first dataset (no overlap)
Graphs(x11_train, y11_train, x11_test, y11_test, wts, bi)

# Reinitialize weights and biases before training on the second dataset
wts, bi = model(2, 3)

# Train and visualize the model on the second dataset (with overlap)
Graphs(x12_train, y12_train, x12_test, y12_test, wts, bi)
