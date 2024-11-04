import math
import random
import matplotlib.pyplot as plt
import numpy as np 
import time
from sklearn.model_selection import train_test_split
import pandas as pd
import itertools

# Task 1: Generate a Complex Non-linear Dataset
def generate_complex_data(n_samples): #generates a complex dataset using given function, with gausian noise
    y = []
    step = 20 / (n_samples - 1) if n_samples > 1 else 0
    x = [-10 + i * step for i in range(n_samples)]
    for x_val in x:
        y_val = (0.1 * x_val ** 3) - (0.5 * x_val ** 2) + (0.2 * x_val) + 3 + 2 * math.sin(2 * x_val) + random.gauss(0, 10)
        y.append(y_val)
    return np.array(x).reshape(-1,1), np.array(y).reshape(-1,1) #returned dataset is converted to np array and reshaped for easier processing


def real_val(x_val): #calculates value of y as per X without noise
    return (0.1 * x_val ** 3) - (0.5 * x_val ** 2) + (0.2 * x_val) + 3 + 2 * np.sin(2 * x_val)


def rand_weight_bias(inputL, next_layer):  # Used to intialize weights and bias, Returns array of small random values for weights and an array of 0s for bias
    weights = []
    bias = []
    for i in range(inputL):
        by_in = [random.uniform(-0.01, 0.01) for j in range(next_layer)]
        weights.append(by_in)
    bias = [0 for _ in range(next_layer)]
    return np.array(weights), np.array(bias,dtype=np.float64)

# Task 3: Implement the Loss Function
def mean_squared_error(y_true, y_predicted): #calculate MSE
    return np.mean((y_true - y_predicted)**2)



def relu(vals): #return relu vals for array 
    return np.maximum(0, vals)



def diff_relu(vals):#return differntial relu vals for array
    return np.where(vals <= 0, 0, 1)



def tanh(vals): #return tanh vals for array
    return np.tanh(vals)



def diff_tanh(vals): #return differntial tan h vals for array
    return 1 - np.tanh(vals) ** 2

# Task 2: Define the Neural Network Architecture
class TwoLayerMLP: 
    def __init__(self,inp,h1,h2,out): #Initializes weights and biases using above rand_weight_bias function made above
        self.w_l1, self.bias_l1 = rand_weight_bias(inp, h1)
        self.w_l2, self.bias_l2 = rand_weight_bias(h1, h2)
        self.w_l3, self.bias_l3 = rand_weight_bias(h2, out)

    def forward(self, X): # forwards data, using matrix dot product to find sum values and relu for layer 1, tanh for layer 2 and identity function for layer 3 ie z3=a3
        self.z1 = np.dot(X, self.w_l1) + self.bias_l1
        self.a1 = relu(self.z1)
        self.z2 = np.dot(self.a1,self.w_l2) + self.bias_l2
        self.a2 = tanh(self.z2)
        self.z3 = np.dot(self.a2, self.w_l3) + self.bias_l3
        
        return self.z3 

    def backward(self, X, y, output, lr): # Backprop using chain rule
        batch_size=len(X) 
        err = output - y  #calculate error
        dW3 = np.dot(self.a2.T, err) / batch_size  #transpose a2 for matix dot, divide by batch size to get mean
        db3 = np.sum(err, axis=0) / batch_size #sums up vals in x axis, divided by batch size to get mean

        #same as above, but instead of identity function with err, we use layer 3 and tanh differential to backprop
        a2_error = np.dot(err, self.w_l3.T) *diff_tanh(self.z2)
        dW2 = np.dot(self.a1.T, a2_error) / batch_size
        db2 = np.sum(a2_error, axis=0) / batch_size

        #same as above, but instead of identity function with err, we use layer 2 and relu differential to backprop
        a1_error = np.dot(a2_error, self.w_l2.T) *diff_relu(self.z1)
        dW1 = np.dot(X.T, a1_error) / batch_size
        db1 = np.sum(a1_error, axis=0) / batch_size

        #update weights and biases as per learning rate
        self.w_l3 -= lr * dW3
        self.bias_l3 -= lr * db3
        self.w_l2 -= lr * dW2
        self.bias_l2 -= lr * db2
        self.w_l1 -= lr * dW1
        self.bias_l1 -= lr * db1

    #Task 3: Implement the Training Loop

    def train(self, X, y, epochs, lr, batch_size): #
        t=time.time()
        lossL = [] #make return  list of loss values for plotting
        y_actual = real_val(X) # for backprop

        for epoch in range(epochs): #randomise data fo each epoch
            dt1 = list(zip(X, y))
            np.random.shuffle(dt1)
            x_sh, y_sh = zip(*dt1)
            x_sh = np.array(x_sh)
            y_sh = np.array(y_sh)

            for i in range(0, len(X), batch_size): #divide data into minibatches, calculate prediction and update model using correct y via backprop
                x_batch = x_sh[i:i + batch_size]
                y_batch = y_sh[i:i + batch_size]
                output = self.forward(x_batch)
                self.backward(x_batch, y_batch, output, lr)

            epoch_loss = mean_squared_error(y, self.forward(X)) #plotting purposes
            lossL.append(epoch_loss)


            if epoch % 100 == 0: #show progress after 100 epochs
                print(f'current epoch:{epoch}, epoch loss: {epoch_loss:.2f}, current actual loss:{mean_squared_error(y_actual, self.forward(X)):.2f}')
            
            if time.time() - t > 60: #stop in 1 min
                print("Training stopped due to time constraints.")
                break

        return lossL

    def predict(self, X):
        return self.forward(X)

# Task 4: Visualize Training Results
def plotG(x, y, epochs, lr, batch_size, model): #makes and displays graphs

    # Split data into training and test sets
    X_train, X_test, y_train, y_test, y_train_actual, y_test_actual = train_test_split(
        x, y, real_val(x), test_size=0.2, random_state=42
    )

    # Train model and plot results
    lossL = model.train(X_train, y_train, epochs, lr, batch_size)
    y_pred = model.predict(x)

    # Plot of complex data
    plt.figure(figsize=(10,6))
    plt.scatter(x, y, s=10, alpha=0.5, label='Noisy Data')
    plt.title('Generated Complex Non-linear Dataset')
    plt.xlabel('X')
    plt.ylabel('y')
    plt.legend()
    plt.show()

    # Plot of Loss
    plt.figure(figsize=(10,6))
    plt.plot(lossL)
    plt.title("Loss during training")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.show()

    # Plot of dataset vs prediction
    plt.figure(figsize=(10,6))
    plt.scatter(x, y, color='red', s=10, alpha=0.5, label='Actual Data (with noise)')
    sorted_indices = X_train.flatten().argsort()
    plt.plot(x[sorted_indices], y_pred[sorted_indices], color='blue', label='Prediction')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()

# Task 5: Hyperparameter Tuning and Analysis
def hyperparameter_tuning(X_train, y_train, X_test, y_test, y_train_actual, y_test_actual):
    # Define hyperparameter options
    learning_rates = [0.0001, 0.001, 0.01]
    hidden_configs = [(128, 64), (256, 128), (64, 32)]

    # Store results
    results = []

    # Iterate over all combinations
    for lr, (h1, h2) in itertools.product(learning_rates, hidden_configs):
        print(f'Training MLP with Learning Rate: {lr}, Hidden Layers: ({h1}, {h2})')
        # Initialize model
        model = TwoLayerMLP(inp=1, h1=h1, h2=h2, out=1)
        # Train model
        loss = model.train(X_train, y_train, epochs=5000, lr=lr, batch_size=64)
        # Evaluate on training and test sets
        train_pred = model.predict(X_train)
        test_pred = model.predict(X_test)
        train_loss = mean_squared_error(y_train, train_pred)
        test_loss = mean_squared_error(y_test_actual, test_pred)  # Using noise-free for test loss

        # Store the results
        results.append({
            'Learning Rate': lr,
            'Hidden Layers': f'({h1}, {h2})',
            'Final Training Loss': train_loss,
            'Final Test Loss': test_loss,
            'Loss Curve': loss
        })

        print(f'Completed: LR={lr}, Hidden Layers=({h1}, {h2}) | Training Loss: {train_loss:.2f}, Test Loss: {test_loss:.2f}\n')

    return results

def plot_loss_curves(tuning_results):
    plt.figure(figsize=(12, 8))
    for result in tuning_results:
        label = f"LR={result['Learning Rate']}, Hidden={result['Hidden Layers']}"
        plt.plot(result['Loss Curve'], label=label)
    
    plt.title('Loss Curves for Different Hyperparameter Configurations')
    plt.xlabel('Epochs')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.show()

# Main Execution
if __name__ == "__main__":
    # Generate data
    x, y  = generate_complex_data(5000)
    
    # Split data into training and test sets
    X_train, X_test, y_train, y_test, y_train_actual, y_test_actual = train_test_split(
        x, y, real_val(x), test_size=0.2, random_state=42
    )
    
    # Initialize and train the default model
    default_MLP = TwoLayerMLP(inp=1, h1=128, h2=64, out=1)
    plotG(x, y, epochs=5000, lr=0.0005, batch_size=64, model=default_MLP)
    
    # Perform hyperparameter tuning
    tuning_results = hyperparameter_tuning(X_train, y_train, X_test, y_test, y_train_actual, y_test_actual)
    
    # Create DataFrame from results
    results_df = pd.DataFrame(tuning_results)
    
    # Display the results
    print("Hyperparameter Tuning Results:")
    print(results_df[['Learning Rate', 'Hidden Layers', 'Final Training Loss', 'Final Test Loss']])
    
    # Plot the loss curves
    plot_loss_curves(tuning_results)
    
    # Save the results table to a CSV file (optional)
    results_df.to_csv('hyperparameter_tuning_results.csv', index=False)
