import math
import random
import matplotlib.pyplot as plt
import numpy as np 
import time

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
    # Initial implemenation without numpy, works but is slow
    # val = sum((y_true[i] - y_predicted[i]) ** 2 for i in range(len(y_true)))
    # k=val / len(y_true)
    # return k[0]
    return np.mean((y_true - y_predicted)**2)



def relu(vals): #return relu vals for array 

    # Initial implemenation without numpy, works but is slow
    # ansF=[]
    # for rows in vals:
    #     ans=[]
    #     for val in rows:
    #         ans.append(max(0, val))
    #     ansF.append(ans)
    # return np.array(ansF)

    #actual
    return np.maximum(0, vals)



def diff_relu(vals):#return differntial relu vals for array

    # Initial implemenation without numpy, works but is slow
    # ansF=[]
    # for rows in vals:
    #     ans=[]
    #     for val in rows:
    #         ans.append(0 if val <= 0 else 1)
    #     ansF.append(ans)
    # return np.array(ansF)


    return np.where(vals <= 0, 0, 1)



def tanh(vals): #return tanh vals for array

    # Initial implemenation without numpy, works but is slow
    # ansF=[]
    # for rows in vals:
    #     ans=[]
    #     for val in rows:
    #         ans.append(math.tanh(val))
    #     ansF.append(ans)
    # return np.array(ansF)

    return np.tanh(vals)
    


def diff_tanh(vals): #return differntial tan h vals for array

    # Initial implemenation without numpy, works but is slow
    # ansF=[]
    # for rows in vals:
    #     ans=[]
    #     for val in rows:
    #         ans.append(1 - math.tanh(val) ** 2)
    #     ansF.append(ans)
    # return np.array(ansF) 

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
                print(f'current epoch:{epoch}, Training loss: {epoch_loss:.2f}, Test loss:{mean_squared_error(y_actual, self.forward(X)):.2f}')
            
            if time.time() - t > 60: #stop in 1 min
                break

        return lossL

    def predict(self, X):
        return self.forward(X)

# Task 4: Visualize Training Results
def plotG(x, y, epochs, lr, batch_size,model): #makes and displays graphs

    # Train model and plot results
    lossL = model.train(x, y, epochs, lr, batch_size)
    y_pred = model.predict(x)

    #plot of complex data
    plt.scatter(x, y)
    plt.title('Generated Complex Non-linear Dataset')
    plt.show()

    # plot of Loss
    plt.plot(lossL)
    plt.title("Loss during training")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.show()

    # plot of dataset vs prediction
    plt.scatter(x, y, color='red', label='Actual Data (with noise)')
    plt.plot(x, y_pred, label='Prediction')
    plt.xlabel("X")
    plt.ylabel("y")
    plt.legend()
    plt.show()


# Task 5: Hyperparameter Tuning and Analysis
x, y  = generate_complex_data(5000)
MLP=TwoLayerMLP(1,256,128,1)
plotG(x,y, epochs=4100, lr=0.0001, batch_size=64, model=MLP)


'''
Hypermeter Analysis:
Variables => Hidden Layer neurons, Epochs, Learning rate, Batch size

Starting: epochs=4000, lr=0.00001, batch_size=64, Hidden layer=(128,64)

Methodology: changing variables independently with the starting vals being the base. See best performance and select variables to get the best possible output

Epochs values tried:
500, 1000, 1500, 2000, 4000 
Depends on Hidden layer,
with (128,64)
Major improvement from 100->1000, occasional improvement after 1000-2000, but sometimes becomes worse after. 
with

Learning Rate:
0.1, 0.01,0.001,0.0005,0.0001,0.00001
Overflowed on higher values 0.1, got terrible values for 0.01, 
After that it got slightly better each decrease, stating at 0.001, best being 0.00001

Batch Size:
64,32,16,128
Impacted how fast the epoch was, with higher batches increases making the algorithmn
Lower batch sizes had better prediction, but it was not linear as there was a lot of difference between 128 and 64, lesser from 64 to 32 but not a lot from 32 to 16 
I got the best performance from 32 instead of 16, as perhaps it was too slow and couldn't process enough epochs under a minute.

Hidden Layer:
(128,64),(256,64),(64,32)
Got more accurate with more higher hidden layer numbers, but became slower.
With no time constraints: (256,64) slightly better with 4000 epochs.
With time constraints: all performed pretty similarly

Hypermeter sets used for comparison:
1. epochs=2000, lr=0.00001, batch_size=32, Hidden layer=(128,64)
current epoch:2000, Training loss: 101.25, Test loss:7.12

2. epochs=1000, lr=0.001, batch_size=64, Hidden layer=(64,32)
current epoch:1000, Training loss: 236.90, Test loss:133.37


'''