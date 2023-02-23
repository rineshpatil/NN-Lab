import numpy as np
from Neurons import duPtron

# y contains augmented input patterns/training instances.
# Note: bias input for the neuron is fixed at 1
y = np.array([[0,0,0,1],[0,0,1,1],[0,1,0,1],[0,1,1,1],[1,0,0,1],[1,0,1,1],[1,1,0,1],[1,1,1,1]])
l=len(y)
# D contains desired outputs(labels) for the AND gate
# d = np.array([0,0,0,0,0,0,0,1])  # labels

# D contains desired outputs(labels) for the NOR gate
d = np.array([1,0,0,0,0,0,0,0])  # labels

# Maximum number of training cylces permitted
MaxCycles = 500

# initializing weights to random values from 0 to 1
a, b = 0, 1
w = (b - a) * np.random.randn(4) + a
wi = w  # storing initial weights
dup = duPtron(wi) # create a discrete unipolar perceptron object
# Training using Perceptron learning
dup.c = 1  # c is a learning constant
k = 0  # training cycle number
while (1):
    sse = 0  # intialize sum of squared error to zero in the begining of each training cycle
    for i in range(l):
        dup.x = y[i] # y[i] is neuron's augmented input vector(training instance)
        o = dup.out()
        et = np.power((d[i] - o), 2)  # et is squared error term
        sse = sse + et
        #print(d[i]-0)
        dw = dup.c * (d[i] - o) * dup.x  # Single Perceptron learning rule
    
        dup.w = dup.w + dw  # learning is taking place

    # One training cycle completed
   
    k = k + 1
    # terminate training when squared error reaches zero or k exceeds MaxCycles
    if (sse == 0 or k >= MaxCycles):  # either network learns or the algorithm or the max cycles exhusted
        print(f"\nTraining converged in {k} iterations with SSE={sse}")
        break

dup.w = np.round(dup.w, 2)  # rounding final weights to two decimal places
wi = np.round(wi, 2)
print(f"Intial weights were {wi}")
print(f"Final weights are {dup.w}")

# printing truth table for NOR gate using recall trained Perceptron model
print("\nTruth table for NOR gate generated using trained neuron is:")
for i in range(0, l):
    # Simulating the gate for each input x using final weights in w obtained during training
    dup.x = y[i]
    o = dup.out()
    print(dup.x, o)

