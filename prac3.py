import numpy as np
from Neurons import dbPtron
# defining class prototypes
x1 = np.array([ 0, 0, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 1, 1, 1, 1]) #1 pattern
x2 = np.array([ 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 1, 0, 0, 1]) #0 pattern

# Finding classifier weights(desigining perceptron classifer)
w=x1-x2
w[25]=0.5*((np.linalg.norm(x2)**2-np.linalg.norm(x1)**2))
print(f'\nThe weights for the single perceptron classifiers are:{w}')

# create a discrete bipolar perceptron
dbp=dbPtron(w)

# Take a noisy/new test pattern Xt
dbp.x=[ 0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 1] #plus pattern

# Perform Classification
if dbp.out()>0:
    print('Given pattern is One 1.')
else:
    print('Given pattern is Zero 0.')


