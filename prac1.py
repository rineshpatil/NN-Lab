from Neurons import *

#define all input combinations for 3 binary inputs
X=[[0,0,0],[0,0,1],[0,1,0],[0,1,1],[1,0,0],[1,0,1],[1,1,0],[1,1,1]]

#Create mp neuron object with appropriate weights and a threshold for AND gate
w=[1,1,1]  # weights
th=2.5
mp1 = MPneuron(w, th)
#generate and print truth table for 3-input AND Gate
print("\nPrinting Truth Table for 3-input AND Gate using MP neuron:")
print("------------------")
print("Input        AND")
print("------------------")
for x in X:
    mp1.x=x
    o=mp1.out()
    print(f"{x}    {o}")
print("------------------")

#Create mp neuron object with appropriate weights and a threshold for OR gate
w=[1,1,1]  # weights
th=0.5
mp1 = MPneuron(w, th)
#generate and print truth table for 3-input OR Gate
print("\nPrinting Truth Table for 3-input OR Gate using MP neuron:")
print("------------------")
print("Input        OR")
print("------------------")
for x in X:
    mp1.x=x
    o=mp1.out()
    print(f"{x}    {o}")
print("------------------")

#Create mp neuron object with appropriate weights and a threshold for NAND gate
w=[-1,-1,-1]  # weights
th=-2.5
mp1 = MPneuron(w, th)
#generate and print truth table for 3-input NAND Gate
print("\nPrinting Truth Table for 3-input NAND Gate using MP neuron:")
print("------------------")
print("Input        NAND")
print("------------------")
for x in X:
    mp1.x = x
    o=mp1.out()
    print(f"{x}    {o}")
print("------------------")

#Create mp neuron object with appropriate weights and a threshold for NOR gate
w=[-1,-1,-1]  # weights
th=-0.5
mp1 = MPneuron(w, th)
#generate and print truth table for 3-input NOR Gate
print("\nPrinting Truth Table for 3-input NOR Gate using MP neuron:")
print("------------------")
print("Input        NOR")
print("------------------")
for x in X:
    mp1.x = x
    o=mp1.out()
    print(f"{x}    {o}")
print("------------------")

#Create mp neuron object with appropriate weights and a threshold for NOT gate
X1=[[0],[1]] # inputs
w=[-1]  # weights
th=0
mp1 = MPneuron(w,th)
#generate and print truth table for single-input NOT Gate
print("\nPrinting Truth Table for single-input NOT Gate using MP neuron:")
print("------------------")
print("Input    NOT")
print("------------------")
for x in X1:
    mp1.x = x
    o=mp1.out()
    print(f"{x}  \t  {o}")
print("------------------")