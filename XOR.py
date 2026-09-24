#The XOR Problem
blurb='''
XOR: Exclusive OR
While things like AND gates can be predicted with one layer, the XOR problem
introduces non-linearity, requiring two layers. It's a good test to ensure
that the neural network is actually learning what we want.

I'm rounding the output results so that (if the network is trained well) the 
answer will round to ones and zeros. The thought process is that the more
effectively the network is trained, the closer the decimal value will be to
the desired output.

Poorly trained : XOR 1, 1 -> 0.44, XOR 1, 0 -> 0.51
Well trained   : XOR 1, 1 -> 0.15, XOR 1, 0 -> 0.89
(The output values above are random values. Getting a 0.44 or worse on one 
iteration may mean bad luck. Getting a 0.44 or worse on multiple means 
bad training)

XOR TRUTH TABLE
|In 1|In 2|Out |
+----+----+----+
|0   |0   |0   |
|0   |1   |1   |
|1   |0   |1   |
|1   |1   |0   |
+----+----+----+

If correctly executed, the program will output 0110. If training goes
poorly, it may result in a 1111, which doesn't match the truth table.
'''

non_liniarity_explanation='''
Nonlinearity Explanation:
In a normal AND gate or and OR gate, an AI can cheat the system by learning
what you don't want it to learn. In an AND gate, the truth table is as follows:

AND TRUTH TABLE
|In 1|In 2|Out |
+----+----+----+
|0   |0   |0   |
|0   |1   |0   |
|1   |0   |0   |
|1   |1   |1   |
+----+----+----+

So an AI can learn by not evaluating the inputs themselves, but rather counting
how many ones there are. If the amount of ones is more than one, it outputs 1.

The same is for an OR gate

OR TRUTH TABLE
|In 1|In 2|Out |
+----+----+----+
|0   |0   |0   |
|0   |1   |1   |
|1   |0   |1   |
|1   |1   |1   |
+----+----+----+

So if the amount of ones is one or greater, it outputs one.

'''
import numpy as np

from networkParts.network import Network
from networkParts.fcLayer import FCLayer
from networkParts.aLayer import ActivationLayer
from networkParts.aLayer import ActivationFunctions
from networkParts.lossFuncs import LossFunctions

#training data
x_train=np.array([ #input data. input shape = 4x1x2
    [[0,0]], 
    [[0,1]],
    [[1,0]], 
    [[1,1]]
])
y_train=np.array([ #expected output for every x_train index
    [[0]], 
    [[1]], 
    [[1]], 
    [[0]]
])

#network
net=Network()
net.add(FCLayer(2, 3)) #input size=2, output size=3
net.add(ActivationLayer(ActivationFunctions.tanh, ActivationFunctions.tanh_prime))
net.add(FCLayer(3,1)) #input size=3, output size=1
net.add(ActivationLayer(ActivationFunctions.tanh, ActivationFunctions.tanh_prime))

#train
net.use(LossFunctions.mse, LossFunctions.mse_prime) #set loss funcs
errors=net.train(x_train, y_train, 1000, 0.1, True)
#print(errors[-1])


print(blurb)
print("Tested output (Compare to the Out column of the truth table):")
#test
tests=1
for n in range(tests):
    out = net.predict(x_train, True)
    for i in out:
        print(round(i[0][0]), end=" ")
        print(i)
    print("")

def console_loop():
    val1=input("[XOR bot] In 1 : ")
    if "quit" in val1.lower():
        return "quit"
    elif "blurb" in val1.lower():
        return "blurb"
    elif "nonlin" in val1.lower():
        return "nonlin"

    val2=input("[XOR bot] In 2 : ")
    if "quit" in val2.lower():
        return "quit"
    elif "blurb" in val2.lower():
        return "blurb"
    elif "nonlin" in val2.lower():
        return "nonlin"
    
    out=net.predict([[int(val1), int(val2)]])
    print(f"[XOR bot] Out  : {round(out[0][0])}")
    print("")
    return True

running=True
print("Type quit to stop, blurb to see the description, and nonlin to see an explination of non-linearity.")
while running:
    ret=console_loop()
    if ret=="quit":
        running=False
        print("Goodbye.")
    elif ret=="nonlin":
        print(non_liniarity_explanation)
    elif ret=="blurb":
        print(blurb)
