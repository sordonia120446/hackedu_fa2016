"""
Feed-fwd neural network for machine learning using Sigmoid function. 

NB:  xrange() is deprecated in Python3+, but is much more memory-efficient for Python2.7.
Switch to range() for Python3+. 

@author: SamO

Created on 10/15/2016

"""

import numpy as np

def nonlin(x, deriv=False):
	"""
	Sigmund function
	"""
	if (deriv == True):
		return( x*(1-x) )

	return (1/( 1 + np.exp(-x) ) )


# input data
x = np.array([ [0,0,1], 
	[0,1,1],
	[1,0,1],
	[1,1,1] ])

# desired output
y = np.array( [ [0],[1],[1],[0] ] )

# seeding
np.random.seed(1) 

# make synapses
syn0 = 2*np.random.random( (3,4) ) - 1 # make 3x4 matrix of weights with a bias of 1; 3 nodes, 4 outputs
syn1 = 2*np.random.random( (4,1) ) - 1 # make 3x4 matrix of weights with a bias of 1; 4 nodes, 1 output

trainingSteps = 6*10**4
for j in xrange(trainingSteps):
	"""
	Reduce error of prediction per iteration by improving paths b/w neurons.  
	Training = updating weights through back-propogation. 
	The more training steps, the more accurate the results.  And the more you have to wait around.  
	"""
	# creating layers 
	l0 = x # input layer
	l1 = nonlin(np.dot(l0, syn0)) # "hidden" layer (also, dot --> matrix multiplication)
	l2 = nonlin(np.dot(l1, syn1)) # output layer; how likely is this output to be "correct"

	# back-propagation
	l2_error = y - l2
	if ( (j % 10000) == 0):
		print('Error:' +  str(np.mean( np.abs( l2_error ) ) ) )

	# calculate deltas
	l2_delta = l2_error*nonlin(l2, deriv=True)

	l1_error = l2_delta.dot(syn1.T) # transpose matrix with .T

	l1_delta = l1_error*nonlin(l1, deriv=True)

	# updating synapses (weights)
	syn1 += l1.T.dot(l2_delta)
	syn0 += l0.T.dot(l1_delta)



print('\nWe are done!')
print('Output should match desired y-values (more or less)\n')
print(l2)















