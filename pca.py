import numpy as np
import numpy.linalg as la

############
## Part 1 ##
############
# Read data
reader = open("data.txt")
data = np.genfromtxt(reader, delimiter=",")
reader.close()

data = np.matrix(data).transpose()
print "m = ", data.shape[0]
print "n = ", data.shape[1], "\n"

############
## Part 2 ##
############
# Compute mean
mean = np.mean(data, axis=1)
print "mean = \n", mean, "\n"

# Compute sdata
sdata = data - mean
sdata_mean = np.mean(sdata, axis=1)
print "sdata_mean = \n", sdata_mean, "\n"

# Compute covariance
cov = np.cov(sdata, bias=True)

# Compute eigendecomposition
eigenvalues, eigenvectors = la.eig(cov)
sorted_indices = np.argsort(eigenvalues)
print "eigenvalues = \n", eigenvalues
print "sorted_indices = \n", sorted_indices, "\n"

############
## Part 3 ##
############
# Compute total variance
total_variance = np.sum(eigenvalues)
print "total_variance = ", total_variance

# Compute ratio of accounted variance
eigenvalues_length = eigenvalues.shape[0]
variance_ratio = np.array([np.sum(eigenvalues[:(i+1)])/total_variance for i in xrange(eigenvalues_length)])
for i in xrange(eigenvalues_length):
	print "l = ", (i+1) , ", R(l) = ", variance_ratio[i]
print ""

# Construct Q
decreasing_indices = sorted_indices[::-1]
Q = np.zeros(eigenvectors.shape)
for i in xrange(eigenvalues_length):
	Q[:,i] = eigenvectors[:,decreasing_indices[i]]
print "Q = \n", Q, "\n"

# Compute vector y
# y = Q^T * (x - mean)
x = np.array([0.21, 0.72, 0.06, 0.36, -0.12, 0.04, 0.00, 0.46, 0.27, 0.59, 0.70])
sx = x - np.array(mean).flatten()
y = np.dot(sx, Q)
print "x = \n", x
print "y = \n", y, "\n"

# Truncate eigen matrix Q for l from 1 to m, 
# and compute respective y_hat with given x
for i in xrange(eigenvalues_length):
	Q_hat = Q[:,:(i+1)]
	y_hat = np.dot(sx, Q_hat)
	print "l = ", (i+1), ", y_hat = \n", y_hat
print ""

# Truncate eigen matrix Q for l from 1 to m, 
# and compute sum squared error on dataset
for i in xrange(eigenvalues_length):
	Q_hat = Q[:,:(i+1)]
	y_hats = np.dot(Q_hat.transpose(), sdata)
	x_hats = np.dot(Q_hat, y_hats) + mean
	norm = la.norm(x_hats - data)
	error = norm * norm
	print "l = ", (i+1), ", E = ", error
