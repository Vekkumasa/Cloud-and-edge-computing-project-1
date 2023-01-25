import numpy as np
import matplotlib.pyplot as plt

# Random values in matrices between 0 and 1
matrix_a = np.random.rand(10000, 1000)
matrix_b = np.random.rand(1000, 10000)
matrix_c = np.random.rand(10000, 1)

# Create matrix D where D=(A*B)*C.
matrix_d = np.matmul(np.matmul(matrix_a, matrix_b), matrix_c)

# cumulative distribution function (CDF) of all the values present in matrix A
# Matrix is collapsed as one dimensional matrix with flatten function
cdf = np.cumsum(np.sort(matrix_a.flatten())) / len(matrix_a.flatten())

# Plot the CDF
plt.plot(np.sort(matrix_a.flatten()), cdf)
plt.xlabel('matrix A values')
plt.ylabel('Cumulative Distribution Function')
plt.show()