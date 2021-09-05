import numpy as np
import matplotlib.pyplot as plt

data = np.random.normal(0, 0.2, size=(1000,4))
[plt.scatter(range(len(data)),data[:,i], s=1)
 for i in range(data.shape[1])]
# plt.xlim([-1,1])
plt.ylim([-1,1])
plt.show()