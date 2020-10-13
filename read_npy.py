import numpy as np
import matplotlib.pyplot as plt

data = np.load("r_log.npy")
plt.plot(data)
plt.show()