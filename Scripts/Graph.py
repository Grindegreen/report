import numpy as np
import matplotlib.pyplot as plt


file_name = "data.thc"
data = np.genfromtxt(file_name, skip_header=0)
x = data[:, 0]
y = data[:, 1]

plt.ylabel("Frequency, Hz")
plt.xlabel("Time, days")
plt.xlim(35, max(x) + 1)
plt.ylim(0.03, 0.11)
plt.grid(True)
plt.title("Evolution of the QPO frequency in MAX J1820+070")
plt.errorbar(x, y, fmt='ro', capsize=0, markersize=5)
plt.show()