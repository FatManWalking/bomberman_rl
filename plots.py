import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import make_interp_spline, BSpline

a = np.load("agent_code/vinnie_the_nextdoor_agent/round_rewards.npy")

T = np.arange(70_001)
xnew = np.linspace(0, 70_001, 1000)
spl = make_interp_spline(T, a, k=3)  # type: BSpline
power_smooth = spl(xnew)

plt.plot(xnew, power_smooth)
plt.ylabel("rewards")

plt.savefig("smooth_rewards.svg")
