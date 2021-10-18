import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

plt.style.use('prl')

start = 4
stop = 6
fns = []
labels = [r'$k_e/k_o = 1$', r'$k_e/k_o = 2$', r'$k_e/k_o = 1/2$']
for i in range(start,stop+1):
    fns.append('data/run_{:d}_output.h5'.format(i))
for fn, label in zip(fns, labels):
    with h5py.File(fn,'r') as df:
        plt.plot(df['scales/t'][:], df['energies/e_1'], label=label)

plt.legend()
plt.xlabel("time")
plt.ylabel(r"$E_1$")
plt.tight_layout()

plt.savefig("e1_vs_time_runs{:d}-{:d}.png".format(start, stop),dpi=300)
