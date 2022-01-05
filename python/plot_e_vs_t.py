import h5py
import matplotlib.pyplot as plt
import numpy as np
import sys

plt.style.use('prl')

skip = 100
start = 4
stop = 6
alpha = 0.7

fns = []
labels = [r'$k_e/k_o = 1$', r'$k_e/k_o = 0.5$', r'$k_e/k_o = 2$']
for i in range(start,stop+1):
    fns.append('data/run_{:d}_output.h5'.format(i))
for fn, label in zip(fns, labels):
    with h5py.File(fn,'r') as df:
        print("E1/E1_e(t=0) = {:15.2e}".format(df['energies/e_1'][0]/(15*15/2*df['energies/e_1_emode'][0]/2)))
        plt.subplot(221)
        plt.plot(df['scales/t'][::skip], df['energies/e_1'][::skip], label=label, alpha=alpha)
        plt.xlabel("time")
        plt.ylabel(r"$E_1$")        
        plt.subplot(222)
        plt.plot(df['scales/t'][::skip], df['energies/e_2'][::skip], label=label, alpha=alpha)
        plt.xlabel("time")
        plt.ylabel(r"$E_2$")

        plt.subplot(223)
        plt.plot(df['scales/t'][::skip], 15*15/2*df['energies/e_1_emode'][::skip]/2, label=label, alpha=alpha)
        plt.xlabel("time")
        plt.ylabel(r"$E_1^{e}$")

        plt.subplot(224)
        plt.plot(df['scales/t'][::skip], df['energies/e_2_emode'][::skip], label=label, alpha=alpha)
        plt.xlabel("time")
        plt.ylabel(r"$E_2^{e}$")

plt.legend()
plt.tight_layout()

plt.savefig("e_vs_time_runs{:d}-{:d}.png".format(start, stop),dpi=300)
plt.close()

for fn, label in zip(fns, labels):
    with h5py.File(fn,'r') as df:
        t = df['scales/t'][:]
        dt = t[1]-t[0]
        freq  = np.fft.rfftfreq(len(t),d=dt)
        e1= df['energies/e_1'][:]
        e1_spec = np.abs(np.fft.rfft(e1))

        plt.loglog(freq[1:],e1_spec[1:], label=label, alpha=alpha)

plt.xlabel(r"$\omega$")
plt.ylabel(r"$|\hat{E}_1|^2$")
plt.legend()
plt.tight_layout()
plt.savefig("e1_spectrum_runs{:d}-{:d}.png".format(start, stop),dpi=300)
