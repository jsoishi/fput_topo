import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt

from utils import calc_energy_error, parse_filename
plt.style.use('prl')

fn = sys.argv[-1]
params = parse_filename(fn)
A = params['A']
N = params['N']
E0 = A*(N+1)*np.sin(np.pi/(2*(N+1)))**2

df = h5py.File(fn, "r")
t = df['scales/t'][:]
err = calc_energy_error(df, E0)
plt.semilogx(t,err)
plt.xlabel("time")
plt.ylabel("Relative Energy Error")
plt.tight_layout()
plt.savefig('{}_err_vs_t.png'.format(fn.strip('.h5')),dpi=300)
