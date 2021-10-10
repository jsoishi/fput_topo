import h5py
import numpy as np
import matplotlib.pyplot as plt
from utils import calc_energy_err
plt.style.use('prl')
A = 1
N = 31
E0 = A*(N+1)*np.sin(np.pi/(2*(N+1)))**2
avg_time = 200
fn = ["fput_N31_A1_alpha0.25_dt0.015707963267948967.h5", "fput_N31_A1_alpha0.25_dt0.031415926535897934.h5","fput_N31_A1_alpha0.25_dt0.0628318530718.h5"]
fn_old = ["fput_N31_A1_alpha0.25_dt0.015707963267948967_old.h5", "fput_N31_A1_alpha0.25_dt0.031415926535897934_old.h5","fput_N31_A1_alpha0.25_dt0.0628318530718_old.h5"]
fn_nocorr = ["fput_N31_A1_alpha0.25_dt0.015707963267948967_nocorrection.h5", "fput_N31_A1_alpha0.25_dt0.031415926535897934_nocorrection.h5","fput_N31_A1_alpha0.25_dt0.0628318530718_nocorrection.h5"]
dt = [0.015707963267948967, 0.031415926535897934, 2*np.pi/100]

rel_err = []
rel_err_old = []
rel_err_noc = []
for filename in fn:
    with h5py.File(filename, "r") as df:
        rel_err.append(calc_energy_error(df, avg_time))

for filename in fn_old:
    with h5py.File(filename, "r") as df:
        rel_err_old.append(calc_energy_error(df, avg_time))

for filename in fn_nocorr:
    with h5py.File(filename, "r") as df:
        rel_err_noc.append(calc_energy_error(df, avg_time))
rel_err = np.array(rel_err)
rel_err_old = np.array(rel_err_old)
rel_err_noc = np.array(rel_err_noc)
dt = np.array(dt)
plt.loglog(dt, rel_err, 'x', label='SABA2C')
plt.loglog(dt, rel_err_old, '+', label='SABA2C old')
plt.loglog(dt, rel_err_noc, 'o', label='SABA2')
plt.loglog(dt, rel_err[0]*(dt/dt[0])**4, color='k', alpha=0.2, label=r'$dt^4$')
plt.loglog(dt, rel_err_old[0]*(dt/dt[0])**4, color='k', alpha=0.2)
plt.legend()
plt.xlabel("dt")
plt.ylabel("Relative Energy Error")
plt.tight_layout()
plt.savefig("rel_en_err_vs_dt.png", dpi=300)
