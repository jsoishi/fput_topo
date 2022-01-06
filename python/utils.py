import re
import numpy as np
import logging
import sys
import h5py

formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s :: %(message)s')
rootlogger = logging.root
rootlogger.setLevel(0)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel('INFO')
stdout_handler.setFormatter(formatter)
rootlogger.addHandler(stdout_handler)

class fput_run:
    def __init__(self, filename):
        self.fn = filename
        df = h5py.File(self.fn, "r")
        self.p = df['tasks/p'][:]
        self.q = df['tasks/q'][:]
        self.t = df['scales/t'][:]

        self.e_tot = df['energies/e_tot'][:]
        self.e_1 = df['energies/e_1'][:]
        self.e_2 = df['energies/e_2'][:]
        self.e_1_emode = df['energies/e_1_emode'][:]
        self.e_2_emode = df['energies/e_2_emode'][:]
        df.close()

def calc_energy_error(df, avg_time = None):
    E0 = df['energies/e_tot'][0]
    rel_en_err = np.abs(df['energies/e_tot']/E0 - 1)

    if avg_time:
        rel_en_err = rel_en_err[df['scales/t'][:] > avg_time].mean()
    return rel_en_err

def parse_filename(fn):
    parts = fn.split("_")[1:]

    args = {}
    for p in parts:
        s = re.search("([a-zA-Z]+)([.\d]+)",p)
        k = s.group(1)
        v = s.group(2)
        v = v.strip('.')
        try:
            if '.' in v:
                v = float(v)
            else:
                v = int(v)
        except ValueError:
            pass
        args[k] = v

    return args

def efreq(H):
    evals, evecs = np.linalg.eig(H)
    indx = np.argsort(evals)
    omega = np.sqrt(evals[indx])
    
    return omega, evecs[:,indx]

def linear_operator(N, k1=1, k2=0.5, pbc=False):

    off = np.zeros(N-1)
    off[::2] = k1
    off[1::2] = k2
    H_1d = -(np.diag(-(k1+k2)*np.ones(N)) + np.diag(off,k=1) + np.diag(off,k=-1))

    if pbc:
        H_1d[0,-1] = -1
        H_1d[-1,0] = -1
        
    return H_1d

def project_sin_mode(p, q, mode):
    """projects p, q onto a single mode.

    Assume discretized lattice such that
    x_n = n dx
    L = (N+1) dx
    dx = L/(N+1)

    then can project onto a mode amplitude via
    f_mode = <mode|f> = \int f(x) sin(mode pi x/L) dx = \sum_n f_n sin(mode pi n dx/(dx (N+1))) L/(N+1)

    and then returns the single mode in lattice space
    f_n = f_mode sin(mode pi n/(N+1))

    """
    N = len(q)-2
    n = np.arange(1,N+1)
    basis_fn = np.zeros_like(q)
    basis_fn[1:-1] = np.sin(mode*n*np.pi/(N+1))
    p_mode = 2*inner_product(p[1:-1], basis_fn[1:-1])/(N+1)
    q_mode = 2*inner_product(q[1:-1], basis_fn[1:-1])/(N+1)

    return p_mode*basis_fn, q_mode*basis_fn

def inner_product(a, b):
    return (a*b).sum()

def project_eigen_mode(p, q, mode, evecs):
    N = len(q) - 2
    basis_fn = np.zeros_like(q)
    n = mode - 1 # ground state is not constant
    basis_fn[1:-1] = evecs[:,n]
    
    p_mode = 2*inner_product(p[1:-1], basis_fn[1:-1])/(N+1)
    q_mode = 2*inner_product(q[1:-1], basis_fn[1:-1])/(N+1)

    return p_mode*basis_fn, q_mode*basis_fn

def eigenmode_transform(p, q, evecs):
    p_hat = evecs.T@p
    q_hat = evecs.T@q

    return p_hat, q_hat
        
