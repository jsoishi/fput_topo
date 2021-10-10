import re
import numpy as np
import logging
import sys

formatter = logging.Formatter('%(asctime)s %(name)s %(levelname)s :: %(message)s')
rootlogger = logging.root
rootlogger.setLevel(0)
stdout_handler = logging.StreamHandler(sys.stdout)
stdout_handler.setLevel('INFO')
stdout_handler.setFormatter(formatter)
rootlogger.addHandler(stdout_handler)


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
