import re
import numpy as np

def calc_energy_error(df, E0, avg_time = None):
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
