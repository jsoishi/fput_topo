import matplotlib.pyplot as plt
from pathlib import Path
import h5py
import sys
import utils
plt.style.use('prl')

fn = Path(sys.argv[-1])
data = utils.fput_run(fn)
nmodes = data.p.shape[1]
data_extent = [data.t[0],data.t[-1], 1, nmodes-2]

plt.subplot(211)
plt.imshow(data.p[:,1:-1].T, extent=data_extent, interpolation='none')
plt.xlabel("time")
plt.ylabel("mass")
plt.title("p")
plt.subplot(212)
plt.imshow(data.q[:,1:-1].T, extent=data_extent, interpolation='none')
plt.xlabel("time")
plt.ylabel("mass")
plt.title("q")
plt.tight_layout()
plt.savefig('../figs/{}_pq_vs_t.png'.format(fn.stem),dpi=300)

