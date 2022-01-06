"""fput_test.py

A simple script to compute the SSH FPUT problem using the SABA2C method outlined in 

Pace & Campbell, Chaos 29, 023132 (2019)

adapted to include spring constants.

Usage:
    fput_test.py <config>

"""
import numba as nb
from numba.experimental import jitclass
import time
import sys
from pathlib import Path
from configparser import ConfigParser
import numpy as np
import h5py
import utils
import logging

logger = logging.getLogger(__name__)

spec = [
    ('chi', nb.float64),
    ('u', nb.int32),
    ('u_exp', nb.int32),
    ('correction', nb.boolean),
    ('g', nb.float64),
    ('c1', nb.float64),
    ('c2', nb.float64),
    ('d1', nb.float64),
    ('k', nb.float64[:])
    ]
@jitclass(spec)
class FPUT(object):
    def __init__(self, chi, k=None, u=3, correction=True):
        self.g = (2-np.sqrt(3))/24.
        self.c1 = 0.5*(1 - 1/np.sqrt(3))
        self.c2 = 1/np.sqrt(3)
        self.d1 = 0.5

        self.chi = chi
        self.u = u
        self.u_exp = u - 1
        self.correction = correction
        self.k = k
    def step(self, p,q, dt):
        if self.correction:
            p,q = self.C_stage(p,q, dt)
        p,q = self.A_stage(p,q, dt, self.c1)
        p,q = self.B_stage(p,q, dt)
        p,q = self.A_stage(p,q, dt, self.c2)
        p,q = self.B_stage(p,q, dt)
        p,q = self.A_stage(p,q, dt, self.c1)
        if self.correction:
            p,q = self.C_stage(p,q, dt)

        assert (p[0] == 0)
        assert (p[-1] == 0)
        assert (q[0] == 0)
        assert (q[-1] == 0)

        return p,q

    def A_stage(self, p, q, dt, c):
        tau = c*dt
        q_out = q + tau*p
        return p, q_out

    def B_stage(self, p, q, dt):

        tau = self.d1*dt
        
        p_out = p
        dq = q[1:] - q[0:-1]
        p_out[1:-1] = p[1:-1] + tau*(self.k[1:]*(dq[1:] + self.chi*dq[1:]**(self.u_exp)) - self.k[0:-1]*(dq[0:-1] + self.chi*dq[0:-1]**(self.u_exp)))

        return p_out, q
 
    def C_stage(self, p, q, dt):
        p_out = p
        two_tau = -self.g * dt**3

        k2n = self.k[2:-1]
        k2np1 = self.k[3:]
        k2nm1 = self.k[1:-2]
        k2nm2 = self.k[0:-3]

        q2n = q[2:-2]
        q2nm1 = q[1:-3]
        q2nm2 = q[0:-4]
        q2np1 = q[3:-1]
        q2np2 = q[4:]
        
        p_out[1] = p[1] \
            -two_tau*(k[0]*(-(self.u_exp*self.chi*q[1]**(self.u-2))-1)+k[1]*(-(self.u_exp*self.chi*(q[2]-q[1])**(self.u-2))-1))*(k[0]*(-self.chi*q[1]**self.u_exp-q[1])+k[1]*(self.chi*(q[2]-q[1])**self.u_exp-q[1]+q[2])) \
            -two_tau*k[1]*(self.u_exp*self.chi* (q[2]-q[1])**(self.u-2)+1)*(k[1]*(-self.chi*(q[2]-q[1])**self.u_exp+q[1]-q[2])+k[2]*(self.chi*(q[3]-q[2])**self.u_exp-q[2]+q[3]))

        p_out[2:-2] = p[2:-2] \
-two_tau*k2nm1*(self.u_exp*self.chi*(q2n-q2nm1)**(self.u-2)+1)*(k2nm2*(-self.chi*(q2nm1-q2nm2)**self.u_exp+q2nm2-q2nm1)+k2nm1*(self.chi*(q2n-q2nm1)**self.u_exp-q2nm1+q2n))\
-two_tau*(k2nm1*(-(self.u_exp*self.chi*(q2n-q2nm1)**(self.u-2))-1)+k2n*(-(self.u_exp*self.chi*(q2np1-q2n)**(self.u-2))-1))*(k2nm1*(-self.chi*(q2n-q2nm1)**self.u_exp+q2nm1-q2n)+k2n*(self.chi*(q2np1-q2n)**self.u_exp-q2n+q2np1))\
-two_tau*k2n*(self.u_exp*self.chi*(q2np1-q2n)**(self.u-2)+1)*(k2n*(-self.chi*(q2np1-q2n)**self.u_exp+q2n-q2np1)+k2np1*(self.chi*(q2np2-q2np1)**self.u_exp-q2np1+q2np2))

        p_out[-2] = p[-2] \
            -two_tau*k[-3]*(self.u_exp*self.chi*(q[-2]-q[-3])**(self.u-2)+1)*(k[-4]*(-self.chi*(q[-3]-q[-4])**self.u_exp+q[-4]-q[-3])+k[-3]*(self.chi*(q[-2]-q[-3])**self.u_exp-q[-3]+q[-2])) \
            -two_tau*(k[-2]*(-(self.u_exp*self.chi*(-q[-2])**(self.u-2))-1)+k[-3]*(-(self.u_exp*self.chi*(q[-2]-q[-3])**(self.u-2))-1))*(k[-2]*(self.chi*(-q[-2])**self.u_exp-q[-2])+k[-3]*(-self.chi*(q[-2]-q[-3])**self.u_exp+q[-3]-q[-2]))


        return p_out, q

def calc_mode_energy(p, q, k, mode, alpha, eigenmode=None):

    if eigenmode is not None:
        p_mode, q_mode = utils.project_eigen_mode(p,q,mode, eigenmode)
    else:
        p_mode, q_mode = utils.project_sin_mode(p,q,mode)
                 
    return hamiltonian(p_mode, q_mode, k, alpha)

@nb.jit(nopython=True)
def hamiltonian(p, q, k, alpha):
    pn = p[0:-1]
    qn = q[0:-1]
    qnp1 = q[1:]
    
    H = 0.5* (pn**2).sum() + (0.5*k*(qnp1 - qn)**2).sum() + (k*alpha/3 * (qnp1 - qn)**3).sum()

    return H

if __name__ == "__main__":
    filename = Path(sys.argv[-1])
    outbase = Path("data")

    config = ConfigParser()
    config.read(str(filename))
    logger.info('Running fput_test.py with the following parameters:')
    logger.info(config.items('parameters'))
    logger.info(config.items('solver'))

    correction=config.getboolean('solver', 'correction')
    time_reversal = config.getboolean('solver','time_reversal')
    N = config.getint('solver','N')
    dt = config.getfloat('solver', 'dt')
    t_stop = config.getfloat('solver', 't_stop')
    analysis_cadence = config.getint('solver', 'analysis_cadence')

    ic_type = config.get('parameters', 'ic_type')
    mode = config.getint('parameters', 'mode')
    alpha = config.getfloat('parameters', 'alpha')
    beta  = config.getfloat('parameters', 'beta')
    ke = config.getfloat('parameters', 'ke')
    ko = config.getfloat('parameters', 'ko')
    A = 1
    cadence = 100

    output_file_name = Path(filename.stem + '_output.h5')
    
    n = np.arange(1,N+1)
    q = np.zeros(N+2,dtype=np.float64)
    p = np.zeros(N+2,dtype=np.float64)
    k = np.ones(N+1,dtype=np.float64)
    # implement SSH chain
    k[::2] = ke #even
    k[1::2] = ko #odd
    fput = FPUT(alpha, correction=correction, k=k)

    # initial conditions
    logger.info("Running with {} initial conditions, mode = {}".format(ic_type, mode))

    H = utils.linear_operator(N, k1=ko, k2=ke)
    omega, evecs = utils.efreq(H)

    if ic_type == 'eigenmode':
        q[1:-1] = evecs[:,mode]
    elif ic_type == 'sin':
        q[1:-1] = A*np.sin(np.pi*mode*n/(N+1))
    elif ic_type == 'sinp':
        p[1:-1] = A*np.sin(np.pi*mode*n/(N+1))
    else:
        q[1:-1] = A*np.sin(np.pi*n/(N+1))

    # data

    e_1 = [calc_mode_energy(p,q,k,1,alpha)]
    e_2 = [calc_mode_energy(p,q,k,2,alpha)]
    e_1_emode = [calc_mode_energy(p,q,k,1,alpha,eigenmode=evecs)]
    e_2_emode = [calc_mode_energy(p,q,k,2,alpha,eigenmode=evecs)]

    e_tot = [hamiltonian(p,q,k, alpha)]
    logger.info("E init = {:7.5f}".format(e_tot[0]))
    logger.info("e_1[0] = {}".format(e_1[0]))

    t_current = 0.
    iteration = 1
    p_list = [p.copy()]
    q_list = [q.copy()]
    t = [t_current]


    while t_current < t_stop:
        p,q = fput.step(p,q,dt)
        t_current += dt
        if iteration == 10:
            start_time = time.time()                    

        if iteration % analysis_cadence == 0:
            t.append(t_current)
            p_list.append(p.copy())
            q_list.append(q.copy())
            e_1.append(calc_mode_energy(p,q,k,1,alpha))
            e_2.append(calc_mode_energy(p,q,k,2,alpha))
            e_1_emode.append(calc_mode_energy(p,q,k,1,alpha,eigenmode=evecs))
            e_2_emode.append(calc_mode_energy(p,q,k,2,alpha,eigenmode=evecs))

            e_tot.append(hamiltonian(p,q,k,alpha))

        if iteration % cadence == 0:
            logger.info("iteration: {:d} t = {:5.2f} e_1 = {:5.2f} e_2 = {:5.2f}".format(iteration, t[-1],e_1[-1], e_2[-1]))

        iteration += 1
    main_loop_time = time.time()
    if time_reversal:
        while t[-1] > 0:
            p,q = fput.step(p,q,-dt)
            e_1.append(calc_mode_energy(p,q,k,1,alpha))
            e_2.append(calc_mode_energy(p,q,k,2,alpha))
            e_1_emode.append(calc_mode_energy(p,q,k,1,alpha,eigenmode=evecs))
            e_2_emode.append(calc_mode_energy(p,q,k,2,alpha,eigenmode=evecs))
            
            e_tot.append(hamiltonian(p,q,k,alpha))
            t.append(t[-1]-dt)
        logger.info("Ef/Ei - 1 = {:5.5e}".format(e_tot[-1]/e_tot[0]-1))
        logger.info("t final = {}".format(t[-1]))
    time_reversal_time = time.time()
    logger.info("main loop time = {:10.5e}".format(main_loop_time - start_time))
    # write data
    with h5py.File(outbase/output_file_name,"w") as outfile:
        outfile['tasks/p'] = np.array(p_list)
        outfile['tasks/q'] = np.array(q_list)
        outfile['scales/t'] = np.array(t)
        outfile['energies/e_1'] = np.array(e_1)
        outfile['energies/e_2'] = np.array(e_2)
        outfile['energies/e_1_emode'] = np.array(e_1_emode)
        outfile['energies/e_2_emode'] = np.array(e_2_emode)
        outfile['energies/e_tot'] = np.array(e_tot)

    logger.info("final iteration = {:d} final time  = {:5.2f}".format(iteration, t[-1]))
