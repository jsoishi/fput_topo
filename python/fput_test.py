"""fput_test.py

A simple script to compute the FPUT problem using the SABA2C method outlined in 

Pace & Campbell, Chaos 29, 023132 (2019)

Usage:
    fput_test.py [--no-correction --dt=<dt>]

Options:
    --no-correction             Turn off SABA2C corrections
    --dt=<dt>                   timestep [default: 0.1]

"""

import numpy as np
import h5py
import matplotlib.pyplot as plt
plt.style.use('prl')

class SABA2C(object):
    g = (2-np.sqrt(3))/24.
    c1 = 0.5*(1 - 1/np.sqrt(3))
    c2 = 1/np.sqrt(3)
    d1 = 0.5
    def A_stage(self, p,q, dt, c):
        raise NotImplementedError("SABA2C is a base class.")
    def B_stage(self, p,q, dt):
        raise NotImplementedError("SABA2C is a base class.")
    def C_stage(self, p,q, dt):
        raise NotImplementedError("SABA2C is a base class.")
    
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

class FPUT(SABA2C):
    def __init__(self, chi, k=None, u=3, correction=True):
        self.chi = chi
        self.u = u
        self.u_exp = u - 1
        self.correction = correction

    def A_stage(self, p, q, dt, c):
        tau = c*dt
        q_out = q + tau*p
        return p, q_out
    
    def B_stage(self, p, q, dt):

        tau = self.d1*dt
        
        p_out = np.zeros_like(p)
        dq = q[1:] - q[0:-1]
        p_out[1:-1] = p[1:-1] + tau*(self.k[1:]*(dq[1:] + self.chi*dq[1:]**(self.u_exp)) - self.k[0:-1]*(dq[0:-1] + self.chi*dq[0:-1]**(self.u_exp)))

        return p_out, q

    # def C_stage(self, p, q, dt):
    #     p_out = np.zeros_like(p)
    #     two_tau = -self.g * dt**3
    #     dq = q[1:] - q[0:-1]
    #     dq2n = dq[2:-1]
    #     dq2np1 = dq[3:]
    #     dq2nm1 = dq[1:-2]
    #     dq2nm2 = dq[0:-3]
        
    #     p_out[1] = p[1] \
    #         + two_tau*(dq[1] - dq[2] + self.chi*(dq[1]**self.u_exp - dq[2]**self.u_exp))*(1 + self.chi*self.u_exp*dq[1]**(self.u - 2)) \
    #         + two_tau*(q[2] - 2*q[1] + self.chi*(dq[1]**self.u_exp - q[1]**self.u_exp))*(2 + self.chi*self.u_exp*(q[1]**(self.u - 2) + dq[1]**(self.u - 2)))
        
    #     p_out[2:-2] = p[2:-2] \
    #         + two_tau*(dq2n - dq2np1 + self.chi*(dq2n**self.u_exp - dq2np1**self.u_exp))*(1+self.chi*self.u_exp*dq2n**(self.u - 2)) \
    #         + two_tau*(dq2n - dq2nm1 + self.chi*(dq2n**self.u_exp - dq2nm1**self.u_exp))*(2 + self.chi*self.u_exp*(dq2nm1**(self.u - 2) + dq2n**(self.u - 2))) \
    #         + two_tau*(dq2nm2 - dq2nm1 + self.chi*(dq2nm2**self.u_exp - dq2nm1**self.u_exp))*(1 + self.chi*self.u_exp*dq2nm1**(self.u - 2))
    #     p_out[-2] = p[-2] \
    #         + two_tau*(q[-3] - 2*q[-2] + self.chi*((-q[-2])**self.u_exp - dq[-2]**self.u_exp))*(2 + self.chi*self.u_exp*(dq[-2]**(self.u - 2) + (-q[-2])**(self.u - 2))) \
    #         + two_tau*(dq[-3] - dq[-2] + self.chi*(dq[-3]**self.u_exp - dq[-2]**self.u_exp))*(1 + self.chi*self.u_exp*dq[-2]**(self.u - 2))

    #     return p_out, q
    def C_stage(self, p, q, dt):
        p_out = np.zeros_like(p)
        two_tau = -self.g * dt**3
        # dq = q[1:] - q[0:-1]
        # dq2n = dq[2:-1]
        # dq2np1 = dq[3:]
        # dq2nm1 = dq[1:-2]
        # dq2nm2 = dq[0:-3]

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

def calc_mode_energy(p, q, k, mode, alpha):

    p_mode, q_mode = project_mode(p,q,mode)
    
                 
    return hamiltonian(p_mode, q_mode, k, alpha)

def project_mode(p, q, mode):
    # x_n = n dx
    # L = (N+1) dx
    # dx = L/(N+1)
    # f_k = <k|f> = \int f(x) sin(k pi x/L) dx = \sum_n f_n sin(k pi n dx/(dx (N+1))) L/(N+1)
    # f(x) = \Sum_k f_k sin(k pi x/L)
    N = len(q)-2
    n = np.arange(1,N+1)

    p_mode = 2*np.sum(p[1:-1]*np.sin(mode*n*np.pi/(N+1)))/(N+1)
    q_mode = 2*np.sum(q[1:-1]*np.sin(mode*n*np.pi/(N+1)))/(N+1)
    #print("mode {:d}: p, q = {:f},{:f}".format(mode, p_mode, q_mode))
    basis_fn = np.zeros_like(q)
    basis_fn[1:-1] = np.sin(mode*n*np.pi/(N+1))
    return p_mode*basis_fn, q_mode*basis_fn

def hamiltonian(p, q, k, alpha):
    pn = p[0:-1]
    qn = q[0:-1]
    qnp1 = q[1:]
    
    H = 0.5* (pn**2).sum() + (0.5*k*(qnp1 - qn)**2).sum() + (k*alpha/3 * (qnp1 - qn)**3).sum()

    return H

if __name__ == "__main__":

    from docopt import docopt

    # parse arguments
    args = docopt(__doc__)
    dt = float(args['--dt'])
    if args['--no-correction']:
        correction=False
    else:
        correction=True

    time_reversal = False
    alpha = 0.25
    N = 31
    A = 1
    cadence = 100
    omega_n = 1
    period = 2*np.pi/omega_n

    print("correction = ", correction)
    outfilename = "fput_N{}_A{}_alpha{}_dt{}".format(N,A,alpha,dt)
    if not correction:
        outfilename +="_nocorrection"
    outfilename += ".h5"
    
    fput = FPUT(alpha, correction=correction)

    n = np.arange(1,N+1)
    q = np.zeros(N+2)
    p = np.zeros(N+2)
    k = np.ones(N+1)
    fput.k = k
    q[1:-1] = A*np.sin(np.pi*n/(N+1))

    # plt.style.use('prl')
    # plt.rcParams["figure.figsize"] = (10,8)
    # plt.plot(q)
    # plt.xlabel("n")
    # plt.ylabel("q")
    # plt.savefig("q_init.png",dpi=100)
    # plt.clf()
    t = [0]
    t_stop = 100*period
    #dt = period/400

    e_1 = [calc_mode_energy(p,q,k,1,alpha)]
    e_2 = [calc_mode_energy(p,q,k,2,alpha)]
    e_tot = [hamiltonian(p,q,k, alpha)]
    print("E init = {:7.5f}".format(e_tot[0]))
    print("e_1[0] = {}".format(e_1[0]))

    iteration = 1
    # fig,ax = plt.subplots()
    # line, = ax.plot(q, label='total')
    # p1,q1 = project_mode(p,q,1)
    # line_m1, = ax.plot(q1, label='mode 1')
    # ax.set_ylim(-1,1)
    # ax.set_xlabel("n")
    # ax.set_ylabel("q")
    # ax.legend()
    # fig.savefig("frames/snap_{:05d}.png".format(0))
    p_list = [p]
    q_list = [q]
    while t[-1] < t_stop:
        p,q = fput.step(p,q,dt)
        p_list.append(p)
        q_list.append(q)
        e_1.append(calc_mode_energy(p,q,k,1,alpha))
        e_2.append(calc_mode_energy(p,q,k,2,alpha))
        e_tot.append(hamiltonian(p,q,k,alpha))
        t.append(t[-1]+dt)
        if iteration % cadence == 0:
            print("iteration: {:d} e_1 = {:5.2f} e_2 = {:5.2f}".format(iteration, e_1[-1], e_2[-1]))
            # line.set_ydata(q)
            # p1,q1 = project_mode(p,q,1)
            # line_m1.set_ydata(q1)
            # fig.canvas.draw()
            # fig.savefig("frames/snap_{:05d}.png".format(iteration))

        iteration += 1
    
    with h5py.File(outfilename,"w") as outfile:
        outfile.create_group("state")
        outfile.create_group("energies")
        outfile.create_group("scales")
        outfile['tasks/p'] = np.array(p_list)
        outfile['tasks/q'] = np.array(q_list)
        outfile['scales/t'] = np.array(t)
        outfile['energies/e_1'] = np.array(e_1)
        outfile['energies/e_2'] = np.array(e_2)
        outfile['energies/e_tot'] = np.array(e_tot)

    if time_reversal:
        while t[-1] > 0:
            p,q = fput.step(p,q,-dt)
            e_1.append(calc_mode_energy(p,q,1,alpha))
            e_2.append(calc_mode_energy(p,q,2,alpha))
            e_tot.append(hamiltonian(p,q,k,alpha))
            t.append(t[-1]-dt)
        print("Ef/Ei - 1 = {:5.5e}".format(e_tot[-1]/e_tot[0]-1))
        print("t final = {}".format(t[-1]))

    # line.set_ydata(q)
    # fig.canvas.draw()
    # fig.savefig("frames/snap_{:05d}.png".format(iteration))

    # e_tot = np.array(e_tot)
    # fig, ax = plt.subplots()
    # ax.plot(t,e_1,label='mode 1')
    # ax.plot(t,e_2,label='mode 2')
    # ax.plot(t,e_tot,'k',alpha=0.4, label='total energy')
    # plt.axvline(1.66e3, alpha=0.4)
    # ax.set_xlabel("time")
    # ax.set_ylabel("Energy")
    # ax.legend(loc='upper right')
    # plt.tight_layout()
    # fig.savefig("energy_vs_time.png",dpi=400)
    # plt.clf()
    # fig, ax = plt.subplots()
    # ax.plot(t,np.abs(e_tot/e_tot[0] -1))
    # ax.axhline(0)
    # ax.set_yscale('log')
    # ax.set_xlabel(r"$t$")
    # ax.set_ylabel(r"$E/E_0 - 1$")

    # fig.savefig("energy_cons_vs_time.png",dpi=100)
    print("final iteration = {:d} final time  = {:5.2f}".format(iteration, t[-1]))
