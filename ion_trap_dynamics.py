import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.pyplot import imshow
import random
import cmath
from scipy import signal
from scipy.optimize import curve_fit
from matplotlib import cm
from matplotlib.colors import ListedColormap,LinearSegmentedColormap
from matplotlib import colors
from scipy.stats import norm
from scipy.stats import cauchy
import matplotlib.mlab as mlab
from scipy import stats


def OneDimIonTrap():

    # saying that, on the average, the ions lie in a 1d line. But still need to consider the dynamics in 3d
    # to get both transverse and longitudinal modes of oscillation

    Nions = 5

    tf = 600
    Ntimes = 3*10**4
    dt = tf/Ntimes
    time = np.linspace(0, tf, Ntimes)

    # define effective spring constants of the harmonic traps; transverse and longitudinal
    kt = 20
    kl = 2 # should be << kt, otherwise they won't lie in a straight line

    # define arrays to hold position information
    ion_x = np.zeros((Nions, Ntimes))
    ion_y = np.zeros((Nions, Ntimes))
    ion_z = np.zeros((Nions, Ntimes))

    # initial positions and velocities; arbitrary
    for ii in range(0, Nions):
        ion_x[ii, 0] = 0.001*ii
        ion_y[ii, 0] = 0.002*ii
        ion_z[ii, 0] = -0.001*ii
        ion_x[ii, 1] = 0.002*ii
        ion_y[ii, 1] = -0.003*ii
        ion_z[ii, 1] = 0.001*ii

    # manually setting the initial z positions to roughly their equilibrium positions for the given k=2;
    # as obtained from the code below
    ion_z[0, 0] = -1.45
    ion_z[0, 1] = -1.45
    ion_z[1, 0] = -0.655
    ion_z[1, 1] = -0.655
    ion_z[2, 0] = -0.05
    ion_z[2, 1] = -0.05
    ion_z[3, 0] = 0.655
    ion_z[3, 1] = 0.655
    ion_z[4, 0] = 1.45
    ion_z[4, 1] = 1.45

    # for each time step...
    for tt in range(1, Ntimes - 1):

        # for each ion..........................................................
        for nn in range(0, Nions):

            #----------------write out the force between ions------------------#
            f_ions_x = 0
            f_ions_y = 0
            f_ions_z = 0

            # calculate the coulomb force on ion nn from all other ions jj
            for jj in range(0, Nions):
                if jj != nn:
                    f_ions_x = f_ions_x + (((ion_z[nn, tt] - ion_z[jj, tt])**2 + (ion_y[nn, tt] - ion_y[jj, tt])**2 + (ion_x[nn, tt] - ion_x[jj, tt])**2)**(-3/2))*(ion_x[nn, tt] - ion_x[jj, tt])
                    f_ions_y = f_ions_y + (((ion_z[nn, tt] - ion_z[jj, tt])**2 + (ion_y[nn, tt] - ion_y[jj, tt])**2 + (ion_x[nn, tt] - ion_x[jj, tt])**2)**(-3/2))*(ion_y[nn, tt] - ion_y[jj, tt])
                    f_ions_z = f_ions_z + (((ion_z[nn, tt] - ion_z[jj, tt])**2 + (ion_y[nn, tt] - ion_y[jj, tt])**2 + (ion_x[nn, tt] - ion_x[jj, tt])**2)**(-3/2))*(ion_z[nn, tt] - ion_z[jj, tt])
            #------------------------------------------------------------------#

            #-------write out the force from the harmonic potential------------#
            f_pot_x = -kt*ion_x[nn, tt]
            f_pot_y = -kt*ion_y[nn, tt]
            f_pot_z = -kl*ion_z[nn, tt]
            #------------------------------------------------------------------#

            # evolve to the next time step

            ion_z[nn, tt+1] = (dt**2)*(f_pot_z + f_ions_z) + 2*ion_z[nn, tt] - ion_z[nn, tt-1]
            ion_y[nn, tt+1] = (dt**2)*(f_pot_y + f_ions_y) + 2*ion_y[nn, tt] - ion_y[nn, tt-1]
            ion_x[nn, tt+1] = (dt**2)*(f_pot_x + f_ions_x) + 2*ion_x[nn, tt] - ion_x[nn, tt-1]

        # ......................................................................

    # Fourier amplitudes
    Z = np.zeros((Nions, Ntimes))
    Y = np.zeros((Nions, Ntimes))
    X = np.zeros((Nions, Ntimes))

    # take out the dc offset before doing the fft
    ion_x_BC_block = np.zeros((Nions, Ntimes))
    ion_y_BC_block = np.zeros((Nions, Ntimes))
    ion_z_BC_block = np.zeros((Nions, Ntimes))

    for ii in range(0, Nions):
        ion_z_BC_block[ii, :] = ion_z[ii,:] - np.mean(ion_z[ii, :])
        ion_y_BC_block[ii, :] = ion_y[ii,:] - np.mean(ion_y[ii, :])
        ion_x_BC_block[ii, :] = ion_x[ii,:] - np.mean(ion_x[ii, :])

    for nn in range(0, Nions):
        Z[nn, :]= np.fft.fft(ion_z_BC_block[nn, :])
        Y[nn, :]= np.fft.fft(ion_y_BC_block[nn, :])
        X[nn, :]= np.fft.fft(ion_x_BC_block[nn, :])

    # freq = np.linspace(-1/(2*dt), 1/(2*dt), tf/(dt))
    freq = np.fft.fftfreq(time.shape[-1])

    plot1 = plt.figure(1)
    for ii in range(0, Nions):
        plt.plot(freq, np.abs(Z[ii, :]),label='Ion #{}'.format(ii))
        plt.xlabel('Frequency (a.u.)',fontsize=20)
        plt.ylabel('Amplitude (a.u.)',fontsize=20)
        plt.title('Oscillations along Z (w/ DC block)')
        plt.legend(prop={"size":16})
    plot2 = plt.figure(2)
    for ii in range(0, Nions):
        plt.plot(time, ion_z[ii, :],label='Ion #{}'.format(ii))
        plt.xlabel('time (a.u.)',fontsize=20)
        plt.ylabel('position (a.u.)',fontsize=20)
        plt.title('Oscillations along Z')
        plt.legend(prop={"size":16})

    #
    # plot3 = plt.figure(3)
    # for ii in range(0, Nions-1):
    #     plt.plot(freq, Y[ii, :],label='Ion #{}'.format(ii))
    #     plt.xlabel('Frequency (a.u.)',fontsize=20)
    #     plt.ylabel('Amplitude (a.u.)',fontsize=20)
    #     plt.title('Oscillations along Y (w/ DC block)')
    #     plt.legend(prop={"size":16})
    # plot4 = plt.figure(4)
    # for ii in range(0, Nions-1):
    #     plt.plot(time, ion_y[ii, :],label='Ion #{}'.format(ii))
    #     plt.xlabel('time (a.u.)',fontsize=20)
    #     plt.ylabel('Amplitude (a.u.)',fontsize=20)
    #     plt.title('Oscillations along Y')
    #     plt.legend(prop={"size":16})
    #
    #
    plot5 = plt.figure(5)
    for ii in range(0, Nions):
        plt.plot(freq, X[ii, :],label='Ion #{}'.format(ii))
        plt.xlabel('Frequency (a.u.)',fontsize=20)
        plt.ylabel('Amplitude (a.u.)',fontsize=20)
        plt.title('Oscillations along X (w/ DC block)')
        plt.legend(prop={"size":16})
    plot6 = plt.figure(6)
    for ii in range(0, Nions):
        plt.plot(time, ion_x[ii, :],label='Ion #{}'.format(ii))
        plt.xlabel('time (a.u.)',fontsize=20)
        plt.ylabel('Amplitude (a.u.)',fontsize=20)
        plt.title('Oscillations along X')
        plt.legend(prop={"size":16})
    #

    plt.show()

    return

OneDimIonTrap()


# compute the equilibrium positions via Newton's method
# copying the code basically directly from Linge and Langtengen's 2020 book, section 7.6.4

def Newton_system(F, J, x, eps):

    F_value = F(x) # obtained from the initial guess
    F_norm = np.linalg.norm(F_value) # want to make the norm of the vector function F as close to zero as possible
    iteration_counter = 0
    while abs(F_norm) > eps and iteration_counter<1e7: # aim for a certain error in a limited number of iterations
        # print(x)
        # Trying to stop the code from diverging...
        # if two of the entries are equal, reassign one of them to a random number
        for nn in range(0, np.size(x)):
            for mm in range(0, nn):
                if np.abs(x[mm]-x[nn])<1e-2:
                    x[mm] = np.sqrt(np.size(x))*np.random.rand(1)[0]
        # If any entry is way too big or too small, reassign it to a random number
        for qq in range(0, np.size(x)):
            if x[qq]>1e2 or x[qq]<1e-3:
                x[qq] = np.sqrt(np.size(x))*np.random.rand(1)[0]


        delta = np.linalg.solve(J(x), -F_value) # Solve for the values (z_n(i) - z_n(i-1)) that set the tangent surface to zero
        x = x + delta # update the new values
        F_value = F(x) # update the function at the new values
        F_norm = np.linalg.norm(F_value) # update the new norm of F
        iteration_counter = iteration_counter+1
        # print(iteration_counter)

    if abs(F_norm)>eps:
        iteration_counter = -1

    return x, iteration_counter

# performing Newton's method for, say, 5 ions, to find their equilibrium positions
def FindIonEquilibria():

    NN=2 # 2NN+1 ions total; N distances to solve for
    N_Min = 2
    N_Max = 2
    k = 1
    # k_min = 0.1
    # k_max = 100
    # num_ks = 10**3
    # kk = np.linspace(k_min, k_max, num_ks)
    # l1 = np.linspace(k_min, k_max, num_ks)
    # l2 = np.linspace(k_min, k_max, num_ks)
    # l2l1 = np.linspace(k_min, k_max, num_ks)
    # for k in range(0, num_ks):
    colors = ['navy', 'coral', 'gold', 'plum', 'sandybrown', 'black', 'slategray']
    for N in range(N_Min, N_Max+1, 1):

        # arange the ions from left to right - ion 0 is the left-most ion not on the x=0 axis

        # First round of code: solve for 5 ions; by the symmetry of the problem, the algebra reduces to 2 equations in 2 unknowns
        # generally for 2n+1 ions, the equations should reduce to n equations in n unknowns, so long as the applied
        # potential is symmetric about z=0
        # def F(x):
        #     return np.array(
        #     [-kk[k]*x[0] + 1/(x[1]+x[0])**2 + 1/(2*x[0])**2 + 1/(x[0])**2 - 1/(x[0]-x[1])**2,
        #     -kk[k]*x[1] + 1/(x[1]-x[0])**2 + 1/x[1]**2 + 1/(x[1]+x[0])**2 + 1/(2*x[1])**2])


        # trying to write it out for general N ions
        def F(x):

            force = np.zeros(N)

            # enter the longitudinal trapping force, and the force from the ion at x=0 (x=0 not included in the x vector)
            for ii in range(0, N):
                force[ii] = -k*x[ii] + 1/x[ii]**2

            # enter the force from all ions in the region x>0, but to the left of ion ii (force to the right)
            for ii in range(0, N):
                for jj in range(0, ii):
                    force[ii] = force[ii] + 1/(x[jj]-x[ii])**2

            # enter the force from all ions in the region x>0, but to the right of ion ii (force to the left)
            for ii in range(0, N):
                for jj in range(ii+1, N):
                    force[ii] = force[ii] - 1/(x[jj]-x[ii])**2

            # enter the force from all ions in the region x<0, to the left of ion ii (force to the right)
            for ii in range(0, N):
                for jj in range(0, N):
                    force[ii] = force[ii] + 1/(x[jj]+x[ii])**2


            return force


        # 5 ions Jacobian
        # def J(x):
        #     return np.array(
        #     [[ -kk[k] - 2/(x[0]+x[1])**3 - 4/(2*x[0])**3 - 2/x[0]**3 - 2/(x[1]-x[0])**3, -2/(x[0]+x[1])**3 + 2/(x[1]-x[0])**3],
        #      [ 2/(x[0]+x[1])**3 - 2/(x[1]-x[0])**3, -kk[k] - 2/(-x[0]+x[1])**3 - 4/(2*x[1])**3 - 2/x[1]**3 - 2/(x[1]+x[0])**3]])

        # N ions
        def J(x):

            Jac = np.zeros((N, N))

            for ii in range(0, N):
                for jj in range(0, N):
                    if ii==jj:
                        Jac[ii, jj] = -k - 2/np.abs(x[ii])**3
                        for m in range(0, ii-1):
                            Jac[ii, jj] = Jac[ii, jj] - 2/np.abs(x[m] - x[ii])**3
                        for m in range(ii+1, N-1):
                            Jac[ii, jj] = Jac[ii, jj] - 2/np.abs(x[m] - x[ii])**3
                        for m in range(0, N-1):
                            Jac[ii, jj] = Jac[ii, jj] - 2/np.abs(x[m] + x[ii])**3
                    elif jj!=ii:
                        Jac[ii, jj] = +2/np.abs(x[jj] - x[ii])**3 + 2/np.abs(x[jj] + x[ii])**3

            return Jac


        x, n = Newton_system(F, J, x=np.linspace(0.1, np.sqrt(2*N), N), eps=1e-2)


        for ii in range(0, N):
            plt.scatter(2*N+1, x[ii], color=colors[N - N_Min + 1])
            plt.scatter(2*N+1, -x[ii], color=colors[N - N_Min + 1])
        plt.scatter(2*N+1, 0, color=colors[N - N_Min + 1])


    # def Power_Law(C, kappa, n):
    #     return (kappa/C)**(-n)
    # pars_1, _ = curve_fit(Power_Law, kk, l1, p0 = [2, 0.5])
    # n1_fit = str(np.round_(pars_1[1],decimals = 2))
    # C1_fit= str(np.round_(pars_1[0],decimals = 5))
    #
    # pars_2, _ = curve_fit(Power_Law, kk, l2, p0 = [2, 0.5])
    # n2_fit = str(np.round_(pars_1[1],decimals = 2))
    # C2_fit= str(np.round_(pars_1[0],decimals = 5))

    # plt.plot(kk, l1, label='L1')
    # plt.plot(kk, Power_Law(kk, *pars_1), linestyle='--', linewidth=2, color='black',label='Power Law Fit: (k/{})^-{}'.format(C1_fit, n1_fit))
    # plt.plot(kk, l2, label='L2')
    # plt.plot(kk, Power_Law(kk, *pars_2), linestyle='--', linewidth=2, color='black',label='Power Law Fit: (k/{})^-{}'.format(C2_fit, n2_fit))
    # plt.plot(kk, l2l1, label='L2/L1')

    plt.xlabel('Number of ions',fontsize=20)
    plt.title('Ion position')
    plt.show()

    # fig = plt.figure()
    #
    # Nions = 3
    # ion_positions = np.zeros((1, Nions))
    #
    # x = []
    #
    # ax = fig.add_subplot(111)
    #
    # for zz in range(0,iteration_counter-1):
    #     for ii in range(0, Nions-1):
    #         ion_positions[1, ii]
    #
    #         p, = plt.plot(local,np.angle(u[:,zz]), 'b-')
    #         #plt.xlim([998, 1001])
    #         #plt.ylim([-0.01, 0.005])
    #         phase.append([p])
    #
    #
    # ani = animation.ArtistAnimation(fig, phase, interval=1, repeat_delay=100)
    #
    # plt.show()

    return

# FindIonEquilibria()
