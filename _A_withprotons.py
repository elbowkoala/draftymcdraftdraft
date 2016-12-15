import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib import animation
from math import sqrt
from mpl_toolkits.mplot3d import axes3d



q_e = -1.0 #-1.602e-19        #Charge of electron
m_e = .10 #9.11e-31         #Mass of electron
qm_e = float(q_e)/m_e            #Charge to mass ratio
q_p = 1.0
m_p = 1.0
qm_p = float(q_p)/m_p
kB = 1.0# 1.38064852e-23   #Boltzmann's constant
epsilon = 1.0 #8.85418782e-12  #Permittivity of vacuum
Te =  1.0 # 116.0          #Electron temperature (1K = 8.6e-5 eV)
B0 = 0.0                   #Set external B-field (z-dir only)
ne = 1.0 #1.0e8            #electron density per cubic meter
v_thermal = 1.0            #sqrt(kB*Te/m) #electron thermal energy
#omega_c = qm*B0            #Bfield frequency
omega_p = 1.0              #sqrt(ne*q**2/(epsilon*m)) #Plasma frequency
l_de = 1.0                 #sqrt((epsilon*kB*Te)/((q**2)*ne))  #Debye length

Nx = 31                    #Number of x-grid
Ny = 31                    #Number of y-grid
dx = 1.0 #l_de
dy = 1.0 #l_de
L = Nx * dx
Npt = 10           #Number of cloud particles per grid cell
#dt = 0.1 #1.0/omega_c
V0 = 0.0
#omdt = omega_c * dt
###############################################################

part_pos_e = L*(np.random.rand(Npt,2))
part_pos_e = np.copy(part_pos_e)
part_vel_e = np.zeros((Npt, 2),float)
part_pos_p = L*(np.random.rand(Npt,2))
part_pos_p = np.copy(part_pos_p)
part_vel_p = np.zeros((Npt, 2),float)


x_contour, y_contour= np.linspace(0,Nx,Nx+1), np.linspace(0,Ny,Ny+1)
X, Y = np.meshgrid(x_contour, y_contour)

#Potential is periodic in y-direction, bound in x-direction
#Set potential at x=0 and x=L (Nx)
V0x = 0.0
VNx = .0

E0_ext = 0.0
Ex_ext = np.zeros((Ny+1, Nx+1),float)
Ey_ext = np.zeros((Ny+1, Nx+1),float)
for i in range(Ny+1):
    for j in range(Nx+1):
        Ey_ext[i,j] += E0_ext

print "part_pos_e:",part_pos_e,"part_pos_p:",part_pos_p
    
def Qgrid(part_pos_e, part_pos_p):
    Qgrid = np.zeros((Ny+1,Nx+1),float)
    jcj_pos_e = np.empty((Npt,2),float)
    ici_pos_e = np.empty((Npt,2),float)

    jcj_pos_p = np.empty((Npt,2),float)
    ici_pos_p = np.empty((Npt,2),float)
    
    for n in range(Npt):
        
        j_e = int((part_pos_e[n,0]/(dx)))
        i_e = int((part_pos_e[n,1]/(dy)))

        ci_e = float((part_pos_e[n,1]/(dy))) - i_e
        cj_e = float((part_pos_e[n,0]/(dx))) - j_e
        
        jcj_pos_e[n,0] = j_e
        jcj_pos_e[n,1] = cj_e
        ici_pos_e[n,0] = i_e
        ici_pos_e[n,1] = ci_e

        j_p = int((part_pos_p[n,0]/(dx)))
        i_p = int((part_pos_p[n,1]/(dy)))

        ci_p = float((part_pos_p[n,1]/(dy))) - i_p
        cj_p = float((part_pos_p[n,0]/(dx))) - j_p
        
        jcj_pos_p[n,0] = j_p
        jcj_pos_p[n,1] = cj_p
        ici_pos_p[n,0] = i_p
        ici_pos_p[n,1] = ci_p

        if i_e > (Ny-1) or j_e > (Nx-1) or i_p > (Ny-1) or j_p > (Nx-1):
            print "offending position n e,p:",n," ",part_pos_e[n,:], part_pos_p[n,:]
        
        Qgrid[i_e,j_e] += q_e*(1.0-cj_e)*(1.0-ci_e) 
        Qgrid[i_p,j_p] += q_p*(1.0-cj_p)*(1.0-ci_p)
        Qgrid[i_e+1,j_e] += q_e*(1.0-cj_e)*ci_e
        Qgrid[i_p+1,j_p] += q_p*(1.0-cj_p)*ci_p
        Qgrid[i_e,j_e+1] += q_e*cj_e*(1.0-ci_e)
        Qgrid[i_p,j_p+1] += q_p*(1.0-ci_p)
        Qgrid[i_e+1,j_e+1] += q_e*cj_e*ci_e
        Qgrid[i_p+1,j_p+1] + q_p*cj_p*ci_p

    Qgrid[0,:] *= 2.0
    Qgrid[Ny,:] *= 2.0
    Qgrid[:,0] *= 2.0
    Qgrid[:,Nx] *= 2.0
    
    x_plot_e = part_pos_e[:,0]
    y_plot_e = part_pos_e[:,1]
    xj_plot_e, yi_plot_e = jcj_pos_e[:,0], ici_pos_p[:,0]

    x_plot_p = part_pos_p[:,0]
    y_plot_p = part_pos_p[:,1]
    xj_plot_p, yi_plot_p = jcj_pos_p[:,0], ici_pos_p[:,0]

    #plt.contour(X,Y,Qgrid,colors='k')
    
    return jcj_pos_e, ici_pos_e, jcj_pos_p, ici_pos_p, Qgrid


#A = np.zeros((Ny+1, Nx+1),float)
#np.fill_diagonal(A, -2.0)
#for i in range(Ny+1):
#    for j in range(Nx+1):
#        if i == j+1:
#            A[i,j] = 1.0
#        if j == i+1:
#            A[i,j] = 1.0
#
#print "shape A",np.shape(A)
#print A



def potsolve(rho_0):
    v_vec = np.zeros((Nx+1, 1),complex)
    phi_f = np.empty((Ny+1, Nx+1),complex)
    phi_grid = np.empty((Nx+1, Nx+1),complex)
    for xi in range(Nx+1):
        rho_before = rho_0[:,xi]
        rho_after = np.fft.rfft(rho_before)
        print "rho_before:,",rho_before
        print "rho_after:",rho_after

        for m in range(Nx+1):
            AA = np.copy(A)
            
            
    for m in range(Nx+1):
        AA = np.copy(A)
        for xi in range(Nx+1):
            for yi in range(Nx+1):
                rho_f[m,xi] += rho_0[yi,xi]*np.exp(-1.0j*2.0*np.pi*m*yi/float(Ny))
                
            rho_f[m,xi] *= (dx*dx)
            if xi == 0:
                rho_f[m,xi] -= V0x
            if xi == Nx:
                rho_f[m,xi] -= VNx
            v_vec[xi] = rho_0[m,xi]
       
        dm = 1.0 + 2.0 * ((float(dx)/dy)*np.sin(np.pi*m/Ny))**2

        for ii in range(Nx+1):
            AA[ii,ii] *= dm
        print "v_vec",v_vec    
        phi_vec = solve(A,v_vec)
        print "solved phi_vec for m:",m,phi_vec
        for xi in range(Nx+1):
            phi_f[m,xi] = phi_vec[xi].real
        
            
    print "phi_grid",phi_f       

    for xi in range(Nx+1):
        for yi in range(Nx+1):
            for m in range(Nx+1):
                phi_grid[yi,xi] += phi_f[m,xi]*np.exp(1.0j*2.0*np.pi*m*yi/float(Ny))
    phi_grid = phi_grid.real
    phi_grid *= (1.0/L)
   
    return phi_grid

#jcj_pos_e, ici_pos_e, jcj_pos_p, ici_pos_p, Qgrid_8 = Qgrid(part_pos_e, part_pos_p)
#rho_8 = Qgrid_8/(dx*dy)
#phi_8 = phi(rho_8)
#Ex_8, Ey_8 = Efield(phi_8)
#Epts_e_8, Epts_p_8 = Epts(Ex_8, Ey_8, jcj_pos_e, ici_pos_e, jcj_pos_p, ici_pos_p)
#potsolve(rho_8)

#Solving for the potential at the grid points using the Jacobi Method with fixed potentials at boundaries
def phi(rho):
    target = 0.01
    phi = np.zeros((Ny+1,Nx+1),float)
    phi[0,:] = V0
    phiprime = np.empty((Ny+1,Nx+1),float)


    delta = 1.0
    while delta > target:

        for i in range(Nx+1):
            for j in range(Ny+1):
                if i==0 or i==Nx or j==0 or j==Ny:
                    phiprime[i,j] = phi[i,j]
                else:
                    phiprime[i,j] = 0.25*(phi[i+1,j] + phi[i-1,j] \
                                          + phi[i,j+1]+phi[i,j-1] \
                                          + rho[i,j]*dx*dy)
        delta = np.max(abs(phi-phiprime))
        phi,phiprime = phiprime, phi
 
    return phi

    
#Solving for the potential using the FFT built-in function.
#Works nicely, but couldn't figure out how to apply boundary conditions in 2-D (FFT assumes periodicity)
def phi_fourier(rho):
    rho_f = np.fft.rfft2(rho)
    phi_f = np.empty_like(rho_f)  
    
    def W(x):
        y = np.exp(1.0j*2.0*np.pi*(x)/float(Nx+1))
        return y.real
    
    for m in range(len(rho_f)):
        for n in range(len(rho_f[0])):
            phi_f[m,n] = (dx*dy*rho_f[m,n])/(4.0 - W(m+1) - W(-m-1) - W(n+1) - W(-n-1))
    
    phi_i = np.fft.irfft2(phi_f)
 
    return phi_i

    
#Solve for the Electric Field at grid points using finite differencing of potential
def Efield(phi):
    
    Ex = np.zeros((Ny+1,Nx+1),float) 
    Ey = np.zeros((Ny+1,Nx+1), float) 

    #Potential difference spanning cell widths
    for i in range(1,Ny):
        for j in range(1,Nx):            
            Ey[i,j] = (phi[i+1,j]-phi[i-1,j])
            Ex[i,j] = (phi[i,j+1]-phi[i,j-1])
    #Dividing by two, averaging over the two cells
    Ex *= 0.5
    Ey *= 0.5

    #We do not have index + 1 or -1 at the boundaries, so just use one cell width and don't divide by two
    Ex[:,0] = (phi[:,1]-phi[:,0])
    Ex[:,Nx] = (phi[:,Nx]-phi[:,Nx-1])
    Ey[0,:] = (phi[1,:]-phi[0,:])
    Ey[Ny,:] = (phi[Ny,:]-phi[Ny-1,:])
    
    #Electric field is defined as the negative of the potential gradient 
    Ex *= -1.0/(dx)
    Ey *= -1.0/(dy)
    
    Ex += Ex_ext
    Ey += Ey_ext
    
    return Ex, Ey


#plt.plot(part_pos_e[:,0], part_pos_e[:,1], "ro", label='- charge')
#plt.plot(part_pos_p[:,0], part_pos_p[:,1], "bo", label='+ charge')

#plt.plot(jcj_pos_e[:,0], ici_pos_e[:,0], "r+", ms=12, label='- gridpnt')
#plt.plot(jcj_pos_p[:,0], ici_pos_p[:,0], "b+", ms=12, label='+ gridpnt')
#plt.xlim([0,15])
#plt.ylim([0,15])
#plt.title('Assigning positions to grid points',fontsize=20)
#plt.xlabel('X position index',fontsize=15)
#plt.ylabel('Y position index',fontsize=15)
#plt.show()


#plt.imshow(Qgrid_8, cmap="hot")
#plt.colorbar()
#plt.gca().invert_yaxis()
#plt.quiver(X,Y,Ex_8,Ey_8,color='y',label='E-lines')
#plt.contour(X,Y,phi_8,colors='k')
#plt.title('Charge Density, Potential, Efield at Gridpoints',fontsize=30)
#plt.legend(loc='lower left',prop={'size':12},shadow=True)
#plt.xlabel('X position',fontsize=15)
#plt.ylabel('Y position',fontsize=15)
#plt.xlim([0,L])
#plt.ylim([0,L])
#plt.show()



#Interpolate back the field onto the particles, using the same exact method as interpolating the particles onto the grid in def Qgrid
def Epts(Ex,Ey,jcj_e,ici_e, jcj_p, ici_p): 
    Epts_e = np.empty((Npt,2),float)
    Epts_p = np.empty((Npt,2),float)

    for n in range(Npt):
        j_e = int(jcj_e[n,0])
        i_e = int(ici_e[n,0])
        
        j_p = int(jcj_p[n,0])
        i_p = int(ici_p[n,0])

        cj_e = jcj_e[n,1]
        ci_e = ici_e[n,1]

        cj_p = jcj_p[n,1]
        ci_p = ici_p[n,1]

        Epts_e[n,0] = Ex[i_e,j_e]*(1.0-ci_e)*(1.0-cj_e) + Ex[i_e+1,j_e]*(cj_e)*(1.0-ci_e) + Ex[i_e,j_e+1]*(1.0-cj_e)*(ci_e) + Ex[i_e+1,j_e+1]*(cj_e)*(ci_e)
        Epts_p[n,0] = Ex[i_p,j_p]*(1.0-ci_p)*(1.0-cj_p) + Ex[i_p+1,j_p]*(cj_p)*(1.0-ci_p) + Ex[i_p,j_p+1]*(1.0-cj_p)*(ci_p) + Ex[i_p+1,j_p+1]*(cj_p)*(ci_p)
        
        Epts_e[n,1] = Ey[i_e,j_e]*(1.0-ci_e)*(1.0-cj_e) + Ey[i_e+1,j_e]*(cj_e)*(1.0-ci_e) + Ey[i_e,j_e+1]*(1.0-cj_e)*(ci_e) + Ey[i_e+1,j_e+1]*(cj_e)*(ci_e)
        Epts_p[n,1] = Ey[i_p,j_p]*(1.0-ci_p)*(1.0-cj_p) + Ey[i_p+1,j_p]*(cj_p)*(1.0-ci_p) + Ey[i_p,j_p+1]*(1.0-cj_p)*(ci_p) + Ey[i_p+1,j_p+1]*(cj_p)*(ci_p)

    
    return Epts_e, Epts_p





#plt.ion()
#fig = plt.figure()
#ax = fig.add_subplot(111)
x_phase_e = part_pos_e[:,0]
v_phase_e = part_vel_e[:,1]
plt.plot(x_phase_e[0::2], v_phase_e[0::2],'k.')
plt.xlabel('X',fontsize=15)
plt.ylabel('Vx',fontsize=15)
plt.title('Phase Space Distribution',fontsize=30)


def evolve(final_tstep):
    part_pos_0_e = part_pos_e
    part_pos_0_e *= 0.5
    part_vel_0_e = part_vel_e

    part_pos_0_p = part_pos_p
    part_vel_0_p = part_vel_p
    part_pos_0_p *= 0.5
    
    t = 0.0
    tstep = 0
    while tstep < final_tstep:
        #print "tstep:",t

        jcj_0_e, ici_0_e, jcj_0_p, ici_0_p, Qgrid_0 = Qgrid(part_pos_0_e, part_pos_0_p)
        rho_0 = Qgrid_0 
        phi_0 = phi_fourier(rho_0)        
        Efx_0, Efy_0 = Efield(phi_0)
        Epts_0_e, Epts_0_p = Epts(Efx_0, Efy_0, jcj_0_e, ici_0_e, jcj_0_p, ici_0_p)

        vmax = max( abs(np.amax(part_vel_0_e)), abs(np.amax(part_vel_0_p)), abs(np.amax(Epts_0_e)))
        #print "vmax:",vmax
        
        if vmax < 1.0:
            dt = 0.01
        else:
            dt = .10*dy/vmax
        #print "dt using vmax:",dt
        #Initially have v(t=0), do half time step back for v(-.5dt) for Verlet
        #Half-step back E-field, half rotation B-field (using theta/2 for rotation)
        if tstep == 0:
            vmax_init = max( qm_e*abs(np.amax(Epts_0_e)), qm_p*abs(np.amax(Epts_0_p)))
            dt = 0.2 * dx / vmax_init
            #print "init dt:",dt
            part_vel_0_e -= 0.5 * dt * qm_e * Epts_0_e
            part_vel_0_p -= 0.5 * dt * qm_p * Epts_0_p
            
            
        part_vel_0_e += dt * qm_e * Epts_0_e
        part_pos_0_e += dt * part_vel_0_e

        part_vel_0_p += dt* qm_p * Epts_0_p
        part_pos_0_p += dt* part_vel_0_p

        part_vel_0_plot_e = part_vel_0_e + 0.5*dt*qm_e*Epts_0_e
        part_vel_0_plot_p = part_vel_0_p + 0.5*dt*qm_p*Epts_0_p
   
        
        #print "new pos 10",part_pos_0[10,:]
        #print "new vel 10",part_vel_0[10,:]
        
        
        for n in range(Npt):
            if part_pos_0_e[n,0] >= (Nx)*dx:
                #print "2offending n pos:",n,part_pos_0_e[n,:]
                part_pos_0_e[n,0] -= Nx*dx 
                #print "2fixed n pos:",n,part_pos_0_e[n,:]

            if part_pos_0_e[n,0] <= 0:
                #print "2offending n pos:",n,part_pos_0_e[n,:]
                part_pos_0_e[n,0] += Nx*dx
                #print "2fixed n pos:",n,part_pos_0_e[n,:]

            if part_pos_0_e[n,1] >= Ny*dy:
                #print "2offending n pos:",n,part_pos_0_e[n,:]
                part_pos_0_e[n,1] -= Ny*dy
                #print "2fixed n pos:",n,part_pos_0_e[n,:]

            if part_pos_0_e[n,1] <= 0:
                #print "2offending n pos:",n,part_pos_0_e[n,:]
                part_pos_0_e[n,1] += Ny*dy
                #print "2fixed n pos:",n,part_pos_0_e[n,:]


            if part_pos_0_p[n,0] >= Nx*dx:
                #print "2offending n pos:",n,part_pos_0_p[n,:]
                part_pos_0_p[n,0] -= Nx*dx
                #print "2fixed n pos:",n,part_pos_0_p[n,:]

            if part_pos_0_p[n,0] <= 0:
                #print "2offending n pos:",n,part_pos_0_p[n,:]
                part_pos_0_p[n,0] += Nx*dx 
                #print "2fixed n pos:",n,part_pos_0_p[n,:]

            if part_pos_0_p[n,1] >= (Ny)*dy:
                #print "2offending n pos:",n,part_pos_0_p[n,:]
                part_pos_0_p[n,1] -= Ny*dy
                #print "2fixed n pos:",n,part_pos_0_p[n,:]

            if part_pos_0_p[n,1] <= 0:
                #print "2offending n pos:",n,part_pos_0_p[n,:]
                part_pos_0_p[n,1] += Ny*dy
                #print "2fixed n pos:",n,part_pos_0_p[n,:]


        #del ax.collections[:]
        x_phase_e = part_pos_0_e[:,0]
        v_phase_e = part_vel_0_e[:,1]
        plt.plot(x_phase_e[0::2], v_phase_e[0::2], "k.")
        #plt.show()
        #plt.cla()
        #ax.imshow(Qgrid_0)
        #plt.gca().invert_yaxis()
        plt.pause(.0001)
        #ax.plot(part_pos_0[:,0], part_pos_0[:,1], "k.")       
        #plt.pause(.0001)
        #ax.contour(X,Y,phi_0)
        #plt.pause(.0001)
        #ax.quiver(X,Y,Efx_0,Efy_0,color='k')     
         
        
        
        #print "next dt:",dt
        t += dt
        tstep += 1
        if tstep%100==0:
            print "tstep:",tstep

evolve(1000)
plt.show()

#plt.ioff()
