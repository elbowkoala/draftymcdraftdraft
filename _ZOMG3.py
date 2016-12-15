import numpy as np
from numpy.linalg import solve
import matplotlib.pyplot as plt
from matplotlib import animation
from math import sqrt
from mpl_toolkits.mplot3d import axes3d



q = 1.0 #-1.602e-19        #Charge of electron
m = .010 #9.11e-31         #Mass of electron
qm = float(q)/m            #Charge to mass ratio
kB = 1.0# 1.38064852e-23   #Boltzmann's constant
epsilon = 1.0 #8.85418782e-12  #Permittivity of vacuum
Te =  1.0 # 116.0          #Electron temperature (1K = 8.6e-5 eV)
B0 = 0.0                   #Set external B-field (z-dir only)
ne = 1.0 #1.0e8            #electron density per cubic meter
v_thermal = 1.0            #sqrt(kB*Te/m) #electron thermal energy
omega_c = qm*B0            #Bfield frequency
omega_p = 1.0              #sqrt(ne*q**2/(epsilon*m)) #Plasma frequency
l_de = 1.0                 #sqrt((epsilon*kB*Te)/((q**2)*ne))  #Debye length

Nx = 15                    #Number of x-grid
Ny = 15                    #Number of y-grid
dx = 1.0 #l_de
dy = 1.0 #l_de
L = Nx * dx
Npt = 20           #Number of cloud particles per grid cell
#dt = 0.1 #1.0/omega_c
V0 = 10.0
#omdt = omega_c * dt
###############################################################

part_pos = L*(np.random.rand(Npt,2))
part_vel = np.zeros((Npt, 2),float)

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

#print Ex_ext
#Ex_ext[0,5] = -13402.0
#print Ex_ext
#print "part_pos:",part_pos
#plt.imshow(Ex_ext)
#plt.gca().invert_yaxis()
#plt.plot(part_pos[:,0],part_pos[:,1],"ko")
#plt.show()
    
    
def Qgrid(part_pos):
    Qgrid = np.zeros((Ny+1,Nx+1),float)
    jcj_pos = np.empty((Npt,2),float)
    ici_pos = np.empty((Npt,2),float)
    for n in range(Npt):
        
        j = int((part_pos[n,0]/(dx)))
        i = int((part_pos[n,1]/(dy)))
    
        ci = float((part_pos[n,1]/(dy))) - i
        cj = float((part_pos[n,0]/(dx))) - j

        
        jcj_pos[n,0] = j
        jcj_pos[n,1] = cj
        ici_pos[n,0] = i
        ici_pos[n,1] = ci

        if i > (Nx-1) or j > (Ny-1):
            print "offending position n:",n," ",part_pos[n,:]
        
        Qgrid[i,j] += (1.0-cj)*(1.0-ci)
        Qgrid[i+1,j] += (1.0-cj)*ci
        Qgrid[i,j+1] += cj*(1.0-ci)
        Qgrid[i+1,j+1] += cj*ci
    Qgrid[0,:] *= 2.0
    Qgrid[Ny,:] *= 2.0
    Qgrid[:,0] *= 2.0
    Qgrid[:,Nx] *= 2.0
    x_plot = part_pos[:,0]
    y_plot = part_pos[:,1]
    #plt.plot(x_plot, y_plot, "ko")
    #plt.imshow(Qgrid)
    #plt.show()
    #plt.gca().invert_yaxis()
    #plt.plot(xj_plot, yi_plot, "r+", ms=12)
    #plt.contour(X,Y,Qgrid)
    
    return jcj_pos, ici_pos, Qgrid


A = np.zeros((Ny+1, Nx+1),float)
np.fill_diagonal(A, -2.0)
for i in range(Ny+1):
    for j in range(Nx+1):
        if i == j+1:
            A[i,j] = 1.0
        if j == i+1:
            A[i,j] = 1.0

#print "shape A",np.shape(A)
#print A

def potsolve(rho_0):
    rho_f = np.zeros((Nx+1, Nx+1),complex)
    v_vec = np.zeros((Nx+1, 1),complex)
    phi_f = np.empty((Ny+1, Nx+1),complex)
    phi_grid = np.empty((Nx+1, Nx+1),complex)
    
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
    print "phi_grid",phi_grid
    #plt.contour(X,Y,phi_grid)

    #plt.show()
    return phi_grid


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
    #plt.contour(X,Y,phi_i,colors='k')
    #plt.gca().invert_yaxis()
    return phi_i
    

def Efield(phi):
    Ex = np.zeros((Ny+1,Nx+1),float) 
    Ey = np.zeros((Ny+1,Nx+1), float) 
    
    for i in range(1,Ny):
        for j in range(1,Nx):            
            Ey[i,j] = (phi[i+1,j]-phi[i-1,j])
            Ex[i,j] = (phi[i,j+1]-phi[i,j-1])
    Ex *= 0.5
    Ey *= 0.5
    
    Ex[:,0] = (phi[:,1]-phi[:,0])
    Ex[:,Nx] = (phi[:,Nx]-phi[:,Nx-1])
    Ey[0,:] = (phi[1,:]-phi[0,:])
    Ey[Ny,:] = (phi[Ny,:]-phi[Ny-1,:])
    

    Ex *= -1.0/(dx)
    Ey *= -1.0/(dy)
    
    Ex += Ex_ext
    Ey += Ey_ext
    
    return Ex, Ey



def Epts(Ex,Ey,jcj,ici): 
    Epts = np.empty((Npt,2),float)

    for n in range(Npt):
        j = int(jcj[n,0])
        i = int(ici[n,0])

        cj = jcj[n,1]
        ci = ici[n,1]

        Epts[n,0] = Ex[i,j]*(1.0-ci)*(1.0-cj) + Ex[i+1,j]*(cj)*(1.0-ci) + Ex[i,j+1]*(1.0-cj)*(ci) + Ex[i+1,j+1]*(cj)*(ci)
        Epts[n,1] = Ey[i,j]*(1.0-ci)*(1.0-cj) + Ey[i+1,j]*(cj)*(1.0-ci) + Ey[i,j+1]*(1.0-cj)*(ci) + Ey[i+1,j+1]*(cj)*(ci)
    
    return Epts



plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)


def evolve(t_final):
    part_pos_0 = part_pos
    part_vel_0 = part_vel
    
    t = 0.0
    
    while t < t_final:
        print "tstep:",t
        
        vmax = abs(np.amax(part_vel_0))
        if vmax < 1.0:
            dt = 0.1
        else:
            dt = 1.0*dx/vmax
        
      
        jcj_0, ici_0, Qgrid_0 = Qgrid(part_pos_0)
        rho_0 = Qgrid_0 
        phi_0 = phi(rho_0)        
        Efx_0, Efy_0 = Efield(phi_0)
        Epts_0 = Epts(Efx_0, Efy_0, jcj_0, ici_0)

        
        #Initially have v(t=0), do half time step back for v(-.5dt) for Verlet
        #Half-step back E-field, half rotation B-field (using theta/2 for rotation)
        if t == 0.0:
            dt = 0.2*dx/(qm*np.amax(Epts_0))
            print "init dt:",dt
            part_vel_0 -= 0.5 * dt * qm * Epts_0
            
            
        part_vel_0 += dt * qm * Epts_0
        part_pos_0 += dt * part_vel_0

        part_vel_0_plot = part_vel_0 + 0.5*dt*qm*Epts_0
        
        #print "new pos 10",part_pos_0[10,:]
        #print "new vel 10",part_vel_0[10,:]
        
        
        for n in range(Npt):
            if part_pos_0[n,0] >= (Nx)*dx:
                print "2offending n pos:",n,part_pos_0[n,:]
                part_pos_0[n,0] = 2.0*Nx*dx - part_pos_0[n,0]
                part_vel_0[n,0] *= -1.0
                print "2fixed n pos:",n,part_pos_0[n,:]

            if part_pos_0[n,0] <= 0:
                print "2offending n pos:",n,part_pos_0[n,:]
                part_pos_0[n,0] = abs(part_pos_0[n,0])
                part_vel_0[n,0] *= -1.0
                print "2fixed n pos:",n,part_pos_0[n,:]

            if part_pos_0[n,1] >= (Ny)*dy:
                print "2offending n pos:",n,part_pos_0[n,:]
                part_pos_0[n,1] -= Ny*dy
                print "2fixed n pos:",n,part_pos_0[n,:]

            if part_pos_0[n,1] <= 0:
                print "2offending n pos:",n,part_pos_0[n,:]
                part_pos_0[n,1] += Ny*dy
                print "2fixed n pos:",n,part_pos_0[n,:]

        del ax.collections[:]
        #ax.plot(part_pos_0[:,0], part_vel_0_plot[:,0], "k.")
        #plt.show()
        #plt.cla()
        ax.imshow(Qgrid_0)
        plt.gca().invert_yaxis()
        plt.pause(.0001)
        #ax.plot(part_pos_0[:,0], part_pos_0[:,1], "k.")       
        #plt.pause(.0001)
        ax.contour(X,Y,phi_0)
        #lt.pause(.0001)
        #ax.quiver(X,Y,Efx_0,Efy_0,color='k')     
         
        plt.show()
        plt.pause(.0001)
        print "max vel:",np.amax(part_vel_0)
        
    
        print "next dt:",dt
        t += dt

evolve(1.0)


evolve(tsteps*dt)
#plt.ioff()
