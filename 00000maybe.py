import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from math import sqrt
from mpl_toolkits.mplot3d import axes3d



q = 1.0 #-1.602e-19            #Charge of electron
m = 1.0 #9.11e-31              #Mass of electron
qm = float(q)/m           #Charge to mass ratio
kB = 1.0 #1.38064852e-23       #Boltzmann's constant
epsilon = 1.0 #8.85418782e-12  #Permittivity of vacuum
Te =  1.0 # 116.0              #Electron temperature (1K = 8.6e-5 eV)
B0 = 0.0                  #Set external B-field (z-dir only)
ne = 1.0 #1.0e8               #electron density per cubic meter
v_thermal = 1.0 #sqrt(kB*Te/m) #electron thermal energy
omega_c = qm*B0           #Bfield frequency
omega_p = 1.0 #sqrt(ne*q**2/(epsilon*m)) #Plasma frequency
l_de = 1.0 #sqrt((epsilon*kB*Te)/((q**2)*ne))  #Debye length

Nx = 5                   #Number of x-grid
Ny = 5                   #Number of y-grid
dx = l_de
dy = l_de
L = Nx * dx
Npt = 1*Nx*Ny           #Number of cloud particles per grid cell
dt = .1 #0.1/omega_p
V0 = 10000000000.0
omdt = 0 #omega_c * dt

print "v_th=",v_thermal
print "omega_p=",omega_p
print "dx=",dx
print "l_de",l_de
print "L=",L
print "dt=",dt



part_pos = L*(np.random.rand(Npt,2))
#part_vel = np.random.rand(Npt,2)
#part_vel -= 0.5* np.ones((Npt,2),float)
part_vel = np.zeros((Npt, 2),float)
print "part_vel",part_vel

x_contour, y_contour= np.linspace(0,Nx,Nx+1), np.linspace(0,Ny,Ny+1)
X, Y = np.meshgrid(x_contour, y_contour)
    
    
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
        
        
        Qgrid[i,j] += (1-cj)*(1-ci)
        Qgrid[i+1,j] += (1-cj)*ci
        Qgrid[i,j+1] += cj*(1-ci)
        Qgrid[i+1,j+1] += cj*ci
    x_plot = part_pos[:,0]
    y_plot = part_pos[:,1]
    #plt.plot(x_plot, y_plot, "ko")
    #plt.imshow(Qgrid)
    #plt.show()
    #plt.gca().invert_yaxis()
    #plt.plot(xj_plot, yi_plot, "r+", ms=12)
    #plt.contour(X,Y,Qgrid)
    
    return jcj_pos, ici_pos, Qgrid


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
    

    Ex *= -1.0/(dx*dy)
    Ey *= -1.0/(dx*dy)
    
    return Ex, Ey



def Epts(Ex,Ey,jcj,ici): 
    Epts = np.empty((Npt,2),float)

    for n in range(Npt):
        j = int(jcj[n,0])
        i = int(ici[n,0])

        cj = jcj[n,1]
        ci = ici[n,1]

        Epts[n,0] = Ex[i,j]*(1-ci)*(1-cj) + Ex[i+1,j]*ci*(1-cj) + Ex[i,j+1]*(1-ci)*cj + Ex[i+1,j+1]*cj*ci
        Epts[n,1] = Ey[i,j]*(1-ci)*(1-cj) + Ey[i+1,j]*ci*(1-cj) + Ey[i,j+1]*(1-ci)*cj + Ey[i+1,j+1]*cj*ci
    Epts *= 1.0
    return Epts


#fig = plt.figure()
#ax = fig.add_subplot(111)
plt.ion()
fig = plt.figure()
ax = fig.add_subplot(111)
      

def evolve(t_final):
    part_pos_0 = part_pos
    part_vel_0 = part_vel
    part_xv_0 = np.hstack([part_pos_0, part_vel_0])

    t = 0.0
    while t < t_final:
        
        jcj_0, ici_0, Qgrid_0 = Qgrid(part_pos_0)
        rho_0 = Qgrid_0 
        phi_0 = phi_fourier(rho_0)
        Efx_0, Efy_0 = Efield(phi_0)
        Epts_0 = 10.0* Epts(Efx_0,Efy_0, jcj_0, ici_0)

        print "before pos 10",part_pos_0[10,:]
        print "before vel 10",part_vel_0[10,:]
        print "Ex, Ey on pos 10",Epts_0[10,:]
        
        for n in range(Npt):
            part_vel_0 += dt * qm * Epts_0[n,:]

        part_pos_0 += dt * part_vel_0

        print "new pos 10",part_pos_0[10,:]
        print "new vel 10",part_vel_0[10,:]


        for n in range(Npt):
            if part_pos_0[n,0] >= (Nx)*dx:
                part_pos_0[n,0] -= (Nx-1)*dx
            elif part_pos_0[n,0] <= dx:
                part_pos_0[n,0] += (Nx-2)*dx
            elif part_pos_0[n,1] >= (Ny*dy):
                part_pos_0[n,1] -= (Ny-1)*dy
            elif part_pos_0[n,1] <= dy:
                part_pos_0[n,1] += (Ny-2)*dy


        
        ax.imshow(Qgrid_0)
        plt.gca().invert_yaxis()
        plt.pause(.0001)
        ax.plot(part_pos_0[:,0], part_pos_0[:,1], "k.")
        plt.pause(.0001)
       
        #plt.pause(.0001)
        ax.contour(X,Y,phi_0)
        plt.pause(.0001)
        ax.quiver(X,Y,Efx_0,Efy_0,color='k')     
         
        plt.show()
        plt.pause(2.0)

        del ax.collections[:]
    
        
        print "tstep:",t
        t += dt

evolve(20*dt)

    




