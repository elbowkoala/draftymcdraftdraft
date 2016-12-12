import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from math import sqrt
from mpl_toolkits.mplot3d import axes3d



q = -1.602e-19            #Charge of electron
m = 9.11e-20              #Mass of electron
qm = float(q)/m           #Charge to mass ratio
kB = 1.38064852e-23       #Boltzmann's constant
epsilon = 8.85418782e-12  #Permittivity of vacuum
Te =  11.0              #Electron temperature (1K = 8.6e-5 eV)
B0 = 0.0                  #Set external B-field (z-dir only)
ne = 1.0e8               #electron density per cubic meter
v_thermal = sqrt(kB*Te/m) #electron thermal energy
omega_c = qm*B0           #Bfield frequency
omega_p = sqrt(ne*q**2/(epsilon*m)) #Plasma frequency
l_de = v_thermal/omega_p  #Debye length
Nx = 31                   #Number of x-grid
Ny = 31                   #Number of y-grid
dx = l_de
dy = l_de
L = Nx * dx
Npt = 10*Nx*Ny           #Number of cloud particles per grid cell
dt = 0.01/omega_p

print "v_th=",v_thermal
print "omega_p=",omega_p
print "dx=",dx
print "l_de",l_de
print "L=",L
print "dt=",dt


      

    
    
def Qgrid(part_pos):
    Qgrid = np.zeros((Ny+1,Nx+1),float)
    for n in range(Npt):
        
        j = int((part_pos[n,0]/float(dx)))
        i = int((part_pos[n,1]/dy))
    
        ci = float((part_pos[n,1]/float(dy))) - i
        cj = float((part_pos[n,0]/dx)) - j
        
        Qgrid[i,j] += (1-cj)*(1-ci)
        Qgrid[i+1,j] += (1-cj)*ci
        Qgrid[i,j+1] += cj*(1-ci)
        Qgrid[i+1,j+1] += cj*ci
    return Qgrid


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
                                          + rho[i,j]*dx*dy )
        delta = np.max(abs(phi-phiprime))
        phi,phiprime = phiprime, phi
    return phi



def Efield(phi):
    Ex = np.zeros((Ny+1,Nx+1),float)
    Ey = np.zeros((Ny+1,Nx+1), float)
    
    for i in range(1,Nx-1):
        for j in range(1,Ny-1):            
            Ey[i,j] = -(phi[i+1,j]-phi[i-1,j])
            Ex[i,j] = -(phi[i,j+1]-phi[i,j-1])
    Ex *= (1/(2.0*dx))
    Ey *= (1/(2.0*dy))
    
    Ex[:,0] = -(1.0/dx)*(phi[1,:]-phi[0,:])
    Ex[:,Nx-1] = -(1.0/dx)*(phi[Nx,:]-phi[Nx-1,:])
    Ey[0,:] = -(1.0/dy)*(phi[:,1]-phi[:,0])
    Ey[Ny-1,:] = -(1.0/dy)*(phi[:,Ny]-phi[:,Ny-1])
    
    return Ex, Ey



def Epts(part_pos): 
    Epts = np.empty((Npt,2),float)
    Qgrid_a = Qgrid(part_pos)
    rho_a = Qgrid_a / (dx*dy)
    phi_a = phi_fourier(rho_a)
    Ex, Ey = Efield(phi_a)

    for n in range(Npt):
        j = int((part_pos[n,0]/float(dx)))
        i = int((part_pos[n,1]/dy))
        
        ci = float((part_pos[n,1]/float(dy))) - i
        cj = float((part_pos[n,0]/dx)) - j

        Epts[n,0] = Ex[i,j]*(1-ci)*(1-cj) + Ex[i+1,j]*ci*(1-cj) + Ex[i,j+1]*(1-ci)*cj + Ex[i+1,j+1]*cj*ci
        Epts[n,1] = Ey[i,j]*(1-ci)*(1-cj) + Ey[i+1,j]*ci*(1-cj) + Ey[i,j+1]*(1-ci)*cj + Ey[i+1,j+1]*cj*ci
    return Ex, Ey, Epts








x_contour, y_contour= np.linspace(0,Nx,Nx+1), np.linspace(0,Ny,Ny+1)
X, Y = np.meshgrid(x_contour, y_contour)


omdt = omega_c*dt
t_final = .1

part_pos = L*(np.random.rand(Npt,2))
part_vel = np.random.rand(Npt,2)
part_vel -= np.ones((Npt,2),float)
part_vel *= v_thermal
part_vel = np.zeros((Npt,2),float)


plt.ion()
fig=plt.figure()
ax = fig.add_subplot(111)





t=0.0
while t < t_final:

    Qgrid_0 = Qgrid(part_pos)
    rho_0 = Qgrid_0 / (dx*dy)
    phi_0 = phi_fourier(rho_0)
    Ex_0, Ey_0, Epts_0 = Epts(part_pos)

    part_xv_0 = np.hstack([part_pos, part_vel])
            
    r0 = part_xv_0[:,0:2]
    v0 = part_xv_0[:,2:4]
            
    vi1 = v0 + 0.5* dt* qm* Epts_0
    vi2 = np.empty_like(vi1)
            
    for n in range(Npt):
        vi2[n,0] = float(np.cos(omdt)*vi1[n,0]) + float(np.sin(omdt)*vi1[n,1])
        vi2[n,1] = -float(np.sin(omdt)*vi1[n,0]) + float(np.cos(omdt)*vi1[n,1])

    v_ph = vi2 + 0.5 * dt * qm * Epts_0
    r0 += dt * v_ph 

    for n in range(Npt):
        if r0[n,0] > L:
            r0[n,0] -= L
        elif r0[n,0] < 0:
            r0[n,0] += L
        elif r0[n,1] > L:
            r0[n,1] -= L
        elif r0[n,1] < 0:
            r0[n,1] += L

    part_pos = r0
    part_vel = v_ph
    
    x_step, y_step = part_pos[:,0], part_pos[:,1]
    ax.clear()
    ax.imshow(phi_0)
    ax.contour(X,Y,phi_0)
    ax.quiver(X,Y,Ex_0,Ey_0)
    plt.gca().invert_yaxis()
    
    plt.show()
    plt.pause(0.001)
    
    t += dt


    

