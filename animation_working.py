import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
import math
from mpl_toolkits.mplot3d import axes3d


Nx = 31     #Number of x-grid
Ny = 31     #Number of y-grid
Npt = 1000    #Number of actual particles
L = 10.0
dx = float(L)/Nx
dy = float(L)/Ny
q = -100.0    #Charge per actual particle
m= .10
qm = float(q)/m
B0 = 0.0   #Set external B-field (z-dir only)
omega = q*B0/m
target = 0.01  #Target accuracy for potential field 
V0 = 0.0    #Boundary potential
dt = 0.005
      

    
    
def Qgrid(part_pos):
    Qgrid = np.zeros((Ny+1,Nx+1),float)
    for n in range(Npt):
        j = int((part_pos[n,0]/float(dx)))
        i = int((part_pos[n,1]/dy))
        
        if i > Ny-1:
            i = Ny-1
            part_pos[n,1] = float(i*dy)
            part_vel[n,1] = -abs(part_vel[n,1])
        elif i < 0:
            i = 0
            part_pos[n,1] = float(i*dy)
            part_vel[n,1] = abs(part_vel[n,1])
        ci = float((part_pos[n,1]/float(dy))) - i
        
        if j > Nx-1:
            j = Nx-1
            part_vel[n,0] = float(j*dx)
            part_vel[n,0] = -abs(part_vel[n,0])
        elif j < 0:
            j = 0
            part_pos[n,0] = float(j*dx)
            part_vel[n,0] = abs(part_vel[n,0])
        cj = float((part_pos[n,0]/dx)) - j
        
        Qgrid[i,j] += (1-cj)*(1-ci)
        Qgrid[i+1,j] += (1-cj)*ci
        Qgrid[i,j+1] += cj*(1-ci)
        Qgrid[i+1,j+1] += cj*ci
    return part_pos, part_vel, Qgrid


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
    part_pos_a, part_vel_a, Qgrid_a = Qgrid(part_pos)
    rho_a = Qgrid_a / (dx*dy)
    phi_a = phi_fourier(rho_a)
    Ex, Ey = Efield(phi_a)

    for n in range(Npt):
        j = int((part_pos_a[n,0]/float(dx)))
        i = int((part_pos_a[n,1]/dy))
        
        if i > Ny-1:
            i = Ny-1
            part_pos_a[n,1] = float(i*dy)
            part_vel_a[n,1] = -part_vel_a[n,1]
        elif i < 0:
            i = 0
            part_pos_a[n,1] = float(i*dy)
            part_vel_a[n,1] = -part_vel_a[n,1]
        ci = float((part_pos_a[n,1]/float(dy))) - i
        
        if j > Nx-1:
            j = Nx-1
            part_vel_a[n,0] = float(j*dx)
            part_vel_a[n,0] = -part_vel_a[n,0]
        elif j < 0:
            j = 0
            part_pos_a[n,0] = float(j*dx)
            part_vel_a[n,0] = -part_vel_a[n,0]
        cj = float((part_pos_a[n,0]/dx)) - j

        Epts[n,0] = Ex[i,j]*(1-ci)*(1-cj) + Ex[i+1,j]*ci*(1-cj) + Ex[i,j+1]*(1-ci)*cj + Ex[i+1,j+1]*cj*ci
        Epts[n,1] = Ey[i,j]*(1-ci)*(1-cj) + Ey[i+1,j]*ci*(1-cj) + Ey[i,j+1]*(1-ci)*cj + Ey[i+1,j+1]*cj*ci
    return part_pos_a, part_vel_a, phi_a, Ex, Ey, Epts








x_contour, y_contour= np.linspace(0,Nx,Nx+1), np.linspace(0,Ny,Ny+1)
X, Y = np.meshgrid(x_contour, y_contour)


omdt = omega*dt
t_final = .1

part_pos = L*(np.random.rand(Npt,2))
part_vel = np.random.rand(Npt,2)
part_vel -= np.ones((Npt,2),float)
part_vel *= 50.0
part_pos[0,:] = (L/2.0), (L/2.0)
part_vel[0,:] = 0.0, 50.0

plt.ion()
fig=plt.figure()
ax = fig.add_subplot(111)





t=0.0
while t < t_final:
    part_pos_0, part_vel_0, Qgrid_0 = Qgrid(part_pos)
    rho_0 = Qgrid_0 / (dx*dy)
    phi_0 = phi_fourier(rho_0)
    Ex_0, Ey_0 = Efield(phi_0)
    part_pos_00, part_vel_00, phi_00, Ex_00, Ey_00, Epts_00 = Epts(part_pos_0)

    part_xv_00 = np.hstack([part_pos_00, part_vel_00])
            
    r0 = part_xv_00[:,0:2]
    v0 = part_xv_00[:,2:4]
            
    vi1 = v0 + 0.5* dt* qm* Epts_00
    vi2 = np.empty_like(vi1)
            
    for n in range(Npt):
        vi2[n,0] = float(np.cos(omdt)*vi1[n,0]) + float(np.sin(omdt)*vi1[n,1])
        vi2[n,1] = -float(np.sin(omdt)*vi1[n,0]) + float(np.cos(omdt)*vi1[n,1])

    v_ph = vi2 + 0.5 * dt * qm * Epts_00
    r0 += dt * v_ph 

    part_pos = r0
    part_vel = v_ph
    x_step, y_step = part_pos[:,0], part_pos[:,1]
    ax.clear()
    ax.imshow(phi_0)
    ax.contour(X,Y,phi_0)
    ax.quiver(X,Y,Ex_0,Ey_0)
    plt.gca().invert_yaxis()
    
    plt.show()
    plt.pause(0.1)
    
    t += dt


    

