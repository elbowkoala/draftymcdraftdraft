import numpy as np
import math
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

Nx = 31     #Number of x-grid
Ny = 31     #Number of y-grid
Npt = 2000    #Number of actual particles
L = 5.0
dx = float(L)/Nx
dy = float(L)/Ny
q = -1.0    #Charge per actual particle
m= 1.0
qm = q*m
B0 = 0.0   #Set external B-field (z-dir only)
omega = q*B0/m
target = 0.01  #Target accuracy for potential field 
V0 = 0.0    #Boundary potential
dt = .1    

part_pos = np.zeros((Npt, 2),float)
part_vel = np.zeros((Npt, 2),float)

part_pos = 5.0*(np.random.rand(Npt,2))
part_vel = np.random.rand(Npt,2)
x_pos = part_pos[:,0]
y_pos = part_pos[:,1]
part_xv = np.hstack([part_pos, part_vel])
#print "part_pos: ",part_pos
#print "part_vel: ",part_vel
#print "part_xv: ",part_xv

def Qgrid(part_pos):
    Qgrid = np.zeros((Ny+1,Nx+1),float)
    for n in range(Npt):
        j = int((part_pos[n,0]/float(dx)))
        i = int((part_pos[n,1]/dy))
        
        if i > Ny-1:
            i = Ny-1
            part_pos[n,1] = float(i*dy)
            part_vel[n,1] = -part_vel[n,1]
        elif i < 0:
            i = 0
            part_pos[n,1] = float(i*dy)
            part_vel[n,1] = -part_vel[n,1]
        ci = float((part_pos[n,1]/float(dy))) - i
        
        if j > Nx-1:
            j = Nx-1
            part_vel[n,0] = float(j*dx)
            part_vel[n,0] = -part_vel[n,0]
        elif j < 0:
            j = 0
            part_pos[n,0] = float(j*dx)
            part_vel[n,0] = -part_vel[n,0]
        cj = float((part_pos[n,0]/dx)) - j
        
        Qgrid[i,j] += (1-cj)*(1-ci)
        Qgrid[i+1,j] += (1-cj)*ci
        Qgrid[i,j+1] += cj*(1-ci)
        Qgrid[i+1,j+1] += cj*ci
    return Qgrid

Qgrid_1 = Qgrid(part_pos)

#plt.plot(x_pos/dx,y_pos/dx,"k.")
#plt.imshow(Qgrid_1)
#plt.colorbar()
#plt.show()



rho_1 = Qgrid_1 / (dx*dy)

def phi_fourier(rho):
    
    #rho_f = np.empty((Ny+1,Nx+1),complex
    print "shape rho: ",np.shape(rho)
    rho_f = np.fft.rfft2(rho)
    print "shape rho_f: ",np.shape(rho_f)
    phi_f = np.empty_like(rho_f)
    print "shape phi_f: ",np.shape(phi_f)
    
    
    def W(x):
        y = np.exp(1.0j*2.0*np.pi*(x)/float(Nx+1))
        return y.real
    
    for m in range(len(rho_f)):
        for n in range(len(rho_f[0])):
            phi_f[m,n] = (dx*dy*rho_f[m,n])/(4.0 - W(m+1) - W(-m-1) - W(n+1) - W(-n-1))
    
    phi_i = np.fft.irfft2(phi_f)
    print "shape phi_i: ",np.shape(phi_i)
    
    return phi_i
print "shape of rho_1: ",np.shape(rho_1)
print "shape of phi_f: ",np.shape(phi_fourier(rho_1))



    
phi_2 = phi_fourier(rho_1)
print "shape phi_2 which is density plotted: ",np.shape(phi_2)
x_contour, y_contour= np.linspace(0,Nx,Nx+1), np.linspace(0,Ny,Ny+1)
X, Y = np.meshgrid(x_contour, y_contour)
#fig = plt.figure()
#ax = fig.gca(projection = '3d') 
#surf = ax.plot_surface(X, Y, phi_2) 
#ax.view_init(elev=60, azim=50)
#ax.dist=8 
#plt.show()
plt.contour(X,Y,phi_2)
plt.imshow(phi_2)
#plt.show()
#plt.imshow(phi_2)

    


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

print np.shape(phi(rho_1))

#phi_1 = phi(rho_1)
#x_contour, y_contour= np.linspace(0,Nx+1,Nx+1), np.linspace(0,Ny+1,Ny+1)
#plt.imshow(phi_1)
#plt.contour(x_contour,y_contour,phi_1)
#plt.show()

   


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

    
Ex_1,Ey_1 = Efield(phi_2)
print "shape of Ex_1, Ey_1, which is quiver plotted: ",np.shape(Ex_1),np.shape(Ey_1)
plt.quiver(X, Y, Ex_1, Ey_1, color='k', headlength=3)
plt.gca().invert_yaxis()
#plt.title('V contour/density with E arrows')
#plt.contour(X, Y, Ex_1, Ey_1)
plt.show()



