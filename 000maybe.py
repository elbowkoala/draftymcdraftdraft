

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from math import sqrt
from mpl_toolkits.mplot3d import axes3d



q = -10.0 #-1.602e-19            #Charge of electron
m = .10 #9.11e-20              #Mass of electron
qm = float(q)/m           #Charge to mass ratio
kB = 1.0 #1.38064852e-23       #Boltzmann's constant
epsilon = 1.0 #8.85418782e-12  #Permittivity of vacuum
Te =  11.0              #Electron temperature (1K = 8.6e-5 eV)
B0 = 0.0                  #Set external B-field (z-dir only)
ne = 1.0e8               #electron density per cubic meter
v_thermal = sqrt(kB*Te/m) #electron thermal energy
omega_c = qm*B0           #Bfield frequency
omega_p = 1.0 #sqrt(ne*q**2/(epsilon*m)) #Plasma frequency
l_de = v_thermal/omega_p  #Debye length
Nx = 5                   #Number of x-grid
Ny = 5                   #Number of y-grid
dx = 1.0 #l_de
dy = 1.0 #l_de
L = Nx * dx
Npt = 5*Nx*Ny           #Number of cloud particles per grid cell
dt = 0.0001/omega_p
V0 = 10000000000.0
omdt = omega_c * dt

print "v_th=",v_thermal
print "omega_p=",omega_p
print "dx=",dx
print "l_de",l_de
print "L=",L
print "dt=",dt


part_posi = L*(np.random.rand(Npt,2))
part_vel = np.random.rand(Npt,2)
part_vel -= np.ones((Npt,2),float)
part_vel *= v_thermal *100
part_posi[0,0] = dx/100.0
part_posi[:,0] *= 0.5

part_pos = part_posi
part_vel[:,0] = 0.0 #v_thermal
part_vel[:,1] = 0.0
      

    
    
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
    #plt.imshow(Qgrid)
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



def phi(rho):
    target = 1e-5
    phi = np.zeros((Ny+3,Nx+3),float)
    #phi[0,:] = V0
    phiprime = np.empty((Ny+3,Nx+3),float)        

    delta = 1.0
    while delta > target:

        for i in range(Nx+3):
            for j in range(Ny+3):
                if i==0 or i==Ny+2 or j==0 or j==Nx+2:
                    phiprime[i,j] = phi[i,j]
                else:
                    phiprime[i,j] = 0.25*(phi[i+1,j] + phi[i-1,j] \
                                          + phi[i,j+1]+phi[i,j-1] \
                                          + .25*rho[i-1,j-1]*dx*dy/float(epsilon))
        delta = np.max(abs(phi-phiprime))
        phi,phiprime = phiprime, phi
    x_test = np.linspace(-1,Nx+2,Nx+3)
    y_test = np.linspace(-1, Ny+2, Ny+3)
    XX, YY = np.meshgrid(x_test,y_test)

    #plt.contour(XX,YY,phi,colors='y')
    return phi

def rho_vec(rho):
    h2 = dx*dy
    y0b = 0.0 #y=0 boundary
    yLb = 0.0 #y=L boundary
    x0b = 0.0 #x=0 boundary
    xLb = 0.0 #x=L boundary
    NN = (Nx+3)*(Ny+3)
    VV = np.empty((NN,1),float)  #AX=V, this is V
    for i in range(1,Ny+2):
        VV[(i*(Nx+3))] = x0b
        VV[(i*(Nx+3)+(Nx+2))] = x0b
        for j in range(1,Nx+1):
            VV[(i*(Nx+3)+j)] = .25*rho[(i-1),(j-1)]*h2/float(epsilon)
    for i in range(Ny+3):
        VV[i] = y0b
        VV[NN-i-1] = yLb
    print VV
    return VV

def A_mat():
    A = np.eye(3)
    print A

        
    


    

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
        j = jcj[n,0]
        i = ici[n,0]

        cj = jcj[n,1]
        ci = ici[n,1]

        Epts[n,0] = Ex[i,j]*(1-ci)*(1-cj) + Ex[i+1,j]*ci*(1-cj) + Ex[i,j+1]*(1-ci)*cj + Ex[i+1,j+1]*cj*ci
        Epts[n,1] = Ey[i,j]*(1-ci)*(1-cj) + Ey[i+1,j]*ci*(1-cj) + Ey[i,j+1]*(1-ci)*cj + Ey[i+1,j+1]*cj*ci

    return Epts








x_contour, y_contour= np.linspace(0,Nx,Nx+1), np.linspace(0,Ny,Ny+1)
X, Y = np.meshgrid(x_contour, y_contour)








#fig = plt.figure()
#ax = fig.add_subplot(111)
plt.ion()
fig = plt.figure()

t_final = 100*dt
t=0.0
while t < t_final:
    
    ax = fig.add_subplot(111)
    jcj_0, ici_0, Qgrid_0 = Qgrid(part_pos)
    rho_0 = Qgrid_0 / (dx*dy)
    phi_0 = phi_fourier(rho_0)
    Efx_0, Efy_0 = Efield(phi_0)
    Epts_0 = Epts(Efx_0,Efy_0, jcj_0, ici_0)
    
    part_xv_0 = np.hstack([part_pos, part_vel])
            
    r0 = part_xv_0[:,0:2]
    v0 = part_xv_0[:,2:4]
    
    x_plot = part_pos[:,0]
    y_plot = part_pos[:,1]
    
    ax.clear()
    
    ax.plot(x_plot, y_plot, "w.")
    ax.contour(X,Y,phi_0)
    ax.quiver(X,Y,Efx_0,Efy_0,color='k')
    
    
    


    #ax.quiver(X,Y,Ex_0,Ey_0)   
    #plt.gca().invert_yaxis()  
    #plt.plot(r0[:,0],r0[:,1],"ko")  
    #plt.show()
            
    vi1 = v0 + 0.5* dt* qm* Epts_0
    vi2 = np.empty_like(vi1)
            
    for n in range(Npt):
        vi2[n,0] = float(np.cos(omdt)*vi1[n,0]) + float(np.sin(omdt)*vi1[n,1])
        vi2[n,1] = -float(np.sin(omdt)*vi1[n,0]) + float(np.cos(omdt)*vi1[n,1])
    v_ph = vi2 + 0.5 * dt * qm * Epts_0
    r0 += dt * v_ph 
    
    for n in range(Npt):
        if r0[n,0] >= (Nx)*dx:
            r0[n,0] = r0[n,0] - (Nx-1)*dx
        elif r0[n,0] <= dx:
            r0[n,0] = r0[n,0] + (Nx-2)*dx
        elif r0[n,1] >= (Ny*dy):
            r0[n,1] = r0[n,1] - (Ny-1)*dy
        elif r0[n,1] <= dy:
            r0[n,1] = r0[n,1] + (Ny-2)*dy

    part_pos = r0
    part_vel = v_ph    
    
    plt.show()
    plt.pause(.10)
    print "tstep:",t
    t += dt


    




