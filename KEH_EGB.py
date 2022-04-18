import numpy as np
import scipy
import math

from scipy import integrate
from scipy import optimize
from scipy.interpolate import PchipInterpolator
from scipy.interpolate import interp1d
from scipy.interpolate import UnivariateSpline
from scipy.special import legendre
from scipy.special import eval_legendre
from scipy.special import factorial2
from urllib.request import urlopen
from numpy import *
from scipy import signal


#DEFINE CONSTANTS

c=2.9979e10
G=6.67408e-8
Msun=1.989e33
Length = G*Msun/c**2
Time = Length/c
Density = Msun/Length**3
length_squared = 1.346790806509621e+13/pow(100000,2)  # (km^2)
kscale = 1.112668301525780e-36
kappa = 1.346790806509621e+13


url_intermediate_e_c = "https://raw.github.com/esmyrnio/py_KEH_EGB/main/guess_solutions/intermediate_e_c.txt"
url_high_e_c = "https://raw.github.com/esmyrnio/py_KEH_EGB/main/guess_solutions/high_e_c.txt"
url_low_e_c = "https://raw.github.com/esmyrnio/py_KEH_EGB/main/guess_solutions/low_e_c.txt"



#LOAD EOS FILE

def load_eos(eos_file):

    path="EoS/"

    f_eos = open(path+eos_file,"r")
    rho,p,h,n0 = np.loadtxt(path+eos_file, skiprows=1, unpack=True)

    global log_e, log_p, log_h, log_n0

    log_e=np.log10(rho*c*c*kscale)
    log_p=np.log10(p*kscale)
    log_h=np.log10(h/(c*c))
    log_n0=np.log10(n0)   

#GUESS SOLUTION

def guess(e_c):

    if e_c<=1.0:
        url = urlopen(url_low_e_c)
    elif e_c<=1.8:
        url = urlopen(url_intermediate_e_c)
    else:
        url = urlopen(url_high_e_c)

    global data_sol
    data_sol = np.loadtxt(url, delimiter=' ', dtype=float)


#DEFINE FIRST ORDER DERIVATIVE

def di(i,f):
    
    if i == 0:
        return (f[i+1]-f[i])/(ds)
    elif i == N-1:
        return (f[i]-f[i-1])/(ds)
    else:
        return (f[i+1]-f[i-1])/(2*ds)


#DEFINE SOURCE TERMS FOR METRIC POTENTIALS AND SCALAR (1st order scalar)

def source_nu(i,s,mu_t,nu_t,phi_t,dnds,dnds2,dmds,dmds2,dfds,dfds2,e_aw,p_aw,r_e,a):
    
    c = 4*np.pi
    
    p = p_aw
    eps = e_aw
    
    emu = np.exp(2*mu_t[i])
    memu = np.exp(-2*mu_t[i])
    
    r = np.sqrt(r_e)*s[i]/(1-s[i])
    r2 = r_e*pow(s[i]/(1-s[i]),2)
    
    ss = pow(1-s[i],2)/np.sqrt(r_e)
    dn = ss*dnds[i]
    dm = ss*dmds[i]
    dphi = ss*dfds[i]
    
    ddphi = pow(1-s[i],4)*dfds2[i]/r_e - dfds[i]*2*pow(1-s[i],3)/r_e
    ddm =   pow(1-s[i],4)*dmds2[i]/r_e - dmds[i]*2*pow(1-s[i],3)/r_e
    
    
    term = 2*dn*r

    if i!=0:
        
        return r*(emu*(c*emu*r2*(3*p+eps)-r*dn*(2+r*(dm+dn))+a*memu*\
                (dphi*(2*r2*pow(dm,3)-2*r2*pow(dm,2)*(dn+dphi)-2*r*pow(dn,2)\
                  *(-2+r*dphi)-r*(pow(dphi,2)*(-4+r*dphi)+4*ddm)+2*dm\
                *(-2+r*(2*r*pow(dn,2)+dn*(4-r*dphi)+dphi*(-2+r*dphi)-2*r*ddm))\
                  +2*dn*(2+r*(dphi*(-4+r*dphi)+2*r*ddm)))+2*r*(-r*pow(dm,2)+2*dm*\
                  (-1+r*dn)+r*pow(dphi,2)+dn*(2-2*r*dphi))*ddphi)))\
                /(emu*r+2*a*dphi*(-2-2*r*dm+r*dphi)) +term
    
    else:
        
        return 0
    

def source_mu(i,s,mu_t,nu_t,phi_t,dnds,dnds2,dmds,dmds2,dfds,dfds2,e_aw,p_aw,r_e,a):

    c = 8*np.pi
    
    p = p_aw
    eps = e_aw
    
    emu = np.exp(2*mu_t[i])


    r = np.sqrt(r_e)*s[i]/(1-s[i])
    r2 = r_e*pow(s[i]/(1-s[i]),2)
    
    ss = pow(1-s[i],2)/np.sqrt(r_e)
    dn = ss*dnds[i]
    dm = ss*dmds[i]
    dphi = ss*dfds[i]
    
    
    ddphi = pow(1-s[i],4)*dfds2[i]/r_e - dfds[i]*2*pow(1-s[i],3)/r_e
    
    term = 2*dm*r

    if i!=0:
        
        return -r*((c*pow(emu,2)*r2*eps+emu*r*dm*(4+r*dm)+a*(2*dm-dphi)\
                 *dphi*(-4+r2*(2*pow(dm,2)-2*dm*dphi+pow(dphi,2)))\
               -4*a*r*(dm-dphi)*(2+r*dm-r*dphi)*ddphi)/\
                 (2*(emu*r+2*a*dphi*(-2-2*r*dm+r*dphi)))) + term
    
    else:
        
        return 0
                       
def source_phi(i,s,dmds,r_e):

    ss = pow(1-s[i],2)/np.sqrt(r_e)
    dm = ss*dmds[i]
    
    return dm

#DEFINE FUNCTION TO INTEGRATE SCALAR SOURCE

def integrate_phi_source():

    ss = pow(1-s,2)/np.sqrt(r_e)
    ph = np.zeros(N)
    
    for i in range(0,N):
        ph[i] = integrate.simps(sources_ph[:i+1]/ss[:i+1],dx=ds) 
        
    return ph


#CUBIC HERMITE INTERPOLATION OF TABULATED EOS  

def p_at_h(hh):
    return pow(10,PchipInterpolator(log_h,log_p)(np.log10(hh)))

def e_at_p(pp):
    return pow(10,PchipInterpolator(log_p,log_e)(np.log10(pp)))

def h_at_p(pp):
    return pow(10,PchipInterpolator(log_p,log_h)(np.log10(pp)))

def p_at_e(ee):
    return pow(10.0,PchipInterpolator(log_e,log_p)(np.log10(ee)));



#LOAD EOS
eos_name = str(input())
load_eos(eos_name)

#CHOOSE E_C
ec_t = double(input())

guess(ec_t)

#GRID
s = data_sol[:,2] # 1301x1
N = len(s)
ds = s[1]-s[0]
surface_indx = np.argmin((s-1/2)**2)

#BUILD GUESS
re_t = data_sol[surface_indx,1]/np.sqrt(kappa)
re_guess = re_t
p_t = data_sol[:,13]*kscale # KAPPA*G/(C*C*C*C)
eps_t = data_sol[:,12]*kappa 
raw_t = data_sol[:,4]
g_t = data_sol[:,5]
nu_t = (g_t+raw_t)/2
mu_t = (g_t-raw_t)/2
phi_t = mu_t

#COMPUTE CENTER
eps_t[0] = ec_t
p_t[0] = p_at_e(ec_t)


# DEFINE THEORY PARAMETER (IN KM^2)
a = double(input())/length_squared # KM^2/length^2

# DEFINE MONITORING FACTORS
relative_error = double(input())
max_iter = int(input())

# DEFINE UNDER-RELAXATION FACTOR

under_relaxation_factor = double(input())

#choose print option

print_option = int(input())

iterations = 0

mu_new = np.zeros(N)
nu_new = np.zeros(N)
phi_new = np.zeros(N)

dfds_main = np.zeros(N)
dmds_main = np.zeros(N)
dnds_main = np.zeros(N)

dnds2_main = np.zeros(N)
dfds2_main = np.zeros(N)
dmds2_main = np.zeros(N)

sources_n = np.zeros(N)
sources_m = np.zeros(N)
sources_ph = np.zeros(N)

h_aw = np.zeros(N)
p_aw = np.zeros(N)
e_aw = np.zeros(N)

h_center = h_at_p(p_at_e(ec_t))
h_min = h_at_p(p_t[surface_indx])


# ----- MAIN ITERATION CYCLE ------ #

while True:
    
    # RESCALE
    
    mu_hat = mu_t/pow(re_t,2)
    nu_hat = nu_t/pow(re_t,2)
    phi_hat = phi_t/pow(re_t,2)



    # SQUARED
    r_e = h_center/(nu_hat[surface_indx]-nu_hat[0])
    
    
    # CALCULATE NEW ENTHALPY
    
    for i in range(0,N):
        
    
        h_aw[i]  = h_min + (r_e*(nu_hat[surface_indx]
                           -nu_hat[i]))


    # CALCULATE NEW PRESSURE AND EN_DENSITY
    
    p_aw[:surface_indx+1] = p_at_h(h_aw[:surface_indx+1])
    e_aw[:surface_indx+1] = e_at_p(p_aw[:surface_indx+1])
    
    p_aw[surface_indx+1:] = 0
    e_aw[surface_indx+1:] = 0
    
    
    # CALCULATE DERIVATIVES
    
    for i in range(0,N):
        
        dfds_main[i] = di(i,phi_hat*r_e)
        dmds_main[i] = di(i,mu_hat*r_e)
        dnds_main[i] = di(i,nu_hat*r_e)
        
    # SMOOTH OUT 1ST ORDER DERIVATIVE SOLUTIONS USING A SAVITZKY-GOLAY FILTER
        
    dfds_real = scipy.signal.savgol_filter(dfds_main,31,3)
    dmds_real = scipy.signal.savgol_filter(dmds_main,31,3)
    dnds_real = scipy.signal.savgol_filter(dnds_main,31,3)

    
    for i in range(0,N):
        
        dfds2_main[i] = di(i,dfds_real)
        dmds2_main[i] = di(i,dmds_real)
        dnds2_main[i] = di(i,dnds_real)
        
    # SMOOTH OUT 2ND ORDER DERIVATIVE SOLUTIONS USING A SAVITZKY-GOLAY FILTER
    
    dfds2_real = scipy.signal.savgol_filter(dfds2_main,31,3)
    dmds2_real = scipy.signal.savgol_filter(dmds2_main,31,3)
    dnds2_real = scipy.signal.savgol_filter(dnds2_main,31,3)

    
    # COMPUTE SOURCE TERMS
    
    for i in range(0,N):
        

        sources_n[i] = source_nu(i,s,mu_hat*r_e,nu_hat*r_e,phi_hat*r_e,dnds_real,dnds2_real,dmds_real,dmds2_real,dfds_real,dfds2_real,e_aw[i],p_aw[i],r_e,a)
        sources_m[i] = source_mu(i,s,mu_hat*r_e,nu_hat*r_e,phi_hat*r_e,dnds_real,dnds2_real,dmds_real,dmds2_real,dfds_real,dfds2_real,e_aw[i],p_aw[i],r_e,a)
        sources_ph[i] = source_phi(i,s,dmds_real,r_e)

    
    # COMPUTE NEW METRIC POTENTIALS AND SCALAR
    
    for i in range(1,N):
        
        
        mu_new[i] = -(((1-s[i])/s[i])*integrate.simps((1/pow(1-s[0:i+1],2))\
                 *sources_m[0:i+1],dx=ds)\
                + integrate.simps((pow(1-s[i:],-1)/s[i:])*sources_m[i:],dx=ds))
        

        nu_new[i] = -(((1-s[i])/s[i])*integrate.simps((1/pow(1-s[0:i+1],2))\
                 *sources_n[0:i+1],dx=ds)\
                + integrate.simps((pow(1-s[i:],-1)/s[i:])*sources_n[i:],dx=ds))


    phi_new = integrate_phi_source()
        

    rel_error = abs(1-np.sqrt(np.sqrt(r_e)/re_t))
        
                                                  
    mu_new[0] = -integrate.simps(sources_m[1:]/(s[1:]*(1-s[1:])),dx=ds)
    nu_new[0] =   -integrate.simps(sources_n[1:]/(s[1:]*(1-s[1:])),dx=ds)

    
    iterations = iterations + 1

    if print_option==0:

        print("/-------------------------------------------/")
        print('Iteration #  ',iterations)
        print('error = ',rel_error)
        print('r_e = ',np.sqrt(r_e))
        print("/-------------------------------------------/")

    if rel_error<= relative_error or iterations == max_iter:

        break

    else:
        
        # UNDER-RELAXATION
        
        re_t = np.sqrt(r_e)
        mu_t = under_relaxation_factor*mu_new + (1-under_relaxation_factor)*mu_t
        nu_t = under_relaxation_factor*nu_new + (1-under_relaxation_factor)*nu_t
        phi_t = under_relaxation_factor*phi_new + (1-under_relaxation_factor)*phi_t

r = np.sqrt(r_e)*s/(1-s)

gravitational_mass = 2*r[N-3]*(np.sqrt(np.exp(mu_new[N-3]))-1)*np.sqrt(kappa)*(c*c/G)/Msun
circumferential_radius = np.sqrt(r_e)*np.exp(mu_new[surface_indx])*np.sqrt(kappa)/pow(10,5)

if print_option == 0:

    print("\n");
    print(" %s                  " % eos_name);    
    print(" %6.5e  e_c           (10^15 gr/cm^3)\n" % ec_t);
    print(" %2.2e     relative error            " % relative_error);
    print(" %.1f        max iterations            " % max_iter);
    print(" %.2f         relaxation factor            \n" % under_relaxation_factor);
    print(" %.2f        alpha            (km^2)\n" % (a*length_squared));
    print(" %6.5e  Mass             (M_sun)\n" % gravitational_mass);
    print(" %6.5e  Radius             (km)\n" % circumferential_radius);
else:

    print("\n");
    print(" %s                  " % eos_name);    
    print(" %6.5e  e_c           (10^15 gr/cm^3)\n" % ec_t);
    print(" %2.2e     relative error            " % relative_error);
    print(" %.1f        max iterations            " % max_iter);
    print(" %.2f         relaxation factor            \n" % under_relaxation_factor);
    print(" %.2f        alpha            (km^2)\n" % (a*length_squared));
    print(" %6.5e  Mass             (M_sun)\n" % gravitational_mass);
    print(" %6.5e  Radius             (km)\n" % circumferential_radius);
    print("\n");

    print("-------------------------------------------------------------------------------\n");   
    print("s        r_is        mu         nu        phi        epsilon        pressure\n");
    print("-------------------------------------------------------------------------------\n"); 

    for w in range(0,N):
        print("%5.4e %5.4e %5.4e %5.4e %5.4e %5.4e %5.4e \n" % (s[w], r[w], mu_new[w], nu_new[w], phi_new[w], e_aw[w], p_aw[w]))
