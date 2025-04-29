import numpy as np
import matplotlib.pyplot as plt

from SEIR import SIER

t_max = 10
dt=0.1
t=np.linspace(0,t_max,int(t_max/dt)+1)

#initial values
#these are normalized population values
N=10000
S0=1-(1/N)
E0=(1/N)
I0=0
R0=0

#initial parameters
#S0,E0,I0,R0: The initial proportions of the population in each
#compartment
#S0 : The fraction of the population initially susceptible to the 
#disease (as small portion, defined as 1/N)
#I0 : The fraction of the population initially infected (none,so 0)
#R0 : The fraction of population initially recovered (none,so 0)

        #beta - the contact rate (how many contacts an infected person has)
        #alpha - the rate at which an exposed person becomes infected
        #gamma - recovery rate
        #S0: The initial number of susceptible individuals.
        #E0: The initial number of exposed individuals.
        #I0: The initial number of infected individuals.
        #R0: The initial number of recovered (or dead) individuals.
        #rho: A function (or constant) that models the effect of social distancing. 

alpha=0.2
gamma=0.5
beta=1.75

model=SIER(beta,gamma,alpha,S0,E0,I0,R0)
U=model.compile(t)

fig=plt.figure()
S_t=plt.plot(t,U[:,0]) #Susceptible population with time
E_t=plt.plot(t,U[:,1]) #Exposed population with time
I_t=plt.plot(t,U[:,2]) #Infected population with time
R_t=plt.plot(t,U[:,3]) #Recoved population with time

plt.legend(['Susceptible','Exposed','Infected','Recovered'])
plt.xlabel('Days')
plt.ylabel('Fraction of Population')
plt.title('SIER model for COVID-19 without Social Distancing')
fig.savefig('without_social_distancing.png')
plt.show()

rho_list=[0.5]
#colors=['red','green','blue']
legend=[]
fig=plt.figure()

for i,rho in enumerate(rho_list):
    model=SIER(beta,alpha,gamma,S0,E0,I0,R0,rho)
    U=model.compile(t)
    E_t=plt.plot(t,U[:,1])
    I_t=plt.plot(t,U[:,2],':')
    legend.append(f'Exposed(rho={rho})')
    legend.append(f'Infected (rho={rho})')

plt.xlabel('Days')
plt.ylabel('Fraction Of Population')
plt.legend(legend)
plt.title('SIER model for COVID-19 with Social Distancing')
fig.savefig('social_distancing.png')
plt.show()
