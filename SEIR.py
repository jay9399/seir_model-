import numpy as np

class SIER:

    def __init__(self,beta,alpha,gamma,S0,E0,I0,R0,rho=1):
        #beta - the contact rate (how many contacts an infected person has)
        #alpha - the rate at which an exposed person becomes infected
        #gamma - recovery rate
        #S0: The initial number of susceptible individuals.
        #E0: The initial number of exposed individuals.
        #I0: The initial number of infected individuals.
        #R0: The initial number of recovered (or dead) individuals.
        #rho: A function (or constant) that models the effect of social distancing. 
        # This can reduce transmission by lowering the contact rate.
        self.initial_conditions=[S0,E0,I0,R0]

        #beta,alpha and gamma can,maybe vary overtime (functions)
        #if a constant value is passed, it will transform to a function

        if(isinstance(beta, (int,float))):
            self.beta=lambda t: beta
        else:
            self.beta=beta
        
        if(isinstance(alpha, (int,float))):
            self.alpha=lambda t: alpha
        else:
            self.alpha=alpha

        if(isinstance(gamma, (int,float))):
            self.gamma=lambda t: gamma
        else:
            self.gamma=gamma

        if(isinstance(rho, (int,float))):
            self.rho=lambda t: rho
        else:
            self.rho=rho

        # This method is used to simulate the SEIR model over time.
        # The method takes an argument timestamp, 
        # which is an array or list of time points where the model should be evaluated.

        #use the semi-implicit Euler method to solve the differential equations that govern the model.

    def compile(self, timestamp):
        n=len(timestamp)

        u=np.zeros((n,len(self.initial_conditions)))
        #creating a n*4 matrix denoting s,i,e,r conditions
        #at each time stamp
        #(proportion of population at s,i,e,r conditions)
        #initially at zeros

        u[0,:]=self.initial_conditions
        #this sets the first row of u to the initial conditions
        #(S0,E0,I0,R0), this corresponds to the initial population 
        # sizes

        dt=timestamp[1]-timestamp[0]

        #this calculates the time stamp dt by subtracting the first time point (timestamp[0])
        #from the second time point(timestamp[1]). This assumes that the timestamps
        # are uniform
        
        for t in range(n-1):
        #this starts a loop that iterates over time stamp from 0 to n-2
        #(all but the last time point)
            u[t+1,0]=u[t,0]-(self.rho(t)*self.beta(t)*u[t,0]*u[t,2])*dt

        #the equation updates the susceptible population S based on the 
        #number of new infections.
        # S{t + 1} = S{t} - rho(t)*beta{t}*S{t}*I{t}*dt

            u[t+1,1]=u[t,1]+(self.rho(t)*self.beta(t)*u[t,0]*u[t,2]-self.alpha(t)*u[t,1])*dt

        # E{t + 1} = E{t} + (rho(t)*beta{t}*S{t}*I{t} - alpha(t)*E{t})*dt

            u[t+1,2]=u[t,2]+(self.alpha(t)*u[t,1]-self.gamma(t)*u[t,2])*dt
        
        # R{t + 1} = R{t} + gamma(t)*I{t}*dt

            u[t+1,3] = u[t,3] + self.gamma(t)*u[t,2]*dt

        return u
        



