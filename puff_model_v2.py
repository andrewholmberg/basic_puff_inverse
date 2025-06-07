import torch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline 
import copy


class puff_model():
    def __init__(self, source_locations,sensor_locations, sigma, u_func, v_func, qs,dt,spread=False,t_max = 45*60):
        self.corners = [(0,0),(200,0),(0,200),(200,200)]
        self.min_u  = torch.min(torch.abs(u_func(torch.linspace(0,t_max,t_max//dt))))
        self.min_v = torch.min(torch.abs(u_func(torch.linspace(0,t_max,t_max//dt))))
        self.sx = sigma[0]
        self.sy = sigma[1]
        self.sz = sigma[2]
        self.num_source_locs = len(source_locations)
        self.puffs = list()
        self.source_locations = source_locations
        self.sigma = sigma
        self.u_func = u_func
        self.v_func = v_func
        self.q = qs
        self.constant = 1/((2*torch.pi)**(3/2) * self.sy**2*self.sz)
        self.sensor_locations = sensor_locations
        self.dt = dt
        self.spread=spread
        t_span = (0,60*60)
        t_eval = np.linspace(0,60*60,60*60)
        cs_temp = CubicSpline(t_eval,solve_ivp(lambda t,y:u_func(t),t_span,[0],'RK45',t_eval)['y'][0])
        self.u_func_ivp = copy.deepcopy(lambda t: cs_temp(t.detach().numpy()))
        cs_temp1 = CubicSpline(t_eval,solve_ivp(lambda t,y:v_func(t),t_span,[0],'RK45',t_eval)['y'][0])
        self.v_func_ivp = copy.deepcopy(lambda t: cs_temp1(t.detach().numpy()))
        print(self.u_func_ivp(torch.tensor([0])),self.u_func_ivp(torch.tensor([1])))
        print(u_func(0),u_func(1))


    '''obs - nxm matrix - n = number of time samples, m = number of sensors'''
    def compute_exp_term(self,source_idx,sensor_idx,t,t0):
        assert t >= t0
        if t == t0: return 0
        eps = 1e-6
        x,y,z = self.sensor_locations[sensor_idx][0],self.sensor_locations[sensor_idx][1], self.sensor_locations[sensor_idx][2]
        x0,y0,z0 = self.source_locations[source_idx][0],self.source_locations[source_idx][1],self.source_locations[source_idx][2]
        sx,sy,sz = 1+torch.sqrt((t-t0)+eps)*self.sx,1+torch.sqrt((t-t0)+eps)*self.sy,1+torch.sqrt((t-t0)+eps)*self.sz
        sx,sy,sz = self.sx,self.sy,self.sz
        assert sx > 0 and sy > 0 and sz > 0

        to_ret = 1/((2*torch.pi)**(3/2) * sy**2*sz)*torch.exp(torch.tensor(-1*(  (x - x0 - self.u_func(t0)*(t-t0))**2  + (y - y0 - self.v_func(t0)*(t-t0))**2)/(2*sy**2)))*(torch.exp(torch.tensor( -1*((z - z0)**2)/(2*sz**2))) + torch.exp(torch.tensor( -1*((z + z0)**2)/(2*sz**2))))
        assert to_ret >= 0
        return to_ret
    
    def return_qp_matrices(self,obs,obs_t):
        '''TO DO: Add Assertion that obs dtype is torch tensor float'''
        Q = torch.zeros(obs.shape[0]*obs.shape[1],self.num_source_locs)
        num_sensors = obs.shape[1]
        num_obs = obs.shape[0]
        obs = obs.T.reshape(-1,1)

        for s in range(num_sensors):
            for o in range(num_obs): 
                t = 0
                temp = torch.zeros(self.num_source_locs)
                while t < obs_t[o]:
                    temp += torch.tensor([self.compute_exp_term(i,s,obs_t[o],t) for i in range(self.num_source_locs)])*self.dt
                    t += self.dt
                Q[s*num_obs + o,:] = temp
        return 1*Q.T@Q, -2*obs.T@Q, Q
        # Q = torch.round(Q,decimals=)
    
    def return_qp_matrices_new(self,obs,obs_t,spread=False,bias_terms=0,weights = None):
        '''TO DO: Add Assertion that obs dtype is torch tensor float'''
        if weights == None:
            weights = weights = torch.diag(torch.ones(len(obs_t)))
        Q = torch.zeros(obs.shape[0]*obs.shape[1],self.num_source_locs)

        num_sensors = obs.shape[1]
        num_obs = obs.shape[0]
        obs = obs.T.reshape(-1,1)
        obs_t = obs_t.reshape(-1,1)
        tr = obs_t.repeat(num_sensors,1)
        # obs_locs = torch.cat([obs_locs])
        # X = torch.zeros(num_obs*num_sensors,self.num_source_locs)

        X= torch.cat([torch.cat([torch.tensor([self.sensor_locations[i][0],self.source_locations[j][0],self.sensor_locations[i][1],self.source_locations[j][1],
                                               self.sensor_locations[i][2],self.source_locations[j][2]]) for j in range(self.num_source_locs)])
                                               .repeat(num_obs,1) for i in range(num_sensors)])
        XX, YY, ZZ = X[:,[6*i for i in range(self.num_source_locs)]+[6*i+1 for i in range(self.num_source_locs)]], X[:,[6*i+2 for i in range(self.num_source_locs)]+[6*i+3 for i in range(self.num_source_locs)]], X[:,[6*i+4 for i in range(self.num_source_locs)]+[6*i+5 for i in range(self.num_source_locs)]]
        
        t = 0


        while t < obs_t[-1].item():
            t0 = torch.tensor([t]).repeat(num_obs*num_sensors,1)
            eps=1e-10
            
            zero = (tr - t0 > 0).float()

            if spread:
                sx,sy,sz =  self.sx*torch.sqrt(((tr-t0)*zero+eps)),  self.sy*torch.sqrt(((tr-t0)*zero+eps)), self.sz*torch.sqrt(((tr-t0)*zero+eps))
            else:
                sx,sy,sz = self.sx,self.sy,self.sz
            Q += zero/((2*torch.pi)**(3/2) * sy**2*sz) * torch.exp(-1*(  (XX[:,:self.num_source_locs] - XX[:,self.num_source_locs:] - self.u_func(t0)*(tr-t0))**2  + 
                                                                            (YY[:,:self.num_source_locs] - YY[:,self.num_source_locs:] - self.v_func(t0)*(tr-t0))**2)/(2*sy**2))*(torch.exp(-1*((ZZ[:,:self.num_source_locs] - ZZ[:,self.num_source_locs:])**2)/(2*sz**2)) +
                                                                                                                                torch.exp( -1*((ZZ[:,:self.num_source_locs] + ZZ[:,self.num_source_locs:])**2)/(2*sz**2)))
  
            t += self.dt
        if bias_terms == 1:
                Q = torch.cat([Q,torch.ones(-1,len(Q))])
        elif bias_terms >1:
            for bias in range(bias_terms):
                temp = torch.zeros(obs.shape)
                temp[bias*num_obs:(bias+1)*num_obs] = 1
                Q = torch.cat([Q,torch.ones(len(Q),1)],dim=1)

        return Q.T@weights@Q, -2*obs.T@Q, Q
    

    def return_puff_matrix(self,obs,obs_t,weights=None):
        '''TO DO: Add Assertion that obs dtype is torch tensor float'''
        if weights is None:
            weights = weights = torch.diag(torch.ones(len(obs_t)))
        Q = torch.zeros(obs.shape[0]*obs.shape[1],self.num_source_locs)

        num_sensors = obs.shape[1]
        num_obs = obs.shape[0]
        obs = obs.T.reshape(-1,1)
        obs_t = obs_t.reshape(-1,1)
        tr = obs_t.repeat(num_sensors,1)
        # obs_locs = torch.cat([obs_locs])
        # X = torch.zeros(num_obs*num_sensors,self.num_source_locs)

        X= torch.cat([torch.cat([torch.tensor([self.sensor_locations[i][0],self.source_locations[j][0],self.sensor_locations[i][1],self.source_locations[j][1],
                                               self.sensor_locations[i][2],self.source_locations[j][2]]) for j in range(self.num_source_locs)])
                                               .repeat(num_obs,1) for i in range(num_sensors)])
        XX, YY, ZZ = X[:,[6*i for i in range(self.num_source_locs)]+[6*i+1 for i in range(self.num_source_locs)]], X[:,[6*i+2 for i in range(self.num_source_locs)]+[6*i+3 for i in range(self.num_source_locs)]], X[:,[6*i+4 for i in range(self.num_source_locs)]+[6*i+5 for i in range(self.num_source_locs)]]

        t = 0


        while t < obs_t[-1].item():
            t0 = torch.tensor([t]).repeat(num_obs*num_sensors,1)
            eps=1e-10
            
            zero = (tr - t0 > 0).float()

            if self.spread:
                sx,sy,sz =  self.sx*torch.sqrt(((tr-t0)*zero+eps)),  self.sy*torch.sqrt(((tr-t0)*zero+eps)), self.sz*torch.sqrt(((tr-t0)*zero+eps))
            else:
                sx,sy,sz = self.sx,self.sy,self.sz
            Q += self.dt*zero/((2*torch.pi)**(3/2) * sy**2*sz) * torch.exp(-1*(  (XX[:,:self.num_source_locs] - XX[:,self.num_source_locs:] - self.u_func(t0)*(tr-t0))**2  + 
                                                                            (YY[:,:self.num_source_locs] - YY[:,self.num_source_locs:] - self.v_func(t0)*(tr-t0))**2)/(2*sy**2))*(torch.exp(-1*((ZZ[:,:self.num_source_locs] - ZZ[:,self.num_source_locs:])**2)/(2*sz**2)) +
                                                                                                                                torch.exp( -1*((ZZ[:,:self.num_source_locs] + ZZ[:,self.num_source_locs:])**2)/(2*sz**2)))
            t += self.dt 
            
        return Q            


    def simulate_concentration(self,X,tt):
        '''TO DO: Add Assertion that obs dtype is torch tensor float'''
        tt = tt.reshape(-1,1)


        num_sensors = X.shape[0]
        num_obs = tt.shape[0]
        weights=None
        if weights is None:
            weights = torch.sparse.spdiags(torch.ones(num_obs*num_sensors),offsets=torch.tensor([0]),shape=(num_obs*num_sensors,num_obs*num_sensors))
        Q = torch.zeros(num_sensors*num_obs,self.num_source_locs)

        # X_repeated = X.repeat(1,tt.shape[0]).reshape(-1,3)
        tr = tt.repeat(X.shape[0],1)
        sensor_locations = X
        # obs_locs = torch.cat([obs_locs])
        # X = torch.zeros(num_obs*num_sensors,self.num_source_locs)
        
        #
        X= torch.cat([torch.cat([torch.tensor([sensor_locations[i][0],self.source_locations[j][0],sensor_locations[i][1],self.source_locations[j][1],
                                               sensor_locations[i][2],self.source_locations[j][2]]) for j in range(self.num_source_locs)])
                                               .repeat(num_obs,1) for i in range(num_sensors)])
        XX, YY, ZZ = X[:,[6*i for i in range(self.num_source_locs)]+[6*i+1 for i in range(self.num_source_locs)]], X[:,[6*i+2 for i in range(self.num_source_locs)]+[6*i+3 for i in range(self.num_source_locs)]], X[:,[6*i+4 for i in range(self.num_source_locs)]+[6*i+5 for i in range(self.num_source_locs)]]
        t = 0


        while t < tt[-1].item():
            t0 = torch.tensor([t]).repeat(num_obs*num_sensors,1)
            eps=1e-10
            zero = (tr - t0 > 0).float()

            if self.spread:
                sx,sy,sz =  self.sx*torch.sqrt(((tr-t0)*zero+eps)),  self.sy*torch.sqrt(((tr-t0)*zero+eps)), self.sz*torch.sqrt(((tr-t0)*zero+eps))
            else:
                sx,sy,sz = self.sx,self.sy,self.sz

            Q += self.dt*zero/((2*torch.pi)**(3/2) * sy**2*sz) * torch.exp(-1*(  (XX[:,:self.num_source_locs] - XX[:,self.num_source_locs:] - self.u_func(t0)*(tr-t0))**2  + 
                                                                            (YY[:,:self.num_source_locs] - YY[:,self.num_source_locs:] - self.v_func(t0)*(tr-t0))**2)/(2*sy**2))*(torch.exp(-1*((ZZ[:,:self.num_source_locs] - ZZ[:,self.num_source_locs:])**2)/(2*sz**2)) + torch.exp( -1*((ZZ[:,:self.num_source_locs] + ZZ[:,self.num_source_locs:])**2)/(2*sz**2)))
            t += self.dt
        num_bias_terms = len(self.q) - self.num_source_locs
        for bias in range(num_bias_terms):
            temp = torch.zeros(Q.shape[0],1)
            temp[bias*Q.shape[0]//num_bias_terms:(bias+1)*Q.shape[0]//num_bias_terms] = 1
            Q = torch.cat([Q,temp],dim=1)
        return Q @self.q          

    def return_puff_matrix_numerical(self,obs,obs_t,weights=None):
        '''TO DO: Add Assertion that obs dtype is torch tensor float'''
        if weights is None:
            weights = weights = torch.diag(torch.ones(len(obs_t)))
        Q = torch.zeros(obs.shape[0]*obs.shape[1],self.num_source_locs)

        num_sensors = obs.shape[1]
        num_obs = obs.shape[0]
        obs = obs.T.reshape(-1,1)
        obs_t = obs_t.reshape(-1,1)
        tr = obs_t.repeat(num_sensors,1)
        # obs_locs = torch.cat([obs_locs])
        # X = torch.zeros(num_obs*num_sensors,self.num_source_locs)

        X= torch.cat([torch.cat([torch.tensor([self.sensor_locations[i][0],self.source_locations[j][0],self.sensor_locations[i][1],self.source_locations[j][1],
                                               self.sensor_locations[i][2],self.source_locations[j][2]]) for j in range(self.num_source_locs)])
                                               .repeat(num_obs,1) for i in range(num_sensors)])
        XX, YY, ZZ = X[:,[6*i for i in range(self.num_source_locs)]+[6*i+1 for i in range(self.num_source_locs)]], X[:,[6*i+2 for i in range(self.num_source_locs)]+[6*i+3 for i in range(self.num_source_locs)]], X[:,[6*i+4 for i in range(self.num_source_locs)]+[6*i+5 for i in range(self.num_source_locs)]]

        t = 0


        while t < obs_t[-1].item():
            t0 = torch.tensor([t]).repeat(num_obs*num_sensors,1)
            eps=1e-10
            zero = (tr - t0 > 0).float()
            if self.spread:
                sx,sy,sz =  self.sx*torch.sqrt(((tr-t0)*zero+eps)),  self.sy*torch.sqrt(((tr-t0)*zero+eps)), self.sz*torch.sqrt(((tr-t0)*zero+eps))
            else:
                sx,sy,sz = self.sx,self.sy,self.sz
            Q += self.dt*zero/((2*torch.pi)**(3/2)*sy**2*sz) * torch.exp(-1*(  (XX[:,:self.num_source_locs] - XX[:,self.num_source_locs:] - (self.u_func_ivp(tr)-self.u_func_ivp(t0)))**2  + 
                                                                            (YY[:,:self.num_source_locs] - YY[:,self.num_source_locs:] - (self.v_func_ivp(tr)-self.v_func_ivp(t0)) )**2)/(2*sy**2))*(torch.exp(-1*((ZZ[:,:self.num_source_locs] - ZZ[:,self.num_source_locs:])**2)/(2*sz**2)) +
                                                                                                                                torch.exp( -1*((ZZ[:,:self.num_source_locs] + ZZ[:,self.num_source_locs:])**2)/(2*sz**2)))
            t += self.dt 
            
        return Q   

    def simulate_concentration_numerical(self,X,tt):
        '''TO DO: Add Assertion that obs dtype is torch tensor float'''
        tt = tt.reshape(-1,1)


        num_sensors = X.shape[0]
        num_obs = tt.shape[0]
        weights=None
        if weights is None:
            weights = torch.sparse.spdiags(torch.ones(num_obs*num_sensors),offsets=torch.tensor([0]),shape=(num_obs*num_sensors,num_obs*num_sensors))
        Q = torch.zeros(num_sensors*num_obs,self.num_source_locs)

        # X_repeated = X.repeat(1,tt.shape[0]).reshape(-1,3)
        tr = tt.repeat(X.shape[0],1)
        sensor_locations = X
        # obs_locs = torch.cat([obs_locs])
        # X = torch.zeros(num_obs*num_sensors,self.num_source_locs)
        
        #
        X= torch.cat([torch.cat([torch.tensor([sensor_locations[i][0],self.source_locations[j][0],sensor_locations[i][1],self.source_locations[j][1],
                                               sensor_locations[i][2],self.source_locations[j][2]]) for j in range(self.num_source_locs)])
                                               .repeat(num_obs,1) for i in range(num_sensors)])
        XX, YY, ZZ = X[:,[6*i for i in range(self.num_source_locs)]+[6*i+1 for i in range(self.num_source_locs)]], X[:,[6*i+2 for i in range(self.num_source_locs)]+[6*i+3 for i in range(self.num_source_locs)]], X[:,[6*i+4 for i in range(self.num_source_locs)]+[6*i+5 for i in range(self.num_source_locs)]]
        t = 0


        while t < tt[-1].item():
            t0 = torch.tensor([t]).repeat(num_obs*num_sensors,1)
            eps=1e-10
            zero = (tr - t0 > 0).float()

            if self.spread:
                sx,sy,sz =  self.sx*torch.sqrt(((tr-t0)*zero+eps)),  self.sy*torch.sqrt(((tr-t0)*zero+eps)), self.sz*torch.sqrt(((tr-t0)*zero+eps))
            else:
                sx,sy,sz = self.sx,self.sy,self.sz

            Q += self.dt*zero/((2*torch.pi)**(3/2)*sy**2*sz) * torch.exp(-1*(  (XX[:,:self.num_source_locs] - XX[:,self.num_source_locs:] - (self.u_func_ivp(tr)-self.u_func_ivp(t0)))**2  + 
                                                                            (YY[:,:self.num_source_locs] - YY[:,self.num_source_locs:] - (self.v_func_ivp(tr)-self.v_func_ivp(t0)) )**2)/(2*sy**2))*(torch.exp(-1*((ZZ[:,:self.num_source_locs] - ZZ[:,self.num_source_locs:])**2)/(2*sz**2)) +
                                                                            torch.exp( -1*((ZZ[:,:self.num_source_locs] + ZZ[:,self.num_source_locs:])**2)/(2*sz**2)))
            t += self.dt
        num_bias_terms = len(self.q) - self.num_source_locs
        for bias in range(num_bias_terms):
            temp = torch.zeros(Q.shape[0],1)
            temp[bias*Q.shape[0]//num_bias_terms:(bias+1)*Q.shape[0]//num_bias_terms] = 1
            Q = torch.cat([Q,temp],dim=1)
        return Q @self.q          