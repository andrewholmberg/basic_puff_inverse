import torch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline 
import copy
torch.set_default_dtype(torch.float32)
import time
import sys
from torchcubicspline import(natural_cubic_spline_coeffs, 
                             NaturalCubicSpline)
class puff_model():
    def __init__(self, source_locations,sensor_locations, sigma, u_func, v_func, qs,dt,spread=False,t_max = 45*60):
        self.corners = torch.tensor([(0,0),(200,0),(0,200),(200,200)])
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
        self.sensor_locations = torch.tensor(sensor_locations)
        self.dt = dt
        self.t_max = t_max
        self.spread=spread
        print(self.compute_num_puffs())
        t_span = (0,60*60)
        t_eval = torch.linspace(0,60*60,60*60)
        # cs_temp = CubicSpline(t_eval,solve_ivp(lambda t,y:u_func(t),t_span,[0],'RK45',t_eval)['y'][0])
        coefs = natural_cubic_spline_coeffs(t_eval,torch.tensor(solve_ivp(lambda t,y:u_func(t),t_span,[0],'RK45',t_eval)['y'][0]).reshape(-1,1))
        cs = NaturalCubicSpline(coefs)
        self.u_func_ivp = copy.deepcopy(lambda t: cs.evaluate(t))
        coefs1 = natural_cubic_spline_coeffs(t_eval, torch.tensor(solve_ivp(lambda t,y:v_func(t),t_span,[0],'RK45',t_eval)['y'][0]).reshape(-1,1))
        # cs_temp1 = CubicSpline(t_eval,solve_ivp(lambda t,y:v_func(t),t_span,[0],'RK45',t_eval)['y'][0])
        # self.v_func_ivp = copy.deepcopy(lambda t: torch.tensor(cs_temp1(t.detach().numpy())))
        cs1 = NaturalCubicSpline(coefs1)
        self.v_func_ivp = copy.deepcopy(lambda t: cs1.evaluate(t))


    def compute_num_puffs(self):
        bounds = self.corners.repeat(1,self.source_locations.shape[0])
        source_locations = self.source_locations[:,:2].reshape(1,-1).repeat(bounds.shape[0],1)
        min_amp  = torch.min(torch.sqrt(self.u_func(torch.linspace(0,self.t_max,self.t_max//self.dt))**2 + self.v_func(torch.linspace(0,self.t_max,self.t_max//self.dt))**2))
        assert bounds.shape == source_locations.shape
        eps=1e-8
        rshp = torch.abs(bounds-source_locations).reshape(-1,2)
        distances = (rshp[:,0]**2 + rshp[:,1]**2)**.5
        times = distances/(min_amp+eps)
        max_t = torch.max(times)
        self.num_puffs = int((max_t//self.dt * 2).item())
        return self.num_puffs
    
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


    def return_puff_matrix(self,X,obs_t,weights=None):
        '''TO DO: Add Assertion that obs dtype is torch tensor float'''

        num_obs = obs_t.shape[0]
        obs_t = obs_t.reshape(-1,1)
        # tr = obs_t.reshape(-1,1,1).repeat(1, self.num_source_locs, self.num_puffs)
        # tm = torch.linspace(0,(self.num_puffs-1)*self.dt, self.num_puffs).repeat(obs_t.shape[0],self.num_source_locs,1)*self.dt
        # t0 = tr - tm
        # XX = X.reshape(-1,3,1).repeat(1,self.num_source_locs,self.num_puffs).reshape(-1,self.num_source_locs,self.num_puffs*3)
        # XS = self.source_locations.reshape(-1,3,1).repeat(num_obs,1,self.num_puffs).reshape(-1,self.num_source_locs,self.num_puffs*3)

        tr = obs_t.reshape(-1, 1, 1).expand(-1, self.num_source_locs, self.num_puffs)

        tm_base = torch.arange(self.num_puffs, device=obs_t.device).float() * self.dt
        tm = torch.linspace(0,(self.num_puffs-1)*self.dt, self.num_puffs).expand(num_obs, self.num_source_locs, -1)
        print('size tm', tm.storage().size(),'size tr',tr.storage().size())
        t0 = tr - tm
        print('size t0', t0.storage().size())

        XX = X.reshape(-1, 1, 1, 3).expand(-1, self.num_source_locs, self.num_puffs, -1)
        XX = XX.reshape(-1, self.num_source_locs, self.num_puffs * 3)

        source_locs = self.source_locations.reshape(1, self.num_source_locs, 1, 3)
        XS = source_locs.expand(num_obs, -1, self.num_puffs, -1)
        XS = XS.reshape(-1, self.num_source_locs, self.num_puffs * 3)




        #XM = "X moved"
        XM = torch.cat([self.u_func_ivp(tr) - self.u_func_ivp(t0), self.v_func_ivp(tr) - self.v_func_ivp(t0)],dim=2)


        print('x shapes XX, XS, XM', XX.shape,XS.shape,XM.shape)

        eps=1e-10
        # zero = (tr - t0 > 0).float()

        zero = 1
        if self.spread:
            sx,sy,sz =  self.sx*torch.sqrt(((tr-t0)*zero+eps)),  self.sy*torch.sqrt(((tr-t0)*zero+eps)), self.sz*torch.sqrt(((tr-t0)*zero+eps))
        else:
            sx,sy,sz = self.sx,self.sy,self.sz
        t1 = time.time()
        Q = self.dt*zero/((2*torch.pi)**(3/2)*sy**2*sz) * torch.exp(-1*(  (XX[:,:,:self.num_puffs] - XS[:,:,:self.num_puffs] - XM[:,:,:self.num_puffs])**2  + 
                                                                        (XX[:,:,self.num_puffs:-self.num_puffs] - XS[:,:,self.num_puffs:-self.num_puffs] - XM[:,:,self.num_puffs:] )**2)/(2*sy**2))*(torch.exp(-1*((XX[:,:,-self.num_puffs:] - XS[:,:,-self.num_puffs:])**2)/(2*sz**2)) +
                                                                                                                       torch.exp( -1*((XX[:,:,-self.num_puffs:] + XS[:,:,-self.num_puffs:])**2)/(2*sz**2)))
        return torch.sum(Q,dim=2).float()

    def simulate_concentration(self,X,tt):
        '''TO DO: Add Assertion that obs dtype is torch tensor float'''
        Q = self.return_puff_matrix(X,tt)
        num_bias_terms = len(self.q) - self.num_source_locs
        for bias in range(num_bias_terms):
            temp = torch.zeros(Q.shape[0],1)
            temp[bias*Q.shape[0]//num_bias_terms:(bias+1)*Q.shape[0]//num_bias_terms] = 1
            Q = torch.cat([Q,temp],dim=1)
        return Q @self.q          