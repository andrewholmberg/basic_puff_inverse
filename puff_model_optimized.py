import torch
import numpy as np
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline 
import copy


class puff_model():
    def __init__(self, source_locations,sensor_locations, sigma, u_func, v_func, qs,dt,spread=False, t0 = 0, t_max = 45*60):
        self.corners = torch.tensor([(0,0),(150,0),(0,150),(150,150)])
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
        self.t0 = t0
        self.t_max = t_max
        self.spread=spread
        self.compute_num_puffs()
        t_span = (t0,t_max)
        t_eval = np.linspace(t0,t_max,int(t_max))
        cs_temp = CubicSpline(t_eval,solve_ivp(lambda t,y:u_func(t),t_span,[0],'RK45',t_eval)['y'][0])
        self.u_func_ivp = copy.deepcopy(lambda t: torch.tensor(cs_temp(t.detach().numpy())))
        cs_temp1 = CubicSpline(t_eval,solve_ivp(lambda t,y:v_func(t),t_span,[0],'RK45',t_eval)['y'][0])
        self.v_func_ivp = copy.deepcopy(lambda t: torch.tensor(cs_temp1(t.detach().numpy())))


    def compute_num_puffs(self):
        bounds = self.corners.repeat(1,self.source_locations.shape[0])
        source_locations = self.source_locations[:,:2].reshape(1,-1).repeat(bounds.shape[0],1)
        min_amp  = torch.median(torch.sqrt(self.u_func(torch.linspace(self.t0,self.t_max,int((self.t_max - self.t0)//self.dt)))**2 + self.v_func(torch.linspace(self.t0,self.t_max,int((self.t_max - self.t0)//self.dt)))**2))

        min_amp = .01 if min_amp < .01 else min_amp
        assert bounds.shape == source_locations.shape
        eps=1e-8
        rshp = torch.abs(bounds-source_locations).reshape(-1,2)
        distances = (rshp[:,0]**2 + rshp[:,1]**2)**.5
        times = distances/(min_amp+eps)
        max_t = torch.max(times)
        self.num_puffs = int((max_t//self.dt * 2).item())
        return self.num_puffs
    


    def return_puff_matrix(self,X,obs_t):
        '''TO DO: Add Assertion that obs dtype is torch tensor float'''
        assert X.shape[0] == obs_t.shape[0]
        num_obs = obs_t.shape[0]
        obs_t = obs_t.reshape(-1,1)
        Q = torch.zeros(num_obs,self.num_source_locs)
        XX = X.reshape(-1,3,1).repeat(1,self.num_source_locs,1).reshape(-1,self.num_source_locs,3)
        XS = self.source_locations.reshape(-1,3,1).repeat(num_obs,1,1).reshape(-1,self.num_source_locs,3)
        tr = obs_t.reshape(-1,1,1).repeat(1, self.num_source_locs,1)
        tr = obs_t.reshape(-1,1,1).repeat(1, self.num_source_locs,1)
        XM1 = self.u_func_ivp(tr)
        XM2 = self.v_func_ivp(tr)
        for step in range(self.num_puffs):
            # tm = torch.linspace(0,(self.num_puffs-1)*self.dt).repeat(obs_t.shape[0],self.num_source_locs,1)*self.dt
            tm = step*self.dt
            t0 = tr - tm
            zero = (t0 >= self.t0).int()
            XM = torch.cat([XM1 - self.u_func_ivp(t0), XM2 - self.v_func_ivp(t0)],dim=2)
            eps=1e-10
            # zero = (tr - t0 > 0).float()

            # zero = 1
            if self.spread:
                sx,sy,sz =  self.sx*torch.sqrt(((tr-t0)*zero+eps))[:,:,0],  self.sy*torch.sqrt(((tr-t0)*zero+eps))[:,:,0], self.sz*torch.sqrt(((tr-t0)*zero+eps))[:,:,0]
            else:
                sx,sy,sz = self.sx,self.sy,self.sz
            Q += self.dt*zero[:,:,0]/((2*torch.pi)**(3/2)*sy**2*sz) * torch.exp(-1*(  (XX[:,:,0] - XS[:,:,0] - XM[:,:,0])**2  + 
                                                                        (XX[:,:,1] - XS[:,:,1] - XM[:,:,1] )**2)/(2*sy**2))*(torch.exp(-1*((XX[:,:,2] - XS[:,:,2])**2)/(2*sz**2)) +
                                                                                                                       torch.exp( -1*((XX[:,:,2] + XS[:,:,2])**2)/(2*sz**2)))
        return Q.float()




 
    def return_puff_matrix_gpt(self, X, obs_t):
        '''Memory-efficient version of return_puff_matrix using a for-loop over puffs'''
        assert isinstance(obs_t, torch.Tensor) and obs_t.dtype == torch.float32

        num_obs = obs_t.shape[0]
        Q_total = torch.zeros(num_obs, self.num_source_locs, dtype=torch.float32, device=X.device)

        X = X.view(num_obs, 1, 3)  # [num_obs, 1, 3]
        source_locs = self.source_locations.view(1, self.num_source_locs, 3)  # [1, num_sources, 3]

        for i in range(self.num_puffs):
            t0 = i * self.dt
            tr = obs_t.view(-1, 1)  # [num_obs, 1]
            t_delay = tr - t0  # [num_obs, 1]

            valid_mask = t_delay >= 0
            if not valid_mask.any():
                continue

            u_disp = self.u_func_ivp(t0) * t_delay  # shape: [num_obs, 1]
            v_disp = self.v_func_ivp(t0) * t_delay
            displacement = torch.stack((u_disp, v_disp, torch.zeros_like(u_disp)), dim=-1)  # [num_obs, 1, 3]

            diff = X - source_locs - displacement  # [num_obs, num_sources, 3]

            if self.spread:
                sx = self.sx * torch.sqrt(t_delay + 1e-6)
                sy = self.sy * torch.sqrt(t_delay + 1e-6)
                sz = self.sz * torch.sqrt(t_delay + 1e-6)
            else:
                sx = self.sx
                sy = self.sy
                sz = self.sz

            norm_xy = ((diff[..., 0]**2 + diff[..., 1]**2) / (2 * sy**2)).clamp(max=100)
            norm_z1 = ((diff[..., 2]**2) / (2 * sz**2)).clamp(max=100)
            norm_z2 = ((-diff[..., 2]**2) / (2 * sz**2)).clamp(max=100)

            scale = self.dt / ((2 * torch.pi)**1.5 * sy**2 * sz)
            Q = scale * torch.exp(-norm_xy) * (torch.exp(-norm_z1) + torch.exp(-norm_z2))
            Q = Q * valid_mask  # Zero out invalid entries

            Q_total += Q

        return Q_total.float()

    def simulate_concentration(self,X,tt):
        '''TO DO: Add Assertion that obs dtype is torch tensor float'''
        Q = self.return_puff_matrix(X,tt)
        num_bias_terms = len(self.q) - self.num_source_locs
        for bias in range(num_bias_terms):
            temp = torch.zeros(Q.shape[0],1)
            temp[bias*Q.shape[0]//num_bias_terms:(bias+1)*Q.shape[0]//num_bias_terms] = 1
            Q = torch.cat([Q,temp],dim=1)
        return Q @self.q          