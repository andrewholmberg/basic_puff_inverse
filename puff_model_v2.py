import torch

class puff:
    def __init__(self,location,sigma,t0,u,v,Q):
        assert len(location) == 3
        self.x0 = location[0]
        self.y0 = location[1]
        self.z0 = location[2]
        self.t0 = t0
        self.location = location
        self.sx = sigma[0]
        self.sy = sigma[1]
        self.sz = sigma[2]
        self.sigma = sigma
        self.Q = Q

    def predict(self,x,y,z,t):
        if t < self.t0:
            return None
        return self.Q/((2*torch.pi)**(3/2) * self.sy**2*self.sz)*torch.exp(-1*(  (x - self.x0 - self.u*t)**2  + (y - self.y0 - self.v*t)**2)/(2*self.sy**2))*(torch.exp( -1*((z - self.z0)**2)/(2*self.sz**2)) + torch.exp( -1*((z + self.z0)**2)/(2*self.sz**2)))

class puff_model():
    def __init__(self, source_locations,sensor_locations, sigma, u_func, v_func, Qs,dt):
        self.sx = sigma[0]
        self.sy = sigma[1]
        self.sz = sigma[2]
        self.num_source_locs = len(source_locations)
        self.puffs = list()
        self.source_locations = source_locations
        self.sigma = sigma
        self.u_func = u_func
        self.v_func = v_func
        self.Q = Qs
        self.constant = 1/((2*torch.pi)**(3/2) * self.sy**2*self.sz)
        self.sensor_locations = sensor_locations
        self.dt = dt
        # for i in range(len(locations)):
        #     self.puffs.append(puff())

    '''obs - nxm matrix - n = number of time samples, m = number of sensors'''
    def compute_exp_term(self,source_idx,sensor_idx,t,t0):
        assert t >= t0
        if t == t0: return 0
        eps = 1e-6
        x,y,z = self.sensor_locations[sensor_idx][0],self.sensor_locations[sensor_idx][1], self.sensor_locations[sensor_idx][2]
        x0,y0,z0 = self.source_locations[source_idx][0],self.source_locations[source_idx][1],self.source_locations[source_idx][2]
        sx,sy,sz = torch.sqrt((t-t0)+eps)*self.sx,torch.sqrt((t-t0)+eps)*self.sy,torch.sqrt((t-t0)+eps)*self.sz+1.5
        # sx,sy,sz = self.sx,self.sy,self.sz
        assert sx > 0 and sy > 0 and sz > 0

        to_ret = 1/((2*torch.pi)**(3/2) * self.sy**2*self.sz)*torch.exp(torch.tensor(-1*(  (x - x0 - self.u_func(t0)*(t-t0))**2  + (y - y0 - self.v_func(t0)*(t-t0))**2)/(2*sy**2)))*(torch.exp(torch.tensor( -1*((z - z0)**2)/(2*sz**2))) + torch.exp(torch.tensor( -1*((z + z0)**2)/(2*self.sz**2))))
        assert to_ret >= 0
        return to_ret
    def return_qp_matrices(self,obs,obs_t):
        '''TO DO: Add Assertion that obs dtype is torch tensor float'''
        
        num_sensors = obs.shape[1]
        num_obs = obs.shape[0]
        obs = obs.T.reshape(-1,1)
        Q = torch.zeros(obs.shape[0]*obs.shape[1],self.num_source_locs)
        for s in range(num_sensors):
            t = 0
            for o in range(num_obs): 
                t = 0
                temp = torch.zeros(self.num_source_locs)
                while t < obs_t[o]:
                    temp += torch.tensor([self.constant*self.compute_exp_term(i,s,obs_t[o],t) for i in range(self.num_source_locs)])*self.dt
                    t += self.dt
                Q[s*num_obs + o,:] = temp
        
        Q = torch.round(Q,decimals=5)
        return Q.T@Q, +2*obs.T@Q, Q
                    
                



'''
pm = puff_model([[0,0,0],[1,1,1]],[[2,2,2],[3,3,3]],[1,1,1],lambda t : 2, lambda t : 2,[1,1],.25)
# pm.return_qp_matrices(torch.tensor([[1,2,3],[4,5,6]]))

obs_t = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
obs = torch.tensor([[1,0],[1,1],[1,0],[0.,1],[1,0],[1,1],[1,0],[0,1],[1,0],[1,1],[1,0],[0,1],[1,0],[1,1],[1,0],[0,1]])
res = pm.return_qp_matrices(obs,obs_t)
print(res)
'''