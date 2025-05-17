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
    def return_qp_matrices_new(self,obs,obs_t,spread=False):
        '''TO DO: Add Assertion that obs dtype is torch tensor float'''
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
        return Q.T@Q, -2*obs.T@Q, Q
                    


pm = puff_model([[0,0,0],[1,1,1]],[[2,2,2],[3,3,3]],[1,1,1],lambda t : 0, lambda t : 2,[1,1],.25)
# pm.return_qp_matrices(torch.tensor([[1,2,3],[4,5,6]]))

obs_t = torch.tensor([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16])
obs = torch.tensor([[1,0],[1,1],[1,0],[0.,1],[1,0],[1,1],[1,0],[0,1],[1,0],[1,1],[1,0],[0,1],[1,0],[1,1],[1,0],[0,1]])
res = pm.return_qp_matrices_new(obs,obs_t)
# print(res)
