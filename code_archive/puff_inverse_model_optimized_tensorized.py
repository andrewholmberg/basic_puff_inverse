from puff_model_optimized_tensorized import puff_model
import torch
import numpy as np 
from cvxopt import matrix, solvers
solvers.options['show_progress'] = False
torch.set_default_dtype(torch.float32)
def solve_qp(P,q,G,h):
    q = q.view(-1,1)
    P = matrix(P.detach().numpy().astype(np.double))
    q = matrix(q.detach().numpy().astype(np.double))
    G = matrix(G.detach().numpy().astype(np.double))
    h = matrix(h.detach().numpy().astype(np.double))
    return torch.tensor(np.array(solvers.qp(P,q,G,h)['x'],dtype=np.float64)).float()

def lasso_qp(P,q,G,h,num_nz,bias_terms=0):
    def f(alpha):
        add = torch.zeros(P.shape[1])
        add[:(P.shape[1] - bias_terms)] = 1
        return solve_qp(P,q+add.reshape(q.shape)*alpha,G,h)[:P.shape[1]- bias_terms]
    return lasso(f,num_nz)

def lasso(f,num_nz):
    alpha = 1
    alpha_old = 0
    expand = True
    num = 0
    iter = 0
    while num != num_nz and abs(alpha_old - alpha) > 1e-12:
        x = f(alpha)
        num = torch.sum((torch.abs(x)>1e-3).int())
        if num > num_nz:
            temp = alpha
            alpha += 10**iter if expand else abs(alpha_old - alpha)/2
            alpha_old = temp
        elif num < num_nz:
            temp = alpha
            alpha -= abs(alpha_old - alpha)/2
            alpha_old = temp
        iter+=1
        if alpha < alpha_old:
            expand = False
    return alpha

class puff_inverse_model:
    
    def __init__(self, pm : puff_model ):
        self.pm = pm
    

    def solve_inverse_problem(self,X,obs_t,obs, num_nz, bias_terms = 0,weights=None):
        num_obs = len(obs_t)
        Q = self.pm.return_puff_matrix(X,obs_t,obs)
        if weights is None:
            weights = weights = torch.diag(torch.ones(torch.numel(obs))).float()
        if bias_terms == 1:
                Q = torch.cat([Q, torch.ones(len(Q), 1)], dim=1)
        elif bias_terms > 1:
            bias_terms = obs.shape[1]
            for bias in range(bias_terms):
                temp = torch.zeros(torch.numel(obs),1)
                temp[bias*num_obs:(bias+1)*num_obs] = 1.
                Q = torch.cat([Q,temp],dim=1)
        P = 2*Q.T@weights@Q
        q = -2*obs.T.reshape(-1,1).T@weights@Q
        G = -1*torch.eye(P.shape[1] , P.shape[1])
        h = torch.zeros(P.shape[1])
        # print(torch.linalg.matrix_rank(G))
        alpha = lasso_qp(P,q,G,h,num_nz,bias_terms=bias_terms)
        add = torch.zeros(P.shape[1])
        add[:(P.shape[1] - bias_terms)] = 1
        lasso_result = solve_qp(P,q+add.reshape(q.shape)*alpha,G,h)
        # print(lasso_result)
        filter = lasso_result.reshape(-1) < 1e-3
        # p = p_torch_n.reshape(-1,1)
        if bias_terms>0:
            filter[-bias_terms:] = False
        G = torch.cat([G,torch.eye(len(filter))[filter]])
        h = torch.cat([h.reshape(-1),torch.zeros(torch.sum(filter))])
        x = solve_qp(P,q,G,h)
        self.Q = Q
        self.pm.q = x
        return x
    
    def solve_inverse_problem_base(self,obs,obs_t,num_nz, weights=None,numerical=False):
        obs_rs = obs.T.reshape(-1,1)
        if numerical:
            Q = self.pm.return_puff_matrix_numerical
        else:
            Q = self.pm.return_puff_matrix(obs,obs_t)
        Q = torch.sum(Q,dim=1).reshape(-1,1)
        Q -= torch.min(Q)
        Q/= torch.max(Q)
        Q = 1 - Q
        # print(Q)
        W = torch.diag(Q.reshape(-1))
        one = torch.ones(torch.numel(obs_rs),1)

        # print(obs_rs.T.shape, W.shape,one.shape)
        bias = obs_rs.T@W@one/(one.T@W@one)
        obs -= bias
        result = self.solve_inverse_problem(obs,obs_t,num_nz,bias_terms=0,weights=weights,numerical=True)
        self.pm.q = torch.cat([result,bias])
        return torch.cat([result,bias])


        

    

    




