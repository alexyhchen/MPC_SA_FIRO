import numpy as np 
import cvxpy as cp
import matplotlib.pyplot as plt

class MPCProblem():

  def __init__(self, u_max, TOCS, Q, Qf, ramp_rate, full_output=False):
    
    self.u_max = u_max
    self.TOCS = TOCS
    self.Q = Q
    self.Qf = Qf
    self.ramp_rate = ramp_rate
    self.T = len(Q)
    self.S = np.zeros(self.T) # assumes init storage 0
    self.R = np.zeros(self.T)
    _, self.NE, self.NL = Qf.shape

    self.full_output = full_output
    if self.full_output:
      self.status = np.zeros(self.T, dtype='bool')
      self.obj = np.zeros(self.T)
      self.uf = np.zeros((self.T, self.NL))
      self.Sf_pre = np.zeros((self.T, self.NE, self.NL))
      self.Sf_post = np.zeros((self.T, self.NE, self.NL))

    self.create_cvx()

  def create_cvx(self):

    S0 = cp.Parameter(name='S0')
    u_max = cp.Parameter(name='u_max', value=self.u_max)
    Rt = cp.Parameter(name='Rt')
    TOCS = cp.Parameter((1,self.NL), name='TOCS', value=np.reshape(self.TOCS, (1,-1)))
    Qf = cp.Parameter((self.NE, self.NL), name='Qf')

    u = cp.Variable((1,self.NL), name='u')
    Sf = S0 + cp.cumsum(Qf - u, axis=1) # forecasted storage, NE x NL
    ESf = cp.sum(Sf[:,-1]) / self.NE

    # minimize expected storage above TOCS
    cost = cp.sum(cp.pos(Sf - TOCS)) / self.NE / self.NL
    # cost = cp.norm(cp.pos(Sf - TOCS), 'inf') # more risk averse version - max 
    cost += cp.neg(ESf - TOCS[0,-1]) # penalize ending below TOCS
    # two options - the terminal storage is either a penalty or a constraint
    # if constraint is infeasible, then use the penalty
    constraints = [u >= 0, u <= u_max] #, ESf == TOCS[0,-1]]

    # ramping rate assumes the same limit in either direction
    # must be assigned to both current and future release
    if self.ramp_rate is not None:
      constraints += [u[0,1:] <= (u[0,:-1] + self.ramp_rate),
                      u[0,1:] >= (u[0,:-1] - self.ramp_rate),
                      u[0,0] <= (Rt + self.ramp_rate),
                      u[0,0] >= (Rt - self.ramp_rate)]

    self.problem = cp.Problem(cp.Minimize(cost), constraints)

  def run(self, solver='ECOS', verbose=False):

    for t in range(self.T-self.NL):
      # Current step t ends 12:00 GMT today
      # Update storage S[t] to make release decision R[t+1]
      self.S[t] = self.S[t-1] + self.Q[t] - self.R[t]

      self.problem.param_dict['S0'].value = self.S[t]
      self.problem.param_dict['Qf'].value = self.Qf[t,:,:self.NL]
      self.problem.param_dict['Rt'].value = self.R[t]

      self.problem.solve(solver = solver)
      self.R[t+1] = self.problem.var_dict['u'].value[0, 0]

      if self.full_output:
        _S0 = self.problem.param_dict['S0'].value
        _Qf = self.problem.param_dict['Qf'].value

        self.status[t+1] = (self.problem.status == 'optimal')
        self.obj[t+1] = self.problem.objective.value
        self.uf[t+1,:] = self.problem.var_dict['u'].value[0,:]
        self.Sf_pre[t+1,:,:] = _S0 + np.cumsum(_Qf, axis=1)
        self.Sf_post[t+1,:,:] = _S0 + np.cumsum(_Qf - self.uf[t+1,:], axis=1)

      # if verbose and t % 100 == 0: #(self.T / 10) == 0:
      #   print('step ', t)