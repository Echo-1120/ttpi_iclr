import torch
torch.set_default_dtype(torch.float64)

class PointMass:
    def __init__(self, dt= 0.01,
                    x_obst=[], r_obst=[], order=1, dim=2,
                    w_obst=0., w_action=0.1, w_goal=0.9, w_scale=1.0, device='cpu'):
        ''' 
            dim: dimension of the space 
            state : (position) for velocity control  or (position, velocity) for acceleration control
            actions : position or velocity
        '''
        self.device = device
        self.dim=dim
        self.dim_state = dim
        self.dim_action = dim

        self.position_min = -100#position_min.reshape(-1) 
        self.position_max = 100#position_max.reshape(-1) 
        
        self.velocity_max = 100#velocity_max.reshape(-1)
        self.velocity_min = -100#velocity_min.reshape(-1)

        self.action_max = 100#action_max
        self.action_min = -100#action_min

        self.dt = dt
        self.order=order

        if order==1: # velocity control
            self.forward_simulate = self.forward_simulate_1
        else: # acceleration control
            self.forward_simulate = self.forward_simulate_2

        self.x_obst = x_obst # positions of the spherical obstacles
        self.r_obst = r_obst # radii of the obstacles

        self.margin = 0.01

        self.b_action = 1.#2*(torch.linalg.norm(action_max)) #0.25*(torch.linalg.norm(action_max)**2)
        self.b_obst =1.#0.5
        self.b_goal = 1.#2*(torch.linalg.norm(position_max)) #0.25*(torch.linalg.norm(position_max)**2)
        self.b_velocity = 1
        self.w_obst = w_obst
        self.w_goal = w_goal # importance on state 
        self.w_action = w_action # importance of low control inputs
        self.w_scale = w_scale

        self.alpha_goal = self.w_goal/self.b_goal
        self.alpha_action = self.w_action/self.b_action
        self.alpha_obst = self.w_obst/self.b_obst

        self.target_position = torch.tensor([[0]*self.dim]).to(self.device)
        self.target_velocity = torch.tensor([[0]*self.dim]).to(self.device)

    def forward_simulate_2(self, state, action):
        '''
        Given (state,action)  find the next state 
        state: (position,velocity)
        action: acceleration
        '''
        position = state[:,:self.dim]
        velocity = state[:,self.dim:]
        action = torch.clip(action, self.action_min, self.action_max)
        d_position = velocity*self.dt
        d_velocity = action*self.dt
        
       
        position_new = torch.clip(position+d_position, self.position_min, self.position_max)
        velocity_new = torch.clip(velocity+d_velocity, self.velocity_min, self.velocity_max)
        # collision_free = self.is_collision_free(position).view(-1,1)

        state_new = torch.cat((position_new,velocity_new),dim=-1)    
        return state_new

    def forward_simulate_1(self, state, action):
        '''
        Given (state,action) find the next state 
        state: position
        action: velocity
        '''
        action = torch.clip(action, self.action_min, self.action_max)
        d_position = action*self.dt
        state_new = torch.clip(state+d_position, self.position_min, self.position_max)
        return state_new

    def get_B(self,state):
        ''' Assuming control affinr form: x_dot = A(x) + B(x)u. Return B(x) given state x'''
        if self.order == 1:
            B = torch.eye(self.dim).to(self.device)
        else:
            B = torch.zeros(2*self.dim,self.dim).to(self.device)
            B[self.dim:,:]=torch.eye(self.dim).to(self.device)
        B = B.view(1,-1,-1).expand(state.shape[0],-1,-1)
        return B
    
    def get_R(self,state):
        R = torch.eye(self.dim_action).to(self.device).view(1,-1,-1).expand(state.shape[0])*self.alpha_action
        return R

    def dist_position(self, position):
        # cost w.r.t. error to goal: (ex,ey) 
        ''' input shape: (batch_size, dim_state) '''
        d_x = torch.linalg.norm(position-self.target_position, dim=1)# batch_size x 1
        return d_x

    def dist_velocity(self, velocity):
        d_v = torch.linalg.norm(velocity-self.target_velocity, dim=1)# batch_size x 1
        return d_v

    def dist_collision(self,position):
        ''' 
            signed distance function (-1,inf), dist<0 inside the obstacle
            input shape: (batch_size,dim), find the distance into obstacle (-1 to inf), -1 completely inside, 0: at the boundary
        '''
        batch_size = position.shape[0]
        dist_obst = torch.zeros(batch_size).to(self.device)
        for i in range(len(self.x_obst)):
             dist_i = -1.0 + (torch.linalg.norm(position-self.x_obst[i], dim=1)/(self.r_obst[i]+self.margin)).view(-1)
             dist_obst = dist_obst + dist_i*(dist_i<0)
        return dist_obst.view(-1)

    def is_collision_free(self,position):
        ''' 
            input shape: (batch_size,dim), check if the state is in obstacle
        '''
        batch_size = position.shape[0]
        hit_obst = torch.zeros(batch_size).to(self.device)
        for i in range(len(self.x_obst)):
             dist_i = (torch.linalg.norm(position-self.x_obst[i], dim=1)/(self.r_obst[i]+self.margin)).view(-1)
             hit_obst = hit_obst + 1.0*(dist_i<1.)
        return (1.-hit_obst).view(-1)
    
    def dist_action(self,actions):
        '''
            Define the control cost
            input_shape = (batch_size, dim_action)
        '''
        d_action = torch.linalg.norm(actions,dim=-1)
        return d_action        


    def reward_state_action(self, state, action):
        ''' 
            Compute the stage reward given the action (ctrl input) and the state
        '''
        next_state = self.forward_simulate(state,action)
        if self.order == 1:

            position_1 = state
            position_2 = next_state
            d_goal_1 = (self.dist_position(state)).view(-1)
            d_goal_2 = (self.dist_position(next_state)).view(-1)
            d_goal = d_goal_1 #0.5*(d_goal_1+d_goal_2)

        else:
            d_goal_1 = torch.linalg.norm(state,dim=-1) #d_pos + d_goal
            d_goal_2 = torch.linalg.norm(next_state,dim=-1)
            position_1 = state[:, :self.dim]
            position_2 = next_state[:, :self.dim]
            d_goal = d_goal_1 #0.5*(d_goal_1+d_goal_2)


        d_obst_1 = 1.0*(self.dist_collision(position_1)).view(-1) # range:(-1,inf), where (-1,0): within the obst, (0,inf): away from obstacle
        d_obst_2 = 1.0*(self.dist_collision(position_2)).view(-1) # range:(-1,inf), where (-1,0): within the obst, (0,inf): away from obstacle
        d_obst = (d_obst_1/0.1).abs() #0.5*(d_obst_1+d_obst_2)
        r_obst = -1*d_obst#-1 + torch.sigmoid(d_obst/0.01) # (-1,0)
#         d_obst = -1 + torch.exp(-p_obst**2)
        
        
        d_action = torch.linalg.norm(action,dim=-1)
        r_action = -1*d_action#-1 + torch.exp(-d_action/self.b_action) #-1+torch.exp(-(d_action/self.b_action)) # -1 + torch.exp(-d_action**2) #torch.exp(-d_action**2) # (-1,0) #-torch.log(1e-2+d_action)#

        r_goal = -1*(d_goal/0.1).abs()

        r_total = (r_goal*self.w_goal+ r_obst*self.w_obst + r_action*self.w_action)
        r_all = torch.cat(( r_goal.view(-1,1), r_action.view(-1,1), r_obst.view(-1,1)),dim=1)
        return r_total, r_all
 
 
################################################################################################################
################################################################################################################

################################################################################################################
################################################################################################################



class SinglePendulum:
    '''
        Dynamics of a single pendulum
        state: (theta, dtheta) # (joinr_angle,joint_vel), joint_angle in (-pi, pi), it is 0 at the stable equilibrium
        action: (u) # joint torquw
    '''
    def __init__(self, state_min, state_max, action_max, action_min, dt=0.01, coef_viscous=0., w_scale=1.0, w_goal=0.9,w_action=0.1, length=1.0, mass=1.0, device='cpu'):
        self.l = length # length of the car
        self.mgl = mass*9.81*self.l # mg/l
        self.g = 9.81
        self.coef_viscous = coef_viscous

        self.state_min = state_min
        self.state_max = state_max # max of ()
        self.action_max = action_max
        self.action_min = action_min
        self.dt=dt
        w_total = w_goal+w_action
        self.w_goal = w_goal/w_total
        self.w_action = w_action/w_total

        self.w_scale = w_scale


    def forward_simulate(self, state, action):
        # action = torch.clip(action,self.action_min,self.action_min)
        theta = state[:,0]
        ctheta = torch.cos(theta)
        stheta = torch.sin(theta)
        dtheta = state[:,-1]
        tau_gravity  = self.mgl*stheta
        tau_viscousity = self.coef_viscous*dtheta
        ddtheta = action[:,0] - tau_gravity - tau_viscousity

        theta = theta + dtheta*self.dt
        dtheta = dtheta + ddtheta*self.dt

        theta =  (torch.abs(theta)<=torch.pi)*theta+(theta>torch.pi)*(theta-2*torch.pi)+(theta< -1*torch.pi)*(2*torch.pi+theta) # theta is in range (-pi,pi)
        state_new = torch.concat((theta.view(-1,1),dtheta.view(-1,1)),dim=-1)
        # state_new = torch.clip(state_new, self.state_min, self.state_max)
        return state_new


    def forward_simulate_mm(self, state, action, l, m):
        # action = torch.clip(action,self.action_min,self.action_min)
        theta = state[:,0]
        ctheta = torch.cos(theta)
        stheta = torch.sin(theta)
        dtheta = state[:,-1]
        tau_gravity  = m*self.g*l*stheta
        tau_viscousity = self.coef_viscous*dtheta
        ddtheta = action[:,0] - tau_gravity - tau_viscousity

        theta = theta + dtheta*self.dt
        dtheta = dtheta + ddtheta*self.dt

        theta =  (torch.abs(theta)<=torch.pi)*theta+(theta>torch.pi)*(theta-2*torch.pi)+(theta< -1*torch.pi)*(2*torch.pi+theta) # theta is in range (-pi,pi)
        state_new = torch.concat((theta.view(-1,1),dtheta.view(-1,1)),dim=-1)
        # state_new = torch.clip(state_new, self.state_min, self.state_max)
        return state_new


    def reward_state_action(self, state, action):
        next_state = self.forward_simulate(state,action)
        d_theta_1 = (state[:,0].abs()-torch.pi).abs()
        d_theta_2 = (next_state[:,0].abs()-torch.pi).abs()
        d_theta = 0.5*(d_theta_1+d_theta_2)
        d_action = (action[:,0]).abs()/(self.action_max[0])
        r_goal = -(d_theta/0.1)**2
        r_action = -(d_action)**2
        return self.w_scale*(self.w_goal*r_goal + self.w_action*r_action)

################################################################################################################
################################################################################################################
################################################################################################################
################################################################################################################


class HardMove:
    '''
        Dynamics of a Hard-Move Agent

        state: (x, y, x_dot, y_dot) 
        action: (switch_i, v_i) i=1,...,n
        v_i is velocity of the system along the thrust given by i-th actuator
        reference: 
    '''
    def __init__(self, dt=0.001, pos_max=1.0, v_max=1.0, w_goal=1.0, w_action=1.0, n=4, device='cpu'):
        self.damping= 0.01
        self.n = n 
        self.w_goal = w_goal
        self.w_action = w_action

        self.state_max = torch.tensor([pos_max]*2+[v_max]*2).to(device)
        self.state_min = -1*self.state_max
        self.device = device
        self.dt = dt

    def forward_simulate(self, state, action):
        state_new = state.clone()
        x = state[:,0]
        y = state[:,1]
        xy_dot = state[:, 2:4]* (1-self.damping)
        theta_actuator = (((torch.arange(self.n)*torch.pi*2)/self.n)).view(1,-1).to(self.device)
        idx_switch = torch.arange(self.n).to(self.device)
        switch_on = action[:,2*idx_switch+1].view(-1,self.n)
        v = action[:,idx_switch*2]*switch_on
        x_dot= xy_dot[:, 0] + (v*torch.cos(theta_actuator)).sum(dim=-1)*self.dt
        y_dot = xy_dot[:, 1] + (v*torch.sin(theta_actuator)).sum(dim=-1)*self.dt
        x = x + x_dot *self.dt
        y = y + y_dot * self.dt
        state_new[:, :2] = torch.concat((x.view(-1,1),y.view(-1,1)),dim=-1)
        state_new[:, 2:] = torch.concat((x_dot.view(-1,1),y_dot.view(-1,1)),dim=-1)
        return state_new #torch.clip(state_new, self.state_min, self.state_max)


    def reward_action(self,action):
        idx_switch = torch.arange(self.n).to(self.device)
        switch_on = action[:,2*idx_switch+1].view(-1,self.n)
        v = action[:,idx_switch*2]
        
        # cost_switch = switch_on.sum(dim=-1).view(-1)
        cost = (torch.abs(v)*switch_on).sum(dim=-1).view(-1)
        return -1*self.w_action*cost      


    def reward_state_action(self,state,action):
        d_goal = torch.linalg.norm(state[:,:2],dim=-1)
        r_goal = -1*(d_goal/0.01)**2
        r_action = self.reward_action(action)
        return self.w_goal*r_goal  + r_action

