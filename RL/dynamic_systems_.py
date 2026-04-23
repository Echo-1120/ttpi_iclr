import torch
torch.set_default_dtype(torch.float64)

class PointMass:
    def __init__(self,dt= 0.01,action_max=1.0, max_vel=0.25, max_pos=1.0, 
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

        self.position_max = torch.tensor([max_pos]*dim).to(device)
        self.position_min = -self.position_max
        self.velocity_max = torch.tensor([max_vel]*dim).to(device)
        self.velocity_min = -self.velocity_max

        self.action_max = torch.tensor([action_max]*dim).to(device)
        self.action_min = -self.action_max 

        self.dt = dt
        self.order=order


        if order==1:
            self.dim_state = dim
            self.state_max = 1*self.position_max
            self.state_min = 1*self.position_min

        elif order==2:
            self.state_max = torch.concat((self.position_max,self.velocity_max),dim=-1)
            self.state_min = torch.concat((self.position_min,self.velocity_min),dim=-1)
            self.dim_state = 2*dim
        
        else:
            self.state_max = torch.concat((self.position_max,self.position_max),dim=-1)
            self.state_min = torch.concat((self.position_min,self.position_min),dim=-1)
            self.dim_state = 2*dim

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
            # position = state
            # d_goal = 1*torch.linalg.norm(state,dim=-1)
            position_1 = state
            position_2 = next_state
            d_goal_1 = (self.dist_position(state)).view(-1)
            d_goal_2 = (self.dist_position(next_state)).view(-1)
            d_goal = d_goal_1 #0.5*(d_goal_1+d_goal_2)

        else:
            # position = state[:,:self.dim]
            # velocity = state[:,self.dim:]
            # d_pos = torch.linalg.norm(position,dim=-1)
            # d_vel = torch.linalg.norm(velocity,dim=-1)
            d_goal_1 = torch.linalg.norm(state,dim=-1) #d_pos + d_goal
            d_goal_2 = torch.linalg.norm(next_state,dim=-1)
            position_1 = state[:, :self.dim]
            position_2 = next_state[:, :self.dim]
#             d_goal_1 = (self.dist_position(position =position_1 )).view(-1)
#             d_goal_2 = (self.dist_position(position = position_2)).view(-1)
            d_goal = d_goal_1 #0.5*(d_goal_1+d_goal_2)


        d_obst_1 = 1.0*(self.dist_collision(position_1)).view(-1) # range:(-1,inf), where (-1,0): within the obst, (0,inf): away from obstacle
        d_obst_2 = 1.0*(self.dist_collision(position_2)).view(-1) # range:(-1,inf), where (-1,0): within the obst, (0,inf): away from obstacle
        d_obst = (d_obst_1/0.01)**2 #0.5*(d_obst_1+d_obst_2)
        r_obst = -1*d_obst#-1 + torch.sigmoid(d_obst/0.01) # (-1,0)
#         d_obst = -1 + torch.exp(-p_obst**2)
        
        
        d_action = torch.linalg.norm(action,dim=-1)
        r_action = -1*d_action**2#-1 + torch.exp(-d_action/self.b_action) #-1+torch.exp(-(d_action/self.b_action)) # -1 + torch.exp(-d_action**2) #torch.exp(-d_action**2) # (-1,0) #-torch.log(1e-2+d_action)#

        r_goal = -1*(d_goal/0.05)**2

        r_total = (r_goal*self.w_goal+ r_obst*self.w_obst + r_action*self.w_action)
        r_all = torch.cat(( r_goal.view(-1,1), r_action.view(-1,1), r_obst.view(-1,1)),dim=1)
        return r_total
 
 
################################################################################################################
################################################################################################################

################################################################################################################
################################################################################################################

class PointMassDS:
    def __init__(self, position_min, position_max, velocity_min, velocity_max, 
                    action_min, action_max, dt= 0.01,
                    order=1, dim=2,
                    w_obst=0., w_action=0.1, w_goal=0.9, w_scale=1.0, device='cpu'):
        ''' 
            dim: dimension of the space 
            state : (position) for velocity control  or (position, velocity) for acceleration control
            actions : position or velocity
        '''
        self.device = device
        self.dim=dim
        self.dim_action = len(action_max)

        self.position_min = position_min.reshape(-1) 
        self.position_max = position_max.reshape(-1) 
        
        self.velocity_max = velocity_max.reshape(-1)
        self.velocity_min = velocity_min.reshape(-1)

        self.action_max = action_max
        self.action_min = action_min

        self.dt = dt
        self.order=order

        if order==1: # velocity control
            self.forward_simulate = self.forward_simulate_1
        else: # acceleration control
            self.forward_simulate = self.forward_simulate_2


        self.margin = 0.01 # for obstacle avoidance

        self.b_action = 0.1#2*(torch.linalg.norm(action_max)) #0.25*(torch.linalg.norm(action_max)**2)
        self.b_obst = 0.1#0.5
        self.b_goal = 0.1#2*(torch.linalg.norm(position_max)) #0.25*(torch.linalg.norm(position_max)**2)
        self.b_velocity = 1*(torch.linalg.norm(velocity_max))
        w_total = w_obst + w_goal +w_action
        self.w_obst = w_obst/w_total
        self.w_goal = w_goal/w_total # importance on state 
        self.w_action = w_action/w_total # importance of low control inputs
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
        velocity = state[:,self.dim:2*self.dim]
        x_obst = state[:,2*self.dim:3*self.dim]
        r_obst = state[:,3*self.dim].view(state.shape[0],1)
        action = torch.clip(action, self.action_min, self.action_max)
        d_position = velocity*self.dt
        d_velocity = action*self.dt
        
        position_new = torch.clip(position+d_position, self.position_min, self.position_max)
        velocity_new = torch.clip(velocity+d_velocity, self.velocity_min, self.velocity_max)
        # collision_free = self.is_collision_free(position).view(-1,1)

        state_new = torch.cat((position_new,velocity_new, x_obst, r_obst),dim=-1)    
        return state_new

    def forward_simulate_1(self, state, action):
        '''
        Given (state,action) find the next state 
        state: position
        action: velocity
        '''
        position = state[:,:self.dim]
        x_obst = state[:,self.dim:2*self.dim]
        r_obst = state[:,2*self.dim].view(state.shape[0],1)
        action = torch.clip(action, self.action_min, self.action_max)
        d_position = action*self.dt
        position_new = torch.clip(position+d_position, self.position_min, self.position_max)
        state_new = torch.concat((position_new,x_obst,r_obst),dim=-1)
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

    def dist_collision(self, position, x_obst, radius_obst):
        ''' 
            signed distance function (-1,inf), dist<0 inside the obstacle
            input shape: (batch_size,dim), find the distance into obstacle (-1 to inf), -1 completely inside, 0: at the boundary
        '''
        batch_size = position.shape[0]
        dist_obst = torch.zeros(batch_size).to(self.device)
        dist_obst = -1.0 + (torch.linalg.norm(position-x_obst, dim=1)/(radius_obst.view(-1)+self.margin)).view(-1)
        return dist_obst.view(-1)

    def is_collision_free(self,position, x_obst, r_obst):
        ''' 
            input shape: (batch_size,dim), check if the state is in obstacle
        '''
        batch_size = position.shape[0]
        hit_obst = torch.zeros(batch_size).to(self.device)
        dist_i = (torch.linalg.norm(position-x_obst, dim=1)/(r_obst+self.margin)).view(-1)
        hit_obst = hit_obst + 1.0*(dist_i<1.)
        return (1.-hit_obst).view(-1)
    
    def dist_action(self,actions):
        '''
            Define the control cost
            input_shape = (batch_size, dim_action)
        '''
        d_action = torch.linalg.norm(actions,dim=-1)**2
        return d_action        


    def reward_state_action(self, state, action):
        ''' 
            Compute the stage reward given the action (ctrl input) and the state
        '''
        if self.order == 1:
            position = state[:,:self.dim]
            x_obst = state[:,self.dim:2*self.dim]
            radius_obst = state[:,-1].view(state.shape[0],1)
            action = action.view(action.shape[0],self.dim_action)
            d_goal = (self.dist_position(position)).view(-1)
            r_goal = -1*(d_goal/self.b_goal)

        else:
            position = state[:, :self.dim]
            velocity = state[:,self.dim:2*self.dim]
            x_obst = state[:,2*self.dim:3*self.dim]
            radius_obst = state[:,-1].view(state.shape[0],1)
            action = action.view(action.shape[0],self.dim) #acceleration
            d_goal = (self.dist_position(position)).view(-1)
            d_velocity = (self.dist_position(velocity)).view(-1)
            r_pos =  -1*(d_goal/self.b_goal)
            r_vel = -1*(d_velocity/self.b_velocity)
            r_goal = 1*r_pos + 0.00*r_vel

        # r_goal = -1.0 + 2.0*(d_goal<0.1) 
        d_obst = (self.dist_collision(position,x_obst,radius_obst)).view(-1) # range:(-1,inf), where (-1,0): within the obst, (0,inf): away from obstacle
        r_obst = (((d_obst/self.b_obst))*(d_obst<0))#-1 + torch.sigmoid(d_obst/0.01) # (-1,0)
#         d_obst = -1 + torch.exp(-p_obst**2)
        
        
        
        d_action = (self.dist_action(action)).view(-1)
        r_action = -1*(d_action/self.b_action) #-1 + torch.exp(-d_action/self.b_action) #-1+torch.exp(-(d_action/self.b_action)) # -1 + torch.exp(-d_action**2) #torch.exp(-d_action**2) # (-1,0) #-torch.log(1e-2+d_action)#
        r_total = self.w_scale*(r_goal*self.w_goal+ r_obst*self.w_obst + r_action*self.w_action)
        r_all = torch.cat(( r_goal.view(-1,1), r_action.view(-1,1), r_obst.view(-1,1)),dim=1)
        return r_total, r_all
 
 
################################################################################################################
################################################################################################################

################################################################################################################
################################################################################################################




class PlanarManipulator:
    def __init__(self, n_joints, link_lengths, theta_max, theta_min, dtheta_max, dtheta_min,
                 action_max, action_min, order=1, dt=0.01,
                 x_obst=[],r_obst=[], n_kp=3,
                 w_goal=0.75,w_action=0.25,w_obst=0, w_scale = 1.0, device="cpu"):
        ''' 
            n_joints: number of joints in the planar manipulator
            max_theta: max joint angle (same for al joints)
            dtheta_max: max step change in joint angle
            link_lengths: a list containing length of each link
            n_kp: number of key-points on each link (for collision check)
        '''
        self.device = device
        self.n_joints = n_joints
        self.link_lengths = link_lengths.to(device) 
        
        assert n_joints== len(link_lengths), 'The length of the list containing link_lengths should match n_joints'
        
        self.theta_max = theta_max.to(device)
        self.theta_min = theta_min.to(device)

        self.dtheta_max = dtheta_max.to(device)
        self.dtheta_min = dtheta_min.to(device)
        
        self.action_max = action_max.to(device)
        self.action_min = action_min.to(device)
        

        self.dt = dt

        self.x_obst = x_obst
        self.r_obst = r_obst

        self.n_kp = n_kp
        assert self.n_kp>=2, 'number of key points should be at least two'
        self.key_points = torch.empty(self.n_joints,self.n_kp).to(device)
        for i in range(n_joints):
            self.key_points[i] = torch.arange(0,self.n_kp).to(device)/(self.n_kp-1)

        self.margin = 0.05

        self.b_action = 1*torch.linalg.norm(action_max)
        
        self.b_goal = 1.0#*torch.sum(self.link_lengths)
        self.b_obst = 1.0

        w_total = w_obst + w_goal +w_action
        self.w_obst = w_obst/w_total
        self.w_goal = w_goal/w_total # importance on state 
        self.w_action = w_action/w_total # importance of low control inputs
        self.w_scale = w_scale

        self.alpha_goal = self.w_goal/self.b_goal
        self.alpha_action = self.w_action/self.b_action
        self.alpha_obst = self.w_obst/self.b_obst

        if order==1:
            self.forward_simulate = self.forward_simulate_1
        else: # acceleration control
            self.forward_simulate = self.forward_simulate_2
        
        self.dim_action = n_joints


    # forward kinematics
    def forward_kin(self, q):
        ''' Given a batch of joint angles find the position of all the key-points and the end-effector  '''
        batch_size = q.shape[0]
        q[:,:self.n_joints] = torch.clip(q[:,:self.n_joints],self.theta_min, self.theta_max)
        q_cumsum = torch.zeros(batch_size,self.n_joints).to(self.device)
        for joint in range(self.n_joints):
            q_cumsum[:,joint] = torch.sum(q[:,:joint+1],dim=1)

        cq = torch.cos(q_cumsum).view(batch_size,-1,1)
        sq = torch.sin(q_cumsum).view(batch_size,-1,1)
        cq_sq = torch.cat((cq,sq),dim=2)

        joint_loc = torch.zeros((batch_size, self.n_joints+1, 2)).to(self.device)
        key_loc = torch.empty((batch_size, self.n_joints,self.n_kp,2)).to(self.device)
        for i in range(self.n_joints):
            joint_loc[:,i+1,:] = joint_loc[:,i,:]+self.link_lengths[i]*cq_sq[:,i,:]
            key_loc[:,i,:,:] = joint_loc[:,i,:][:,None,:] + (joint_loc[:,i+1,:]-joint_loc[:,i,:])[:,None,:]*self.key_points[i].reshape(1,-1,1)
    
        end_loc = joint_loc[:,-1,:]
        # find the orientation of end-effector in range (0,2*pi)
        theta_orient = torch.fmod(q_cumsum[:,-1],2*torch.pi)
        theta_orient[theta_orient<0] = 2*torch.pi+theta_orient[theta_orient<0]

        return key_loc, joint_loc, end_loc, theta_orient 

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

    def forward_simulate_2(self, state, action):
        '''
        Given (state,action)  find the next state 
        state: (theta,dtheta) 
        action: acceleration
        '''
        x_goal = state[:,:2]
        theta = state[:,2:2+self.n_joints]
        dtheta = state[:,2+self.n_joints:]
        action = torch.clip(action, self.action_min, self.action_max)
        d_dtheta = action*self.dt
        d_theta = dtheta*self.dt
       
        theta_new = torch.clip(theta+d_theta, self.theta_min, self.theta_max)
        dtheta_new = torch.clip(dtheta+d_dtheta, self.dtheta_min, self.dtheta_max)
        # kp_loc, joint_loc, x_ee, theta_ee = self.forward_kin(theta_new) # get position of key-points and the end-effector

        # collision_free = 1.0 - self.collision(theta).view(-1,1)

        state_new = torch.cat((x_goal, theta_new, dtheta_new),dim=-1)
        # state_new = state_new*(collision_free)+ (1-collision_free)*state
    
        return state_new

    def forward_simulate_1(self, state, action):
        '''
        Given (state,action) find the next state 
        state: theta
        action: velocity
        '''
        x_goal = state[:,:2]
        theta = state[:,2:]
        action = torch.clip(action, self.action_min, self.action_max)
        d_theta = action*self.dt
        theta_new = torch.clip(theta+d_theta, self.theta_min, self.theta_max)
        kp_loc, joint_loc, x_ee, theta_ee = self.forward_kin(theta_new) # get position of key-points and the end-effector
        # collision_free = 1.0 - self.collision(kp_loc).view(-1,1)
        state_new = torch.cat((x_goal, theta_new),dim=-1)
        # state_new = state_new*(collision_free)+ (1-collision_free)*state
        return state_new
        


    def dist_goal(self, x_goal, x):
        '''
        metric for error in end-effector pose from the desired
        x_goal: desired position of the end-effector, batch x 2
        x : actual position/location of the end-effector, batch x 2
        '''
        d_goal = torch.linalg.norm(x_goal-x, dim=-1)
        return d_goal

    def dist_action(self,action):

        ''' 
        Quantify joint angle sweep 
        theta_t: batch x joint, d_theta: batch  x joint
        '''
        d_action = torch.linalg.norm(action,dim=-1)
        return d_action


    def dist_collision(self, kp_loc):
        ''' 
            signed distance # (-1, 0) inside the obstacle, (1,inf) outside
            kp_loc: batch x joint x key-point x coordinate
        '''
        batch_size=kp_loc.shape[0]
        d_collisions = torch.zeros(batch_size).to(self.device)
        for i in range(len(self.x_obst)):
            dist2centre = torch.linalg.norm(kp_loc-self.x_obst[i].view(1,1,1,-1), dim=-1).view(batch_size,-1)
            dist_in = -1.0+(dist2centre/(self.r_obst[i]+self.margin)) 
            d_collisions += torch.min(dist_in,dim=-1)[0]
        return d_collisions.view(-1)

    def collision(self, kp_loc):
        ''' 
            return if the manipulator is in collision
            kp_loc: batch x joint x key-point x coordinate
        '''
        batch_size=kp_loc.shape[0]
        d_collisions = torch.zeros(batch_size).to(self.device)
        for i in range(len(self.x_obst)):
            dist2centre = torch.linalg.norm(kp_loc-self.x_obst[i].view(1,1,1,-1), dim=-1).view(batch_size,-1)
            dist_in = (dist2centre/(self.r_obst[i]+self.margin))
            dist_in = 1.0*(dist_in<1) #(1-dist_in)*(dist_in<1)
            d_collisions += torch.sum(dist_in,dim=-1)
        return 1.0*(d_collisions>0).view(-1)

    def reward_state_action(self,state,action):
        next_state = self.forward_simulate(state,action)
        x_goal_1 = state[:,:2] # desired position of the end-effector
        x_goal_2 = next_state[:,:2] # desired position of the end-effector
        theta_1 = state[:,2:2+self.n_joints]
        theta_2 = next_state[:,2:2+self.n_joints]

        kp_loc_1, joint_loc_1, x_ee_1, theta_ee_1 = self.forward_kin(theta_1) # get position of key-points and the end-effector
        kp_loc_2, joint_loc_2, x_ee_2, theta_ee_2 = self.forward_kin(theta_2) # get position of key-points and the end-effector

        d_obst_1= self.dist_collision(kp_loc_1)
        d_obst_2= self.dist_collision(kp_loc_2)
        r_obst_1 = 1*(torch.tanh(d_obst_1/self.b_obst))
        r_obst_2 = 1*(torch.tanh(d_obst_2/self.b_obst))
        r_obst = 0.5*(r_obst_1+r_obst_2)

        d_goal_1 = self.dist_goal(x_goal_1, x_ee_1.view(-1,2))
        d_goal_2 = self.dist_goal(x_goal_2, x_ee_2.view(-1,2))
        # r_goal = -1 + torch.exp(-(d_goal/0.1)**2)#1-d_goal#-1*(d_goal/self.b_obst)#-1 + torch.exp(-(d_goal/self.b_goal)**)#-1*(d_goal/self.b_goal)**2 # (0,1) #(-1.0 + (2.0/(1.0+d_goal))*(1.-collision)) -1.*collision #torch.exp(-d_goal**2) # (0,1)
        d_goal = 0.5*(d_goal_1+d_goal_2)
        r_goal = -1*(d_goal/0.1)**2#
        d_action = (self.dist_action(action)).view(-1)/self.b_action
        r_action = -d_action#-1+1/(1+d_action)# -1+torch.exp(-(d_action/self.b_action)**2)#-1*(d_action/self.b_action)**2 #torch.exp(-d_action**2) # (-1,0) -1.0 + 1.0/(1.0+d_action) #

        r_total= (r_goal*self.w_goal+r_obst*self.w_obst+r_action*self.w_action)#/(self.w_goal+self.w_obst+self.w_action)
        r_all = torch.cat(( r_goal.view(-1,1), r_action.view(-1,1), r_obst.view(-1,1)),dim=1)
        return r_total, r_all


#####################################################################################################
#####################################################################################################
#####################################################################################################

class CarRobot:
    '''
        Kinematics of a car-like robot
        state: (x,y,theta,phi)
        action: (u_1, u_2)
        where u_1 is linear velocity of the wheel
        and u_2 is angular velocity of the steering angle
        phi: steering angle
        (x,y): position of the center of the rear wheel
    '''
    def __init__(self, state_min, state_max, action_max, action_min, w_scale=1.0, w_goal=0.9,w_action=0.1, axis_length=1.0, dt=0.01, device='cpu'):
        self.axis_length = axis_length # length of the car
        self.state_min = state_min # min of (x,y,theta,phi)
        self.state_max = state_max
        self.action_max = action_max
        self.action_min = action_min
        self.dt=0.01
        self.b_action = 1*torch.linalg.norm(action_max)
        self.b_goal = 1

 
        self.w_goal = w_goal
        self.w_action = w_action


        self.alpha_goal = self.w_goal/self.b_goal
        self.alpha_action = self.w_action/self.b_action

        self.dim=4
        self.dim_action = 2

    def get_B(self,state):
        ''' Assuming control affinr form: x_dot = A(x) + B(x)u. Return B(x) given state x'''
        x = state[:,0]; y = state[:,1]; theta = state[:,2]; phi = state[:,3]
        B = torch.zeros(state.shape[0],self.dim,2).to(self.device)
        B[:,0,0] = torch.cos(theta) #dx
        B[:,1,0] = torch.sin(theta) #dy
        B[:,2,0] = torch.tan(phi)/self.axis_length #dtheta
        B[:,3,1] = 1.
        return B

    def get_R(self,state):
        R = torch.eye(self.dim_action).to(self.device).view(1,-1,-1).expand(state.shape[0])*self.alpha_action
        return R

    def forward_simulate(self, state, action):
        # action = torch.clip(action,self.action_min,self.action_min)
        x = state[:,0]; y = state[:,1]; theta = state[:,2]; phi = state[:,3]
        d_x = action[:,0]*torch.cos(theta)
        d_y = action[:,0]*torch.sin(theta)
        d_theta = action[:,0]*torch.tan(phi)/self.axis_length
        d_phi = action[:,1]

        # no_corner_contact  = 1.0*((state[:,0].abs()<(self.state_max[0]-1e-2)) + (state[:,1].abs()<(self.state_max[1]-1e-2)))
        # no_corner_contact = 1.0*(no_corner_contact)

        x = x+d_x*self.dt 
        y = y+d_y*self.dt
        theta = theta + d_theta*self.dt
        phi = phi + d_phi*self.dt

        theta =  (torch.abs(theta)<torch.pi)*theta+(theta>torch.pi)*(theta-2*torch.pi)+(theta<-torch.pi)*(2*torch.pi+theta) # theta is in range (-pi,pi)
        state_new = torch.concat((x.view(-1,1),y.view(-1,1),theta.view(-1,1),phi.view(-1,1)),dim=-1)
        state_new = torch.clip(state_new, self.state_min, self.state_max)
        return state_new

    def reward_state_action(self,state,action):
        next_state = self.forward_simulate(state,action)
        d_pos_1 = torch.linalg.norm(state[:,:2], dim=-1)
        d_pos_2 = torch.linalg.norm(next_state[:,:2], dim=-1)
        d_theta_1 = torch.abs(state[:,2])
        d_theta_2 = torch.abs(next_state[:,2])**2
        d_pos = 0.5*(d_pos_1+d_pos_2)
        d_theta = 0.5*(d_theta_1+d_theta_2)
        r_pos = -1*(d_pos)**2
        r_theta = -1*d_theta
        d_goal = d_pos + d_theta
        r_goal =  r_pos+0*r_theta
        d_action = torch.linalg.norm(action,dim=-1)
        r_action = -1*(d_action**2)
        r_all = self.w_goal*r_goal + self.w_action*r_action
        return r_all

#####################################################################################################
#####################################################################################################
#####################################################################################################

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


#####################################################################################################
#####################################################################################################
#####################################################################################################

class CartPole:
    '''
        Dynamics of a single pendulum
        state: (theta, dtheta) # (joinr_angle,joint_vel), joint_angle in (-pi, pi), it is 0 at the stable equilibrium
        action: (u) # joint torquw
    '''
    def __init__(self,
                     w_scale=1.0, w_goal=0.9,w_action=0.1, 
                     b_c=0.1, b_p=0.1, 
                     length=0.3365, mass_pendulum=0.127, 
                     mass_cart=0.57, dt=0.01,
                     force_control=True, device='cpu'):
        self.l = length # length of the pendulum
        self.g = 9.81 # gravity
        self.b_p = b_p # damping coeficient 
        self.b_c = b_c
        self.M = mass_cart # mass of cart
        self.m = mass_pendulum


        self.state_max = torch.tensor([100000000,100,torch.pi,10*torch.pi]).to(device)
        self.state_min = -1*self.state_max
        self.action_max = torch.tensor([1000]).to(device) # max  force or acceleration
        self.action_min = -self.action_max
    
        # self.state_max = state_max
        # self.state_min = state_min
        # self.action_max = action_max
        # self.action_min = action_min
        self.dt=dt
        w_total = w_goal+w_action
        self.w_goal = w_goal/w_total
        self.w_action = w_action/w_total

        self.w_scale = w_scale


        self.forward_simulate = self.forward_simulate_2 if force_control else self.forward_simulate_1


    def forward_simulate_1(self,state,action):
        '''
        Cart acceleration Control
        state: [position_cart, pole_angle, vel_cart, vel_pole]
        control: acceleration of the cart  Or Force on the cart
        '''
        x = state[:,0]
        dx = state[:,1]
        theta = state[:,2]
        dtheta = state[:,3]
        
        
        no_corner_contact = (x.abs()<(self.state_max[0]))*1.0 
        # no_corner_contact += (theta.abs() > (self.state_max[0]+1e-2))*1.0 
        # no_corner_contact += 1.0*(dtheta.abs()<(self.state_max[-1]-1e-2))
        # no_corner_contact = 1.0*(no_corner_contact>0)

        ddx = action[:,0]*no_corner_contact #-1*dx/self.dt*(1-no_corner_contact)
        ddtheta = -1*(self.g*torch.sin(theta)+ddx*torch.cos(theta)+self.b_p*dtheta/(self.m*self.l))/self.l
        ddtheta = ddtheta*no_corner_contact
        
        x = x+dx*self.dt*no_corner_contact        
        dx = (dx+ddx*self.dt)*no_corner_contact
        theta = theta + dtheta*self.dt*no_corner_contact
        dtheta = (dtheta + ddtheta*self.dt)*no_corner_contact

        theta =  (torch.abs(theta)<=torch.pi)*theta+(theta>torch.pi)*(theta-2*torch.pi)+(theta<-torch.pi)*(2*torch.pi+theta) # theta is in range (-pi,pi)
        state_new = torch.concat((x.view(-1,1), dx.view(-1,1),theta.view(-1,1),dtheta.view(-1,1)),dim=-1)
        # state_new = torch.clip(state_new, self.state_min, self.state_max)
        return state_new
       



    def forward_simulate_2(self, state, action):
        ''' Force control of the cart '''
        # action = torch.clip(action,self.action_min,self.action_min)
        
        
        x = state[:,0]
        dx = state[:,1]#*(x.abs() < self.state_max[0]-0.01)
        theta = state[:,2]
        ctheta = torch.cos(theta)
        stheta = torch.sin(theta)
#         theta = state[:,2]
        dtheta = state[:,-1]
        F = action[:,0] # force on cart
        no_corner_contact = (x.abs()<self.state_max[0])*1.0
        
        a0 = 1.0/((self.m+self.M)*((self.m*self.l**2)/4)-(0.5*self.m*self.l*ctheta)**2)
        b1 = -0.5*self.m*self.l*stheta*(dtheta**2) + F  - self.b_c*dx 
        b2 = 0.5*self.m*self.g*self.l*stheta -self.b_p*dtheta
        a1 = (self.m*self.l**2)/4.0
        a2 = -0.5*self.m*self.l*ctheta
        a3 = a2
        a4 = self.m + self.M        


        ddx = a0*(a1*b1+a2*b2)
        ddtheta = a0*(a3*b1+a4*b2)

        x = x + dx*self.dt
        dx = (dx + ddx*self.dt)*no_corner_contact#*(x.abs() < self.state_max[0]-0.01)
        

        theta = theta + dtheta*self.dt*no_corner_contact 
        dtheta = (dtheta + ddtheta*self.dt)*no_corner_contact 

        theta =  (torch.abs(theta)<=torch.pi)*theta+(theta>torch.pi)*(theta-2*torch.pi)+(theta<-torch.pi)*(2*torch.pi+theta) # theta is in range (-pi,pi)
        state_new = torch.concat((x.view(-1,1), dx.view(-1,1),theta.view(-1,1),dtheta.view(-1,1)),dim=-1)
        # state_new = torch.clip(state_new, self.state_min, self.state_max)
        return state_new

    def reward_state_action(self,state,action):
        next_state = self.forward_simulate(state,action)
        theta_1 = state[:,2] 
        theta_2 = next_state[:,2] 
        x1= state[:,0].abs()
        x2= next_state[:,0].abs()
        dx1 = state[:,1].abs()
        dx2= next_state[:,1].abs()
        d_theta_1 = (theta_1.abs()-torch.pi).abs()
        d_theta_2 = (theta_2.abs()-torch.pi).abs()

        dx = 0.5*(dx1+dx2)
        
        d_theta = 0.5*(d_theta_1+d_theta_2)
        d_x = 0.5*(x1+x2)#/self.state_max[0] 
        d_action = (action[:,0]).abs()#/self.action_max[0]

        r_goal = -1*d_theta/torch.pi - 1*(d_x/0.5) -1*(dx/0.5) 

        r_action = -d_action
        return self.w_scale*(self.w_goal*r_goal + self.w_action*r_action)

#########################################################################################################
#########################################################################################################
#########################################################################################################

class DoublePendulum:
    '''
        Dynamics of a double pendulum
        state: (theta_1, dtheta_1, theta_2, dtheta_2) # (joinr_angle,joint_vel), joint_angle in (-pi, pi), it is 0 at the stable equilibrium
        action: (tau1,tau2) # joint torque
        ref: http://underactuated.mit.edu/acrobot.html
    '''
    def __init__(self, state_min, state_max, action_max, action_min, fully_actuated=False,  w_scale=1.0, w_goal=0.9, w_action=0.1, b=0., length=1.0, mass=1.0, dt=0.01, device='cpu'):
        self.l = length # length of the rod
        self.g = 9.81 # gravity
        self.b = b # damping coeficient for cart
        self.m = mass# mass of the rod
        self.I = (self.m*self.l**2)/3.0 # mass moment of inerta about the pivot

        self.fully_actuated = fully_actuated

        self.state_min = state_min # (x, dot_x, theta, dot_theta)
        self.state_max = state_max 
        self.action_max = action_max # max  force
        self.action_min = action_min

        self.dt=dt
        w_total = w_goal+w_action
        self.w_goal = w_goal/w_total
        self.w_action = w_action/w_total

        self.w_scale = w_scale


    def forward_simulate(self, state, action):
        theta_1 = state[:,0]
        dtheta_1 = state[:,1]
        theta_2 = state[:,2]
        dtheta_2 = state[:,3]
        ctheta_1 = torch.cos(theta_1);ctheta_2 = torch.cos(theta_2)
        stheta_1 = torch.sin(theta_1);stheta_2 = torch.sin(theta_2)
        stheta_12 = torch.sin(theta_1+theta_2)
        
        if self.fully_actuated:
            tau_1 = action[:,0] # torque
            tau_2 = action[:,1]
        else:
            tau_2 = action[:,0]
            tau_1 = tau_2*0.
        
        a1 = self.I
        a2 = -1*(self.I+self.m*self.l**2*ctheta_2)
        a3 = a2
        a4 = 2*self.I+self.m*self.l**2+self.m*self.l**2*ctheta_2
        a0 = 1.0/(a1*a4-a2**2)

        b1 = self.m*(self.l**2)*stheta_2*(dtheta_1*dtheta_2+0.5*(dtheta_2**2))-self.m*self.g*self.l*(1.5*stheta_1+0.5*stheta_12) + tau_1  
        b2 = -0.5*self.m*(self.l**2)*stheta_2*(dtheta_1**2)-0.5*self.m*self.g*self.l*stheta_12 + tau_2

        ddtheta_1 = a0*(a1*b1+a2*b2)
        ddtheta_2 = a0*(a3*b1+a4*b2)

        theta_1 = theta_1 + dtheta_1*self.dt
        dtheta_1 = dtheta_1 + ddtheta_1*self.dt
        theta_2 = theta_2 + dtheta_2*self.dt
        dtheta_2 = dtheta_2 + ddtheta_2*self.dt

        theta_1 =  (torch.abs(theta_1)<torch.pi)*theta_1+(theta_1>torch.pi)*(theta_1-2*torch.pi)+(theta_1<-torch.pi)*(2*torch.pi+theta_1) # theta is in range (-pi,pi)
        theta_2 =  (torch.abs(theta_2)<torch.pi)*theta_2+(theta_2>torch.pi)*(theta_2-2*torch.pi)+(theta_2<-torch.pi)*(2*torch.pi+theta_2) # theta is in range (-pi,pi)

        state_new = torch.concat((theta_1.view(-1,1), dtheta_1.view(-1,1),theta_2.view(-1,1),dtheta_2.view(-1,1)),dim=-1)
        state_new = torch.clip(state_new, self.state_min, self.state_max)
        return state_new

    def reward_state_action(self,state,action):
        print("here")
        theta_j1_1 = (state[:,0].abs()-torch.pi).abs()
        theta_j2_1 = (state[:,2].abs()-torch.pi).abs()
        next_state = self.forward_simulate(state,action)
        theta_j1_2 = (next_state[:,0].abs()-torch.pi).abs()
        theta_j2_2 = (next_state[:,2].abs()-torch.pi).abs()
        theta_j1 = 0.5*(theta_j1_1+theta_j1_2)
        theta_j2 = 0.5*(theta_j2_1+theta_j2_2)
        d_action = (action[:,0]).abs()
        r_goal = -0.5*(theta_j1+theta_j2)**2 #-0.01*d_dx#-0.1*d_dtheta - 
        r_action = -d_action
        ret = self.w_goal*r_goal + self.w_action*r_action 
        print(ret.mean())
        return ret
#########################################################################################################
#########################################################################################################
#########################################################################################################

class HardMove:
    '''
        Dynamics of a Hard-Move Agent

        state: (x, y) 
        action: (switch_i, v_i) i=1,...,n
        v_i is velocity of the system along the thrust given by i-th actuator
        reference: 
    '''
    def __init__(self, dt=0.001, pos_max=1.0, w_goal=1.0, w_action=1.0, n=4, device='cpu'):
        self.n = n 
        self.w_goal = w_goal
        self.w_action = w_action
        self.state_max = torch.tensor([pos_max]*2).to(device)
        self.state_min = -1*self.state_max
        self.device = device
        self.dt = dt

    def forward_simulate(self, state, action):
        x = state[:,0]
        y = state[:,1]
        theta_actuator = (((torch.arange(self.n)*torch.pi*2)/self.n)).view(1,-1).to(self.device)
        idx_switch = torch.arange(self.n).to(self.device)
        switch_on = action[:,2*idx_switch+1].view(-1,self.n)
        v = action[:,idx_switch*2]*switch_on
        x = x + (v*torch.cos(theta_actuator)).sum(dim=-1)*self.dt
        y = y + (v*torch.sin(theta_actuator)).sum(dim=-1)*self.dt
        state_new = torch.concat((x.view(-1,1),y.view(-1,1)),dim=-1)
        return torch.clip(state_new, self.state_min, self.state_max)


    def reward_action(self,action):
        idx_switch = torch.arange(self.n).to(self.device)
        switch_on = action[:,2*idx_switch+1].view(-1,self.n)
        v = action[:,idx_switch*2]
        
        # cost_switch = switch_on.sum(dim=-1).view(-1)
        cost = (torch.abs(v)*switch_on).sum(dim=-1).view(-1)
        return -1*self.w_action*cost       

    def reward_state_action(self,state,action):
        d_goal = torch.linalg.norm(state,dim=-1)
        r_goal = -1*(d_goal/0.01)**2
        r_action = self.reward_action(action)
        return self.w_goal*r_goal + r_action

# #########################################################################################################
# #########################################################################################################
# #########################################################################################################

# class Perching:
#     '''
#         Dynamics of a Perching System
#         state: (x, y, theta, phi, dx, dy, dtheta, dphi) 
#         action: (ddphi, f, psi)
#         ref: 
#     '''
#     def __init__(self, w_goal=0.9, w_action=0.1, 
#                 m=0.05, l=0.35, rho = 1.292, Sw=0.1, Se=0.025, I= 0.006, lw = -0.03, le = 0.04, lt = 0.05,
#                 dt=0.01, device='cpu'):

#         self.l = l # length 
#         self.g = 9.81 # gravity
#         self.m = m# mass of the rod
#         self.I = I # 
#         self.rho = rho
#         self.Sw = Sw
#         self.Se = Se
#         self.lw = lw
#         self.le = le
#         self.lt = lt

#         self.dt=dt
#         self.w_goal = w_goal
#         self.w_action = w_action
#     def forward_simulate(self, state, action):

#     def forward_simulate(self, state, action):
#         pass
#     def reward_state_action(self,state,action):
#         pass
