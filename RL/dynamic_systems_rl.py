import torch
import numpy
# torch.set_default_dtype(torch.float64)
import matplotlib
# matplotlib.use('Agg')
from matplotlib import pyplot as plt
import matplotlib.patches as patches

def logcos_reward(action, action_max, w_action): # related to inverse sigmoid
    alpha = action_max*2/torch.pi
    alpha_1 = 1.15*alpha
    r_action = (alpha*torch.log(torch.cos(action/alpha_1)+1e-6)).sum(dim=-1)
    return w_action*r_action

def normalize_angle(theta): # normalize angle between (-PI.PI)
    return ((theta + torch.pi) % (2 * torch.pi)) - torch.pi


class PointMass:
    def __init__(self, action_max=1.0, max_vel=1.0, max_pos=1.0, dt= 0.01,
                    x_obst=[], r_obst=[], order=1, dim=2, log_action=False,
                    w_obst=1., w_action=1, w_goal=1, device='cpu'):
        ''' 
            dim: dimension of the euclidean space the point-mass lives in
            state : (position) for velocity control  or (position, velocity) for acceleration control
            actions : velocitym or acceleration
        '''
        self.device = device
        self.dim=dim
        self.position_max = torch.tensor([max_pos]*dim).to(device)
        self.position_min = -self.position_max
        self.velocity_max = torch.tensor([max_vel]*dim).to(device)
        self.velocity_min = -self.velocity_max

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
        self.dim_action = dim

        self.action_max = torch.tensor([action_max]*dim).to(device)
        self.action_min = -self.action_max 

        self.dt = dt # time-step
        self.order=order

        if order==1: # velocity control
            self.forward_simulate = self.forward_simulate_1
        elif order==2: # acceleration control
            self.forward_simulate = self.forward_simulate_2
        else:
            self.forward_simulate = self.forward_simulate_3

        self.x_obst = x_obst # positions of the spherical obstacles
        self.r_obst = r_obst # radii of the obstacles

        self.margin = 0.01 # for obstacle avoidance

        self.b_action = 1.
        self.b_obst =1.
        self.b_goal = 1.
        self.b_velocity = 1
        self.w_obst = w_obst
        self.w_goal = w_goal# importance on state 
        self.w_action = w_action # importance of low control inputs
        self.log_action = log_action


    def forward_simulate_3(self, state, action, dt):
        '''
        Given (state,action) find the next state 
        state: (position_current, position_target)
        action: velocity
        '''
        # action = torch.clip(action, self.action_min, self.action_max)
        d_position = action*dt
        state_new = 1*state
        state_new[:,:self.dim] = state[:,:self.dim] + d_position
        # state_new = torch.clip(state+d_position, self.position_min, self.position_max)
        # print(state[0],action[0],dt,state_new[0])
        return state_new#torch.clip(state_new, self.state_min, self.state_max)

    def forward_simulate_2(self, state, action, dt):
        '''
        Given (state,action)  find the next state 
        state: (position,velocity)
        action: acceleration
        '''

        position = state[:,:self.dim]
        velocity = state[:,self.dim:]
        action = torch.clip(action, self.action_min, self.action_max)
        d_position = velocity*dt
        d_velocity = action*dt
        
       
        position_new = torch.clip(position+d_position, self.position_min, self.position_max)
        velocity_new = torch.clip(velocity+d_velocity, self.velocity_min, self.velocity_max)
        # collision_free = self.is_collision_free(position).view(-1,1)

        state_new = torch.cat((position_new,velocity_new),dim=-1)    
        return state_new

    def forward_simulate_1(self, state, action, dt):
        '''
        Given (state,action) find the next state 
        state: position
        action: velocity
        '''
        action = torch.clip(action, self.action_min, self.action_max)
        d_position = action*dt
        state_new = torch.clip(state+d_position, self.position_min, self.position_max)
        return state_new#torch.clip(state_new, self.state_min, self.state_max)

    def get_Bt(self,state):
        ''' Assuming control affine form: x_dot = A(x) + B(x)u. Return B(x).T given state x'''
        if self.order == 1:
            Bt = torch.eye(self.dim_state).to(self.device)
        else:
            B = torch.zeros(self.dim_state,self.dim).to(self.device)
            B[-self.dim:,:]=torch.eye(self.dim).to(self.device)
            Bt = B.T
        Bt = Bt[None,:,:].expand(state.shape[0],-1,-1)
        return Bt # batch x dim_action x dim_state
    
    def dist_position(self, position):
        # cost w.r.t. error to goal: (ex,ey) 
        ''' input shape: (batch_size, dim_state) '''
        d_x = torch.linalg.norm(position-self.target_position, dim=1)# batch_size x 1
        return d_x

    def dist_velocity(self, velocity):
        d_v = torch.linalg.norm(velocity, dim=1)# batch_size x 1
        return d_v

    def dist_collision(self,position):
        ''' 
            signed distance function (-1,inf), dist<0 inside the obstacle
            input shape: (batch_size,dim), find the distance into obstacle (-1 to inf), -1 completely inside, 0: at the boundary
        '''
        batch_size = position.shape[0]
        dist_obst = torch.zeros(batch_size,1).to(self.device)
        for i in range(len(self.x_obst)):
             dist_i = -1 + (torch.linalg.norm(position-self.x_obst[i], dim=1)/(self.r_obst[i]+self.margin)).view(-1)
             dist_obst = torch.min(torch.concat((dist_obst.view(-1,1), torch.clip(dist_i,-torch.inf, 0.).view(-1,1)),dim=-1), dim=-1)[0].view(-1,1)
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



    def reward_action(self, action): # related to inverse sigmoid 
        if self.log_action:
            r = logcos_reward(action, self.action_max, self.w_action)
        else:
            r =  -self.w_action*torch.linalg.norm(action,dim=-1)
        return r

    
    def reward_state_action(self, state, action):
        ''' 
            Compute the stage reward given the action (ctrl input) and the state
        '''
        if self.order == 1:
            d_goal = (torch.linalg.norm(state,dim=-1)/0.05).view(-1).abs()

        elif self.order==2:
            d_goal = (torch.linalg.norm(state[:,:self.dim],dim=-1)/0.05).view(-1).abs() #d_pos + d_goal
            # d_goal += 0.1*(torch.linalg.norm(state[:,self.dim:],dim=-1)/0.01).view(-1).abs() 
        else:
            d_goal = (torch.linalg.norm(state[:,:self.dim]-state[:,self.dim:],dim=-1)/0.1).view(-1).abs() #d_pos + d_goal

        r_goal = -1*d_goal
        d_obst = 1.0*((self.dist_collision(state[:,:self.dim])).view(-1)/0.01).abs() # range:(-1,inf), where (-1,0): within the obst, (0,inf): away from obstacle
        r_obst = -1*d_obst

        r_action = self.reward_action(action)
        
        r_total = (r_goal*self.w_goal+ r_obst*self.w_obst) + r_action
        return r_total
    




 
################################################################################################################
################################################################################################################

################################################################################################################
################################################################################################################

class PlanarManipulator:
    def __init__(self, n_joints, link_lengths, theta_max, theta_min, dtheta_max, dtheta_min,
                 action_max, order=1, dt=0.01,
                 x_obst=[],r_obst=[], n_kp=3,
                 w_goal=0.75,w_action=0.25,w_obst=0,
                 joint_limit=False,
                 log_action=False,
                 device="cpu"):
        ''' 
            n_joints: number of joints in the planar manipulator
            max_theta: max joint angle (same for al joints)
            dtheta_max: max step change in joint angle
            link_lengths: a list containing length of each link
            n_kp: number of key-points on each link (for collision check)
        '''
        self.log_action = log_action
        self.device = device
        self.n_joints = n_joints
        self.link_lengths = link_lengths.to(device) 
        
        assert n_joints== len(link_lengths), 'The length of the list containing link_lengths should match n_joints'
        
        self.theta_max = theta_max.to(device)
        self.theta_min = theta_min.to(device)

        self.dtheta_max = dtheta_max.to(device)
        self.dtheta_min = dtheta_min.to(device)
        
        self.action_max = action_max
        self.action_min = -1*action_max
        

        self.dt = dt

        self.x_obst = x_obst
        self.r_obst = r_obst

        self.n_kp = n_kp
        assert self.n_kp>=2, 'number of key points should be at least two'
        self.key_points = torch.empty(self.n_joints,self.n_kp).to(device)
        for i in range(n_joints):
            self.key_points[i] = torch.arange(0,self.n_kp).to(device)/(self.n_kp-1)

        self.margin = 0.05

        self.w_obst = w_obst
        self.w_goal = w_goal # importance on state 
        self.w_action = w_action # importance of low control inputs
        
        self.order=order
        self.dim = n_joints
        if order==1:
            self.dim_state = self.n_joints+2
            self.forward_simulate = self.forward_simulate_1
        elif order==2: # acceleration control
            self.dim_state = 2*self.n_joints+2
            self.forward_simulate = self.forward_simulate_2
        else:
            self.dim_state = self.n_joints*2
            self.forward_simulate = self.forward_simulate_3

        self.dim_action = n_joints
        self.joint_limit = joint_limit


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

    def get_Bt(self,state):
        ''' Assuming control affine form: x_dot = A(x) + B(x)u. Return B(x).T given state x'''

        B = torch.zeros(state.shape[0], self.dim_state, self.n_joints).to(self.device)
        if self.order == 1:
            q_cumsum = torch.zeros(state.shape[0],self.n_joints).to(self.device)
            for joint in range(self.n_joints):
                q_cumsum[:,joint] = torch.sum(state[:,2+joint:],dim=1)

            l_cq_cumsum = torch.cos(q_cumsum).view(state.shape[0],1,-1)*self.link_lengths.view(1,-1)
            l_sq_cumsum = torch.sin(q_cumsum).view(state.shape[0],1,-1)*self.link_lengths.view(1,-1)
            l_cs_sumsum = torch.cat((-l_sq_cumsum,l_cq_cumsum),dim=1)

            for joint in range(self.n_joints):
                B[:,:2, joint] = torch.sum(l_cs_sumsum[:,:,joint:], dim=-1)        
            B[:,2:, :] = torch.eye(self.n_joints)[None,:,:].expand(state.shape[0],-1,-1)
        else:
            B[:, -self.n_joints:,:] = torch.eye(self.n_joints)[None,:,:].expand(state.shape[0],-1,-1)
        Bt = B.permute([0,2,1])
        return Bt # batch x dim_action x dim_state

    def reward_action(self, action): # related to inverse sigmoid
        return self.w_action*torch.linalg.norm(action,dim=-1)/self.action_max[0]#logcos_reward(action, self.action_max, self.w_action)

    def forward_simulate_3(self, state, action, dt):
        '''
        Given (state,action) find the next state 
        state: theta
        action: velocity
        '''
        theta = state[:,:self.n_joints]
        theta = theta + action*dt

        if self.joint_limit:
            theta = torch.clip(theta, self.theta_min, self.theta_max)# if joint limit
        else:
            theta = normalize_angle(theta) # between -pi and pi
        state_new = torch.cat((theta, state[:,self.n_joints:]),dim=-1)
        return state_new
      

    def forward_simulate_2(self, state, action, dt):
        '''
        Given (state,action)  find the next state 
        state: (theta,dtheta) 
        action: acceleration
        '''
        x_goal = state[:,:2]
        theta = state[:,2:2+self.n_joints]
        dtheta = state[:,2+self.n_joints:]
        action = torch.clip(action, self.action_min, self.action_max)
        
        theta = theta + dtheta*dt
        dtheta = dtheta + action*dt
        

        if self.joint_limit:
            theta = torch.clip(theta, self.theta_min, self.theta_max)# if joint limit
        else:
            theta = normalize_angle(theta) # between -pi and pi
        # dtheta = torch.clip(dtheta, self.dtheta_min, self.dtheta_max)

        state_new = torch.cat((x_goal, theta, dtheta),dim=-1)
    
        return  state_new

    def forward_simulate_1(self, state, action, dt):
        '''
        Given (state,action) find the next state 
        state: theta
        action: velocity
        '''
        x_goal = state[:,:2]
        theta = state[:,2:]
        action = torch.clip(action, self.action_min, self.action_max)
        theta = theta + action*dt

        if self.joint_limit:
            theta = torch.clip(theta, self.theta_min, self.theta_max)# if joint limit
        else:
            theta = normalize_angle(theta) # between -pi and pi
        state_new = torch.cat((x_goal, theta),dim=-1)
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
        if self.order==1 or self.order==2:
            x_goal = state[:,:2] # desired position of the end-effector
            theta = state[:,2:2+self.n_joints]
            d_goal = self.dist_goal(x_goal, x_ee.view(-1,2))/0.1
        else:
            theta = state[:,:self.n_joints]
            theta_goal = state[:,self.n_joints:]
            d_goal = torch.linalg.norm((theta-theta_goal),dim=-1)
        kp_loc, joint_loc, x_ee, theta_ee = self.forward_kin(theta) # get position of key-points and the end-effector
        d_obst= self.dist_collision(kp_loc)
        r_obst = 1*(torch.tanh(d_obst/1.0))
        r_goal = -1*(d_goal).abs()#
        r_action = self.reward_action(action)
        r_total= (r_goal*self.w_goal+r_obst*self.w_obst)+r_action
        return r_total


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

        theta =  (torch.abs(theta)<=torch.pi)*theta+(theta>torch.pi)*(theta-2*torch.pi)+(theta<-torch.pi)*(2*torch.pi+theta) # theta is in range (-pi,pi)
        state_new = torch.concat((x.view(-1,1),y.view(-1,1),theta.view(-1,1),phi.view(-1,1)),dim=-1)
        # state_new = torch.clip(state_new, self.state_min, self.state_max)
        return state_new

    def reward_action(self, action): # related to inverse sigmoid
        return logcos_reward(action, self.action_max, self.w_action)

    def reward_state_action(self,state,action):
        d_pos = torch.linalg.norm(state[:,:2], dim=-1)
        d_theta = torch.abs(state[:,2])
        r_goal = -1*(d_pos)**2
        r_action = -self.reward_action(action)
        reward = self.w_goal*r_goal + self.w_action*r_action
        return reward


#####################################################################################################
#####################################################################################################
#####################################################################################################

class SinglePendulum_v2:
    '''
        Dynamics of a single pendulum
        state: (theta, dtheta) # (joinr_angle,joint_vel), joint_angle in (-pi, pi), it is 0 at the stable equilibrium
        action: (u) # joint torque
    '''
    def __init__(self, state_min, state_max, action_max, coef_viscous=0.,
                  w_goal=1.0,w_action=0.001, length=1.0, mass=1.0, device='cpu',
                  use_gym=False):
        self.l = length # length of the rod
        self.lc = 0.5*self.l # com
        self.m = mass
        self.I = (mass*self.lc**2)/3.0
        self.mgl = mass*9.81*self.lc 
        self.g = 9.81
        self.coef_viscous = coef_viscous

        self.state_min = state_min
        self.state_max = state_max # max of ()
        self.dim_state = 2 # (theta, dtheta)
        self.dim_action = 1
        self.action_max = action_max
        self.action_min = -action_max
        self.w_goal = w_goal
        self.w_action = w_action
        self.device = device
        self.use_gym = use_gym

    def get_Bt(self,state):
        ''' Assuming control affine form: x_dot = A(x) + B(x)u. Return B(x).T given state x'''
        Bt = torch.ones(1,state.shape[-1]).to(self.device)*(1/self.I)
        Bt[:,:1] = 0
        Bt = Bt[None,:,:].expand(state.shape[0],-1,-1)
        return Bt # batch x dim_action x dim_state

    def reward_action(self, action): # related to inverse sigmoid
        return logcos_reward(action, self.action_max, self.w_action)
 

    def forward_simulate(self, state, action, dt):
        theta = state[:,0]
        dtheta = state[:,1]

        tau_gravity  = self.mgl*torch.sin(theta)
        tau_viscousity = self.coef_viscous*dtheta
        u = torch.clip(action[:,0], self.action_min,self.action_max)

        ddtheta = (1/self.I)*(u- tau_gravity - tau_viscousity)

        
        dtheta = dtheta + ddtheta*dt
        dtheta = torch.clip(dtheta, self.state_min[-1],self.state_max[-1])
        
        theta = theta + dtheta*dt
        
        theta = ((theta + torch.pi) % (2 * torch.pi)) - torch.pi

        # theta =  (torch.abs(theta)<=torch.pi)*theta+(theta>torch.pi)*(theta-2*torch.pi)+(theta<-torch.pi)*(2*torch.pi+theta) # theta is in range (-pi,pi)

        state_new = torch.concat((theta.view(-1,1),dtheta.view(-1,1)),dim=-1)
        return state_new
    
    def reward_action(self, action): # related to inverse sigmoid
        return logcos_reward(action, self.action_max, self.w_action)
  
    def reward_state_action(self, state, action):
        theta = state[:.0]
        d_vel = state[:,-1]
        dtheta = (theta.abs()-torch.pi).abs()
        r_goal = -(dtheta)**2 - 0.1*d_vel**2
        if not self.use_gym:
            r_action =  self.reward_action(action)
        else:
            r_action = -1*(self.w_action*action**2).view(-1)

        return (self.w_goal*r_goal + r_action)



#####################################################################################################
#####################################################################################################
#####################################################################################################

class SinglePendulum:
    '''
        Dynamics of a single pendulum
        state: (theta, dtheta) # (joinr_angle,joint_vel), joint_angle in (-pi, pi), it is 0 at the stable equilibrium
        action: (u) # joint torque
    '''
    def __init__(self, state_min, state_max, action_max, coef_viscous=0.,
                  w_goal=1.0,w_action=0.001, length=1.0, mass=1.0, device='cpu',
                  use_gym=False):
        self.l = length # length of the rod
        self.lc = 0.5*self.l # com
        self.m = mass
        self.I = (mass*self.lc**2)/3.0
        self.mgl = mass*9.81*self.lc 
        self.coef_viscous = coef_viscous

        self.state_min = state_min
        self.state_max = state_max # max of ()
        self.dim_state = 3 # (stheta, ctheta, dtheta)
        self.dim_action = 1
        self.action_max = action_max
        self.action_min = -action_max
        self.w_goal = w_goal
        self.w_action = w_action
        self.device = device
        self.use_gym = use_gym

    def get_Bt(self,state):
        ''' Assuming control affine form: x_dot = A(x) + B(x)u. Return B(x).T given state x'''
        Bt = torch.ones(1,state.shape[-1]).to(self.device)*(1/self.I)
        Bt[:,:2] = 0
        Bt = Bt[None,:,:].expand(state.shape[0],-1,-1)
        return Bt # batch x dim_action x dim_state

    def reward_action(self, action): # related to inverse sigmoid
        return logcos_reward(action, self.action_max, self.w_action)
 

    def forward_simulate(self, state, action, dt):
        stheta = state[:,0]
        ctheta = state[:,1]
        dtheta = state[:,2]
        theta = torch.arctan2(stheta,ctheta)

        tau_gravity  = self.mgl*stheta
        tau_viscousity = self.coef_viscous*dtheta
        u = torch.clip(action[:,0], self.action_min,self.action_max)
        
        ddtheta = (1/self.I)*(u- tau_gravity - tau_viscousity)
        
        theta = theta + dtheta*dt
        dtheta = dtheta + ddtheta*dt             

        stheta = torch.sin(theta)
        ctheta = torch.cos(theta)

        state_new = torch.concat((stheta.view(-1,1),ctheta.view(-1,1),dtheta.view(-1,1)),dim=-1)
        return state_new


    def reward_action(self, action): # related to inverse sigmoid
        return logcos_reward(action, self.action_max, self.w_action)
  
    def reward_state_action(self, state, action):
        theta = torch.arctan2(state[:,0], state[:,1])
        d_vel = state[:,-1]
        dtheta = (theta.abs()-torch.pi).abs()
        r_goal = -(dtheta)**2 - 0.01*d_vel**2
        if not self.use_gym:
            r_action =  self.reward_action(action)
        else:
            r_action = -1*(self.w_action*action**2).view(-1)

        return (self.w_goal*r_goal + r_action)


#####################################################################################################
#####################################################################################################
#####################################################################################################

class CartPole:
    '''
        Dynamics of a single pendulum
        state: (theta, dtheta) # (joinr_angle,joint_vel), joint_angle in (-pi, pi), it is 0 at the stable equilibrium
        action: (u) # joint torquw
    '''
    def __init__(self, action_max,
                     w_goal=0.9,w_action=0.1, 
                     b_c=0.01, b_p=0.01, 
                     length=0.1, mass_pendulum=0.25, 
                     mass_cart=0.25, 
                     force_control=True, log_action=False, device='cpu'):
        self.log_action = log_action
        self.device = device
        self.l = length # length of the pendulum
        self.g = 9.81 # gravity
        self.b_p = b_p # damping coeficient 
        self.b_c = b_c
        self.M = mass_cart # mass of cart
        self.m = mass_pendulum

        # state: (x,dx, stheta, ctheta, dtheta)
        # theta: (-pi,pi), 0 is facing down (stable equilibrium)
        self.state_max = torch.tensor([100,100,1,1,20*torch.pi]).to(device)
        self.state_min = -1*self.state_max
        self.action_max = action_max# max  force or acceleration
        self.action_min = -self.action_max

        self.w_goal = w_goal
        self.w_action = w_action

        self.force_control = force_control
        self.forward_simulate = self.forward_simulate_2 if force_control else self.forward_simulate_1

        

    def forward_simulate_1(self,state,action,dt):
        '''
        Cart acceleration Control
        state: [position_cart, vel_cart, sin(pole_angle), cos(pole_angle), pole_ang_vel]
        control: acceleration of the cart  Or Force on the cart
        '''
        x = state[:,0]
        dx = state[:,1]
        stheta = state[:,2]
        ctheta = state[:,3]
        dtheta = state[:,4]
        
        theta = torch.arctan2(stheta,ctheta)

        ddx = action[:,0]
        ddtheta = -1*(self.g*stheta+ddx*ctheta+self.b_p*dtheta/(self.m*self.l))/self.l
        # ddtheta =  (-self.m*self.l*ctheta*stheta*dtheta**2-\
        #             ddx*ctheta + (self.b_c + self.b_p)*dx*ctheta-\
        #                 self.b_p*self.l*dtheta -\
        #                 (self.m+self.M)*self.g*stheta)/(self.m*self.l**2*(4.0/3.0-(self.m*ctheta**2)/(self.m + self.M)))

            
        x = x+dx*dt       
        dx = dx+ddx*dt
        
        theta = theta + dtheta*dt
        dtheta = dtheta + ddtheta*dt
        

        stheta = torch.sin(theta)
        ctheta = torch.cos(theta)
        state_new = torch.concat((x.view(-1,1), dx.view(-1,1),stheta.view(-1,1),ctheta.view(-1,1),dtheta.view(-1,1)),dim=-1)
        return state_new
       



    def forward_simulate_2(self, state, action, dt):
        ''' Force control of the cart '''
        # action = torch.clip(action,self.action_min,self.action_min)
        x = state[:,0]
        dx = state[:,1]
        stheta = state[:,2]
        ctheta = state[:,3]
        dtheta = state[:,4]
        
        theta = torch.arctan2(stheta,ctheta) 
        F = action[:,0] # force on cart
        
        a0 = 1/(self.M + self.m*stheta**2)

        ddx = a0*(F - self.b_c*dx - self.b_p*ctheta*dtheta/self.l + self.m*stheta*(self.l*dtheta**2+self.g*ctheta))
        ddtheta = (a0/self.l)*(-ctheta*F + self.b_c*dx*ctheta - self.b_p*dtheta*(self.m+self.M)/(self.m*self.l) +self.b_p*stheta -self.m*self.l*ctheta*stheta*dtheta**2-(self.m+self.M)*self.g*stheta)


        x = x + dx*dt
        dx = dx + ddx*dt
        
        theta = theta + dtheta*dt
        dtheta = dtheta + ddtheta*dt
        
        
        stheta = torch.sin(theta)
        ctheta = torch.cos(theta)
        state_new = torch.concat((x.view(-1,1), dx.view(-1,1),stheta.view(-1,1),ctheta.view(-1,1),dtheta.view(-1,1)),dim=-1)
        return state_new

    def get_Bt(self,state):
        ''' Assuming control affine form: x_dot = A(x) + B(x)u. Return B(x).T given state x'''
        ctheta = state[:,3]
        stheta = state[:,2]
        Bt = torch.zeros(state.shape[0],1,state.shape[-1]).to(self.device)
        if self.force_control:
            a0 = 1/(self.M + self.m*stheta**2)
            Bt[:,:,1] = a0.view(-1,1)
            Bt[:,:,-1] = (-ctheta*a0/self.l).view(-1,1)
        else:    
            Bt[:,:,1] = 1 ## ddx = a
            # ddtheta = (--)*a
            # Bt[:,:,-1] = (-ctheta/(self.m*self.l**2*(4.0/3.0-(self.m*ctheta**2)/(self.m + self.M)))).view(-1,1)
            Bt[:,:,-1] = (-1*ctheta/self.l).view(-1,1)
        return Bt # batch x dim_action x dim_state

    def reward_action(self, action): # related to inverse sigmoid
        return logcos_reward(action, self.action_max, self.w_action)
 
    def reward_state_action(self,state,action):
        theta = torch.arctan2(state[:,2],state[:,3]) 
        d_x = state[:,0]
        dx = state[:,1]

        d_theta = (torch.pi-theta.abs()).abs()  

        r_goal =  -1*(4*(d_theta/torch.pi))**2 - 1*((d_x.abs()/0.5))**3 - 1*(4*dx.abs()/self.state_max[1])**2
        # if self.log_action:
        #     r_action = self.w_action*torch.abs(action)/self.action_max 
        # else: 
        r_action =  logcos_reward(action, self.action_max, self.w_action)
        return (self.w_goal*r_goal + r_action)

#########################################################################################################
#########################################################################################################
#########################################################################################################

class DoublePendulum2:
    '''
        Dynamics of a double pendulum
        state: (theta_1, dtheta_1, theta_2, dtheta_2) # (joinr_angle,joint_vel), joint_angle in (-pi, pi), it is 0 at the stable equilibrium
        action: (tau1,tau2) # joint torque
        ref: http://underactuated.mit.edu/acrobot.html
    '''
    def __init__(self, action_max,
                   fully_actuated=False, 
                   w_goal=0.9, w_action=0.1, 
                   device='cpu'):
        
        self.l1 = 1 # length of the rod
        self.l2 = self.l1
        self.lc1 = self.l1 # com of rod-1
        self.lc2 = self.l2


        self.g = 0*9.81 # gravity
        self.b1= 0. # damping coeficient for joint-1
        self.b2 = 0 # of joint-2

        self.m1 = 1 # mass of the rod-1
        self.m2 = 1 # mass of rod-2

        self.I1 = (self.m1*self.lc1**2) # mass moment of inerta about the pivot-1
        self.I2 = (self.m2*self.lc2**2) # mass moment of inerta about the pivot-2

        self.fully_actuated = fully_actuated

        self.state_max = torch.tensor([1,1, 4*torch.pi, 1, 1, 4*torch.pi]).to(device)
        self.state_min = -self.state_max # (stheta1, ctheta1, dtheta1, stheta2, ctheta2, dtheta2)
        self.action_max = action_max # max  force
        self.action_min = -action_max
        self.w_goal = w_goal
        self.w_action = w_action
        self.device = device

    def get_Bt(self,state):
        ''' Assuming control affine form: x_dot = A(x) + B(x)u. Return B(x).T given state x'''
        s1 = state[:,0]
        c1 = state[:,1]
        s2 = state[:,3]
        c2 = state[:,4]

        detM = (self.I1+self.I2+self.m2*self.l1**2+2*self.m2*self.l1*self.lc2*c2)*self.I2 -\
                (self.I2+self.m2*self.l1*self.lc2*c2)**2
        m11 = self.I2/detM
        m12 = -(self.I2+self.m2*self.l1*self.lc2*c2)/detM
        m21 = 1*m12
        m22 = (self.I1+self.I2+self.m2*self.l1**2+2*self.m2*self.l1*self.lc2*c2)/detM        
         

        if self.fully_actuated:
            Bt = torch.zeros(state.shape[0],2,6).to(self.device)
            Bt[:,0,2] = m11
            Bt[:,1,2] = m12
            Bt[:,0,5] = m21
            Bt[:,1,5] = m22
        else: # acrobot
            Bt = torch.zeros(state.shape[0],1,6).to(self.device)
            Bt[:,0,2] = m12
            Bt[:,0,5] = m22
        return Bt

    def forward_simulate(self, state, action, dt):
        s1 = state[:,0]
        c1 = state[:,1]
        dtheta1 = state[:,2]
        s2 = state[:,3]
        c2 = state[:,4]
        dtheta2 = state[:,5]
        theta1 = torch.arctan2(s1,c1)
        theta2 = torch.arctan2(s2,c2)
        s12 = torch.sin(theta1+theta2)
        
        if self.fully_actuated:
            tau1 = action[:,0] # torque
            tau2 = action[:,1]
        else:
            tau2 = action[:,0]
            tau1 = tau2*0.
        
        detM = (self.I1+self.I2+self.m2*self.l1**2+2*self.m2*self.l1*self.lc2*c2)*self.I2 -\
                (self.I2+self.m2*self.l1*self.lc2*c2)**2
        m11 = self.I2/detM
        m12 = -(self.I2+self.m2*self.l1*self.lc2*c2)/detM
        m21 = 1*m12
        m22 = (self.I1+self.I2+self.m2*self.l1**2+2*self.m2*self.l1*self.lc2*c2)/detM        
         
        C1  = self.b1*dtheta1 - 2*self.m2*self.l1*self.lc2*s2*dtheta1*dtheta2 - self.m2*self.l1*self.lc2*s2*dtheta2**2
        C2  = self.b2*dtheta2 + self.m2*self.l1*self.lc2*s2*dtheta1**2

        t1 = -self.m1*self.g*self.lc1*s1 - self.m2*self.g*(self.l1*s1+self.lc2*s12)
        t2 = -self.m2*self.g*self.lc2*s12

        ddtheta1 = m11*(-C1 + t1 + tau1 ) + m12*(-C2 + t2 + tau2)
        ddtheta2 = m21*(-C1 + t1 + tau1 ) + m22*(-C2 + t2 + tau2)

        
        dtheta1n = dtheta1 + ddtheta1*dt
        dtheta2n = dtheta2 + ddtheta2*dt

        theta1 = theta1 + 0.5*(dtheta1+dtheta1n)*dt
        theta2 = theta2 + 0.5*(dtheta2+dtheta2n)*dt

        
        s1 = torch.sin(theta1); c1 = torch.cos(theta1)
        s2 = torch.sin(theta2); c2 = torch.cos(theta2)

        state_new = torch.concat((s1.view(-1,1),c1.view(-1,1), dtheta1n.view(-1,1), s2.view(-1,1), c2.view(-1,1),dtheta2n.view(-1,1)),dim=-1)
        return state_new

    def reward_action(self, action): # related to inverse sigmoid
        return logcos_reward(action, self.action_max, self.w_action)

    def reward_state_action(self,state,action):
        s1 = state[:,0]
        c1 = state[:,1]
        s2 = state[:,3]
        c2 = state[:,4]
        theta1 = torch.arctan2(s1,c1)
        theta2 = torch.arctan2(s2,c2)
        theta_j1 = (theta1.abs()-torch.pi).abs()
        theta_j2 = (theta2.abs()).abs()
        r_goal = -1*(theta_j1+theta_j2) #-0.01*d_dx#-0.1*d_dtheta - 
        r_action = self.reward_action(action)
        reward = self.w_goal*r_goal + r_action 
        return reward

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
    def __init__(self, action_max=6,
                   fully_actuated=False, 
                   w_goal=0.5, w_action=0.5, 
                   device='cpu'):
        self.l = 0.5 # length of the rod
        self.g = 9.81 # gravity
        self.m = 1.0 # mass of the rod
        self.com = 1*self.l
        self.b = 0.1 # damping coeficient 
        self.cf = 0. # coulomb friction
        self.Ir = 0. # motor inertia
        self.gr = 6 # gear ration
        self.I = (self.m*self.com**2)# self.l**2/3.0 # mass moment of inerta about the pivot

        self.fully_actuated = fully_actuated
        
        self.state_max = torch.tensor([torch.pi,torch.pi, 4*torch.pi, 4*torch.pi]).to(device)
        self.state_min = -self.state_max # (theta1,theta2 dtheta1, dtheta2)
        self.action_max = action_max # max  force
        self.action_min = -action_max
        self.w_goal = w_goal
        self.w_action = w_action

        self.device = device

    def get_Bt(self,state):
        ''' Assuming control affine form: x_dot = A(x) + B(x)u. Return B(x).T given state x'''
        
        invM =  self.get_inv_mass_matrix(state) 
        if self.fully_actuated: # both the joints are activated
            Bt = torch.zeros(state.shape[0],2,4).to(self.device)
            Bt[:,0,2] =invM[:,0,0]
            Bt[:,1,2] = invM[:,0,1]
            Bt[:,0,3] = invM[:,1,0]
            Bt[:,1,3] = invM[:,1,1]
        else: # only second joint is activated
            Bt = torch.zeros(state.shape[0],1,4).to(self.device)
            Bt[:,0,2] = invM[:,0,1]
            Bt[:,0,3] = invM[:,1,1]
        return Bt
    
    def get_inv_mass_matrix(self,state):
        c2 = torch.cos(state[:,1])
        m00 = self.I + self.I + self.m*self.l**2 + 2*self.m*self.l*self.com*c2 + self.gr**2*self.Ir + self.Ir
        m01 = self.I + self.m*self.l*self.com*c2 - self.gr*self.Ir 
        m10 = 1*m01
        m11 = self.I + self.gr**2*self.Ir
        det_M = m00*m11-m01**2
        invM = torch.zeros(state.shape[0],2,2).to(self.device)
        invM[:,0,0] = m11/det_M
        invM[:,1,1] = m00/det_M
        invM[:,0,1] = -1*m01/det_M
        invM[:,1,0] = -1*m01/det_M 
        return invM
    
    def get_coriolis_matrix(self,state):
        s2 = torch.sin(state[:,1])
        dtheta_2 = state[:,3]
        dtheta_1 = state[:,2]
        C = torch.zeros(state.shape[0],2,2).to(self.device)
        C00 = -2*self.m*self.l*self.com*s2*dtheta_2
        C01 = C00/2
        C10 = -1*self.m*self.l*self.com*s2*dtheta_1
        C[:,0,0] = C00
        C[:,0,1] = C01
        C[:,1,0] = C10
        return C
    
    def get_gravity_vector(self,state):
        s1 = torch.sin(state[:,0])
        s12 = torch.sin(state[:,0]+state[:,1])
        G = torch.zeros(state.shape[0],2).to(self.device)
        G0 = -(self.m*self.com + self.m*self.l)*self.g*s1 - self.m*self.com*self.g*s12
        G1 = -self.m*self.com*self.g*s12
        G[:,0] = G0
        G[:,1] = G1
        return G
    
    def get_coulomb_vector(self,state):
        F = torch.zeros(state.shape[0],2)
        return F


    def forward_simulate(self, state, action, dt):
        theta_1 = state[:,0]
        theta_2 = state[:,1]
        dtheta_1 = state[:,2]
        dtheta_2 = state[:,3]
        
        if self.fully_actuated:
            tau_1 = action[:,0] # torque
            tau_2 = action[:,1]
            
        else:
            tau_2 = action[:,0]
            tau_1 = tau_2*0.

        in_torque = torch.concat((tau_1.view(-1,1),tau_2.view(-1,1)),dim=-1)

        invM = self.get_inv_mass_matrix(state)
        G = self.get_gravity_vector(state)
        C = self.get_coriolis_matrix(state)

        force = G + in_torque - torch.einsum('bij,bj->bi',C,state[:,2:])
        ddtheta = torch.einsum('bij,bj ', invM, force)

        dtheta_1_n = dtheta_1 + ddtheta[0]*dt
        dtheta_2_n = dtheta_2 + ddtheta[1]*dt

        theta_1 = theta_1 + dtheta_1*dt#0.5*(dtheta_1+dtheta_1_n)*dt
        theta_2 = theta_2 + dtheta_2*dt#0.5*(dtheta_2+dtheta_2_n)*dt
  
        theta_1 =  (torch.abs(theta_1)<=torch.pi)*theta_1+(theta_1>torch.pi)*(theta_1-2*torch.pi)+(theta_1<-torch.pi)*(2*torch.pi+theta_1) # theta is in range (-pi,pi)
        theta_2 =  (torch.abs(theta_2)<=torch.pi)*theta_2+(theta_2>torch.pi)*(theta_2-2*torch.pi)+(theta_2<-torch.pi)*(2*torch.pi+theta_2) # theta is in range (-pi,pi)

        state_new = torch.concat((theta_1.view(-1,1),theta_2.view(-1,1),  dtheta_1_n.view(-1,1), dtheta_2_n.view(-1,1)),dim=-1)
        return state_new

    def reward_action(self, action): # related to inverse sigmoid
        return logcos_reward(action, self.action_max, self.w_action)

    def reward_state_action(self,state,action):
        theta_err = (state[:,0].abs()-torch.pi).abs() + state[:,1].abs()
        # r_goal = -2-torch.cos(state[:,0])-torch.cos(state[:,0]+state[:,1]) #-1*(theta_j1+theta_j2) #-0.01*d_dx#-0.1*d_dtheta - 
        r_goal = -1*theta_err #-1 + torch.exp(-theta_err)
        r_action =self.reward_action(action)
        reward = self.w_goal*r_goal + r_action 
        return reward
    


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
    def __init__(self, pos_max=1.0, w_goal=1.0, w_action=1.0, n=4, device='cpu'):
        self.n = n 
        self.w_goal = w_goal
        self.w_action = w_action
        self.state_max = torch.tensor([pos_max]*2).to(device)
        self.state_min = -1*self.state_max
        self.device = device

    def forward_simulate(self, state, action, dt):
        x = state[:,0]
        y = state[:,1]
        theta_actuator = (torch.arange(self.n)*torch.pi*2/self.n).view(1,-1).to(self.device)
        idx_switch = torch.arange(self.n).to(self.device)
        switch_on = action[:,2*idx_switch+1].view(-1,self.n)
        v = action[:,idx_switch*2]*switch_on
        dx = (v*torch.cos(theta_actuator)).sum(dim=-1)*dt
        dy = (v*torch.sin(theta_actuator)).sum(dim=-1)*dt
        x = x + dx
        y = y + dy

        state_new = torch.concat((x.view(-1,1),y.view(-1,1)),dim=-1)
        return state_new


    def reward_action(self,action):
        idx_switch = torch.arange(self.n).to(self.device)
        switch_on = action[:,2*idx_switch+1].view(-1,self.n)
        v = action[:,idx_switch*2]
        return -1*self.w_action*(switch_on*v).sum(dim=-1)       

    def reward_state_action(self,state,action):
        d_goal = torch.linalg.norm(state,dim=-1)
        r_goal = -1*(d_goal/0.1)**2
        r_action = self.reward_action(action)
        return self.w_goal*r_goal + r_action


#########################################################################################################
#########################################################################################################
#########################################################################################################

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
#         pass
#     def forward_simulate(self, state, action):
#         pass
#     def reward_state_action(self,state,action):
#         pass
