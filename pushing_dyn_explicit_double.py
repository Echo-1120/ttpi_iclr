'''
    explicit dynamics version for planar pushing task, where the
    different contact modes (sticking, sliding up, sliding down, separation and
    face-switching mechanism are formulated in the forward dynamics function as
    conditions. This formulation removes the hard constraints required by the
    implicit version.)
'''
import torch
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
import time, os
import pybullet as p
import pybullet_data
from scipy import integrate
import math
import pdb
torch.set_default_dtype(torch.float64)

# from config import *
"""
Configuration
"""
SLIDER_R = .12 / 2
PUSHER_R = .01/2
PUSHER_X = -SLIDER_R - PUSHER_R
PUSHER_Y = 0
SLIDER_INIT = [0, 0., 0.04]
SLIDER_INIT_THETA = 0
# PUSHER_INIT = [1.3*PUSHER_X, 0, 0]#W.R.T slider-centered frame
PUSHER_ORI = [torch.pi, 0, 0]
TABLE_POS = [SLIDER_INIT[0], 0, 0]
CONTACT_TOL = 1E-3*5
DT = 0.01 # stepping time
T = 500
VEL_BOUND = [[0, 0.05], [-0.05, 0.05]]
ACC_BOUND = [[-1, 1], [-1, 1]]
### boundary
# X_BOUND = [[-0.25, 0.25], [-0.25, 0.25], [-np.pi, np.pi], [-0.25, PUSHER_X],[-SLIDER_R, SLIDER_R]] --> [x, y, theta, px, py]
# U_BOUND = [[0, 0.05], [-0.05, 0.05]] --> Velocity
# CONTACT_FACE = {-1, -2, 1, 2}
"""
Pusher-slider-system
"""
class pusher_slider_sys():
    def __init__(self, p_target, dt=0.01, device="cpu"):
        self.dt = dt
        self.Dx = 6 # 7 #[slider_x, slider_y, slider_theta, pusher_x, pusher_y, pusher_xdot, pusher_ydot]
        self.Du = 2
        self.slider_r = torch.tensor(SLIDER_R).to(device)
        self.pusher_r = torch.tensor(PUSHER_R).to(device)
        self.pusher_hheight = 0.12 #higher height to switch face
        self.contact_tol = torch.tensor(CONTACT_TOL).to(device)  # Tolerance for contact evaluation
        self.px = (-self.slider_r - self.pusher_r).to(device)
        self.u_ps = 0.3  # Friction coefficient between pusher and slider
        self.u_gs = 0.35  # Friction  coefficient between ground and slider
        val, _ = integrate.dblquad(lambda x, y: np.sqrt(x**2 + y**2), 0, self.slider_r, 0, self.slider_r)
        
        self.c = (val / pow(self.slider_r, 2))
        self.device = device
        self.p_target = p_target

    def R_func(self, x):
        R_Matrix = [[torch.cos(x), -torch.sin(x)], [torch.sin(x), torch.cos(x)]]
        return torch.array(R_Matrix)

    def C_func(self, x):
        C_Matrix = [[torch.cos(x), torch.sin(x)], [-torch.sin(x), torch.cos(x)]]
        return torch.array(C_Matrix)

    def gama_t_func(self, px, py):
        u_ps = self.u_ps
        c = self.c
        gama_t = (u_ps * c ** 2 - px * py + u_ps * px ** 2) / (c ** 2 + py ** 2 - u_ps * px * py)
        return gama_t.view(-1,1)

    def gama_b_func(self, px, py):
        u_ps = self.u_ps
        c = self.c
        gama_b = (-u_ps * c ** 2 - px * py - u_ps * px ** 2) / (c ** 2 + py ** 2 + u_ps * px * py)
        return gama_b.view(-1,1)

    def Q_func(self, px, py):
        c = self.c
        Q1_M = [[c ** 2 + px ** 2, px * py], [px * py, c ** 2 + py ** 2]]
        Q2_M = c ** 2 + px ** 2 + py ** 2
        Q_Matrix = Q1_M / Q2_M
        return np.array(Q_Matrix)

    def b1_func(self, px, py):
        bs = len(py)
        out_ = torch.empty(bs,1,2).to(self.device)
        c = self.c
        out_[:,0,0] = -py / (c ** 2 + px ** 2 + py ** 2)
        out_[:,0,1] = px / (c ** 2 + px ** 2 + py ** 2)
        return out_

    def b2_func(self, px, py, gama_t):
        bs = len(py)
        out_ = torch.zeros(bs,1,2).to(self.device)
        c = self.c
        out_[:,0,0]= (-py+ gama_t.squeeze(-1) * px) / (c ** 2 + px ** 2 + py ** 2)
        return out_

    def b3_func(self, px, py, gama_b):
        bs = len(py)
        out_ = torch.zeros(bs,1,2).to(self.device)
        c = self.c
        out_[:,0,0] = (-py + gama_b.squeeze(-1) * px) / (c ** 2 + px ** 2 + py ** 2)
        return out_

    def dynamics(self, xs, us):
        # faceid: -1 (left), -2 (bottom), 1 (right), 2 (top)
        us = torch.cat((us[:,1:],us[:,0].view(-1,1)),dim=-1)
        faceid = us[:,-1].view(-1,1)
        faceid = (faceid-2.0)*(faceid<2)+(faceid-1.0)**(faceid>1)
        s_xy = xs[:,:2]
        s_theta = xs[:,2].view(-1,1)
        p_x = xs[:,3]; p_y = xs[:,4]
        vn = us[:, 0][:, None]
        vt = us[:, 1][:, None]
        u = 1*us[:, :2].double()
        bs = xs.size(0)
        face_beta = (torch.abs(faceid-(-1))*torch.pi/2).to(self.device) # batch x 1
        R_mat = torch.empty(bs,2,2).to(self.device) #np.array(self.R_func(face_beta))
        R_mat[:,0,0] = torch.cos(face_beta).squeeze(dim=-1)
        R_mat[:,-1,-1] = torch.cos(face_beta).squeeze(dim=-1)
        R_mat[:,0,1] = -1*torch.sin(face_beta).squeeze(dim=-1)
        R_mat[:,1,0] = torch.sin(face_beta).squeeze(dim=-1)

        R_theta = torch.empty(bs,2,2).to(self.device) #np.array(self.R_func(face_beta))
        R_theta[:,0,0] = torch.cos(s_theta).squeeze(dim=-1)
        R_theta[:,-1,-1] = torch.cos(s_theta).squeeze(dim=-1)
        R_theta[:,0,1] = -1*torch.sin(s_theta).squeeze(dim=-1)
        R_theta[:,1,0] = torch.sin(s_theta).squeeze(dim=-1)

        Q_mat = torch.empty(bs,2,2).to(self.device) #np.array(self.R_func(face_beta))
        Q_mat[:,0,0] = self.c**2 + p_x**2
        Q_mat[:,-1,-1] = self.c**2 + p_y**2
        Q_mat[:,0,1] = p_x*p_y
        Q_mat[:,1,0] = p_x*p_y
        Q_mat = Q_mat/(self.c**2+p_x**2+p_y**2)[:, None, None]
        """
        Let's assume xs[3:5] and us are defined in left face frame
        """
        # u = R_mat.T.dot(x[5:7]) # Represent u in left face frame
        # u = x[5:7]
        # x[3: 5] = R_mat.T.dot(x[3: 5]) #convert to left face frame
        gama_t = self.gama_t_func(self.px, p_y).to(self.device) # bs x 1
        gama_b = self.gama_b_func(self.px, p_y).to(self.device) # bs x 1 

        D = torch.zeros(bs,1,2).to(self.device)
        P1 = torch.eye(2).to(self.device)[None,:,:].expand(bs,-1,-1) # bs x 2 x 2
        P2 = torch.cat([P1[:, 0:1],torch.cat([gama_t, torch.zeros_like(gama_t)], axis=-1)[:, None]],axis=1)  # bs x 2 x 2
        P3 = torch.cat([P1[:, 0:1], torch.cat([gama_b, torch.zeros_like(gama_b)], axis=-1)[:, None]],axis=1) # bs x 2 x 2

        c1 = torch.zeros(bs,1,2).to(self.device) # bs x 1 x 2
        c2 = torch.cat([-gama_t, torch.ones_like(gama_t)], axis =-1)[:, None, :].to(self.device) # bs x 1 x 2
        c3 = torch.cat([-gama_b, torch.ones_like(gama_b)], axis=-1)[:, None, :].to(self.device) # bs x 1 x 2
        b1 = self.b1_func(self.px,p_y).view(-1,2)[:, None] # bs x 1 x 2
        b2 = self.b2_func(self.px,p_y, gama_t).view(-1,2)[:, None, :] # bs x 1 x 2
        b3 = self.b3_func(self.px,p_y, gama_b).view(-1,2) [:, None, :] # bs x 1 x 2
        # dyn1 = torch.empty(bs,7,2)
        dyn1 = torch.cat([torch.einsum('ijk,ikl,ilm -> ijm', R_theta, Q_mat.double(), P1),
                          b1, D, c1], axis=1) # bs x 5 x 2
        dyn2 = torch.cat([torch.einsum('ijk,ikl,ilm -> ijm', R_theta,Q_mat.double(), P2.double()),
                          b2, D, c2], axis=1) # bs x 5 x 2
        dyn3 = torch.cat([torch.einsum('ijk,ikl,ilm -> ijm', R_theta,Q_mat.double(), P3.double()),
                          b3, D, c3], axis=1) # bs x 5 x 2
        dyn4 = torch.cat([torch.zeros(bs, 3,2).to(self.device), torch.tile(torch.eye(2)[None].to(self.device), (bs, 1,1))], axis=1)

        cond_cone_stick = 1 * torch.logical_and(vt >= gama_b * vn, vt <= gama_t * vn) # bs x 1
        cond_cone_up = 1 * (vt > gama_t * vn) # bs x 1
        cond_cone_down = 1 * (vt < gama_b * vn) # bs x 1

        cond_touch = (torch.abs(p_x - self.px) <= self.contact_tol)[:, None]

        #Assuming vn is always >0, so that the seperation mode is only determined by the distance between pusher and slider
        cond1 = 1 * torch.logical_and(cond_cone_stick, cond_touch)# bs x 1
        cond2 = 1 * torch.logical_and(cond_cone_up, cond_touch)# bs x 1
        cond3 = 1 * torch.logical_and(cond_cone_down, cond_touch)# bs x 1
        cond4 = 1 - cond1 - cond2 - cond3# bs x 1

        
        # f is the system state w.r.t. global frame, but assuming pushing the initial left face
        f = cond1 * torch.einsum('ijk,ikl -> ijl', dyn1, u[:,:, None])[..., 0] # bs x 5
        f += cond2 * torch.einsum('ijk,ikl -> ijl', dyn2.double(), u[:,:, None])[..., 0] # bs x 5
        f += cond3 * torch.einsum('ijk,ikl -> ijl', dyn3.double(), u[:,:, None])[..., 0] # bs x 5
        f += cond4 * torch.einsum('ijk,ikl -> ijl', dyn4.double(), u[:,:, None])[..., 0] # bs x 5

        # transform the state obtained by assuming left face to current face
        f[:, :2][:, :, None] = torch.einsum('ijk,ikl -> ijl', R_mat, f[:, :2][:, :, None])
            
        # vn += us[:, 0][:, None] * self.dt
        # vt += us[:, 1][:, None] * self.dt
        # print("shape1", (xs[:, :5]+f*self.dt).shape)
        # print("shape2", xs[:, -1][:, None].shape)
        # return torch.cat(((xs[:, :5]+f*self.dt), xs[:, -1][:, None]), dim=1).to(self.device)
        return torch.cat(((xs[:, :5]+f*self.dt), us[:, -1][:, None]), dim=1).to(self.device).double()

    # def cost_func(self, xs, us, tol=0.025,  w_face=1, scale=0.25):
    #     pos_error = torch.linalg.norm(xs[:,:2]-self.p_target[:,:2], dim=-1)/(tol*scale)

    #     ori_error = (xs[:,2]-self.p_target[:,2]).abs()/tol
    #     ori_error=(ori_error)*torch.exp(-(pos_error**2))
    #     # ori_error = (xs[:,2]-self.p_target[:,2]).abs()/(1*torch.pi)
        
    #     fs_error = 1.0*(xs[:, -1] != us[:, -1])
    #     vel_error = torch.linalg.norm(us[:, :2], dim=-1)
    #     error = ori_error**2 + pos_error**2 + w_face*fs_error + vel_error*0.01

    #     # # refine orientation error
    #     # ext_error = torch.cat([(error[:,2]%(2*torch.pi))[:, None], (error[:,2]%(2*torch.pi)-2* torch.pi)[:, None], (error[:,2]%(2*torch.pi)+2*torch.pi)[:, None]], dim=1)
    #     # error[:, 2], _ = torch.max(ext_error, 1)
    #     return error
    

    def cost_func(self, xs, us, tol=0.01, scale=0.3):
        consider_next_state = True
        if consider_next_state: # the cost function is more informative w.r.t the action
            next_state = self.dynamics(xs,us)
            pos_error_1 = torch.linalg.norm(xs[:,:2]-self.p_target[:,:2], dim=-1)
            ori_error_1 = (xs[:,2]-self.p_target[:,2]).abs()
            pos_error_2 = torch.linalg.norm(next_state[:,:2]-self.p_target[:,:2], dim=-1)
            ori_error_2 = (next_state[:,2]-self.p_target[:,2]).abs()
            pos_error =  0.5*(pos_error_2 + pos_error_1)/(scale*tol)
            ori_error = 0.5*(ori_error_2 + ori_error_1)/(tol*torch.pi)
        else:
            pos_error = torch.linalg.norm(xs[:,:2]-self.p_target[:,:2], dim=-1)/(tol*scale)
            ori_error = (xs[:,2]-self.p_target[:,2]).abs()/(tol*torch.pi)

        cost_state = 0.5*pos_error + 0.5*ori_error
        cost_face_switch = (xs[:, -1] != us[:, 0])
        cost_vel = torch.linalg.norm(us[:, 1:], dim=-1)
        cost = cost_state+cost_face_switch #*(1 + 0.01*cost_face_switch + 0.001*cost_vel)
        # fs_error = (xs[:, -1] != us[:, -1])*1/3*0.1*0.1
        # vel_error = torch.linalg.norm(us[:, :2], dim=-1)*0.01
        # error = pos_error #+ fs_error + vel_error

        # # refine orientation error
        # ext_error = torch.cat([(error[:,2]%(2*torch.pi))[:, None], (error[:,2]%(2*torch.pi)-2* torch.pi)[:, None], (error[:,2]%(2*torch.pi)+2*torch.pi)[:, None]], dim=1)
        # error[:, 2], _ = torch.max(ext_error, 1)
        return cost#1-torch.exp(-4**(pos_error/scale)) + 1-torch.exp(-1**(pos_error/0.03)**2)

    def cost_func_v2(self, xs, us, tol=0.03, scale=0.01):
        pos_error = torch.linalg.norm(xs[:,:2]-self.p_target[:,:2], dim=-1)#/tol

        ori_error = (xs[:,2]-self.p_target[:,2]).abs()/(1*torch.pi)
        # ori_error=ori_error*torch.exp(-(pos_error*scale/tol)**1/3)
        
        fs_error = (xs[:, -1] != us[:, -1])*1/3*0.4
        vel_error = torch.linalg.norm(us[:, :2], dim=-1)*0.01
        error = pos_error#+ pos_error + fs_error + vel_error

        # # refine orientation error
        # ext_error = torch.cat([(error[:,2]%(2*torch.pi))[:, None], (error[:,2]%(2*torch.pi)-2* torch.pi)[:, None], (error[:,2]%(2*torch.pi)+2*torch.pi)[:, None]], dim=1)
        # error[:, 2], _ = torch.max(ext_error, 1)
        return pos_error#**2

    def pybullet_init(self, p_target):
        # %% Start Pybullet
        physicsClient = p.connect(p.GUI)  # pybullet with visualisation
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.resetDebugVisualizerCamera(cameraDistance=0.5, cameraYaw=30, cameraPitch=-40,
                                     cameraTargetPosition=[0.5, 0, 0.5])
        p.resetSimulation()
        p.setTimeStep(1 / 1000)
        p.setGravity(0, 0, -9.8)
        """start a real time simulation, so no simulation steps are needed"""
        p.setRealTimeSimulation(0)

        plane_id = p.loadURDF('plane.urdf')
        table_id = p.loadURDF('/mesh/urdf/table/table.urdf', basePosition=TABLE_POS,
                              baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, SLIDER_INIT_THETA]),
                              useFixedBase=1, flags=p.URDF_USE_SELF_COLLISION)

        # %%% Construct the robot system
        # object-centered reference frame
        object_path = 'mesh/urdf/cube/cube.urdf'

        # initialization
        
        slider_init = SLIDER_INIT  # slider initial position
        # ee_pos = np.array(slider_init) + np.array(PUSHER_INIT)
        # ee_orn = p.getQuaternionFromEuler([0, 0, 0])

        self.sliderId = p.loadURDF(
            fileName=object_path, basePosition=slider_init,
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, SLIDER_INIT_THETA]), useFixedBase=0,
            flags=p.URDF_USE_SELF_COLLISION
        )

        # global_target = robot_sys.scf2rbf(p_target)[:3]
        global_target = p_target
        targetId = p.loadURDF(
            fileName='mesh/urdf/cube/target.urdf', basePosition=np.hstack((global_target[:2], SLIDER_INIT[2])),
            baseOrientation=p.getQuaternionFromEuler([0.0, 0.0, global_target[2]]), useFixedBase=1,
            flags=p.URDF_USE_SELF_COLLISION
        )

    def vis_traj(self, x):
        #visualization of current slider state
        slider_orn = p.getQuaternionFromEuler([0, 0, x[2]])
        p.resetBasePositionAndOrientation(self.sliderId, np.hstack((x[0:2], SLIDER_INIT[2])), slider_orn)

if __name__ == "__main__":

    # target
    p_target = torch.tensor([0., 0., 0.]).view(1,-1)

    # system definition
    ps_sys = pusher_slider_sys(p_target=p_target)
    sys_dx = ps_sys.Dx #state dimension
    sys_du = ps_sys.Du #control dimension

    #boundary
    state_max = torch.tensor([0.25, 0.25, np.pi, -0.065, 0.06])
    state_min = torch.tensor([-0.25, -0.25, -np.pi, -0.065, -0.06])

    vel_max = torch.tensor([0.05, 0.05])
    vel_min = torch.tensor([0, -0.05])

    face_id = torch.tensor([-2, -1, 1, 2])

    # # system initialization
    # slider_state_init = np.hstack([SLIDER_INIT[:2], SLIDER_INIT_THETA])
    # pusher_pos_init = np.array([PUSHER_X, 0])  # Initial state: pusher and slider are with contact
    # pusher_vel_init = np.array([0, 0])
    # x0 = np.hstack((slider_state_init, pusher_pos_init, pusher_vel_init))
    # xs = np.tile(np.hstack([slider_state_init, pusher_pos_init, pusher_vel_init]), (T+1, 1))
    # us = np.tile(np.array([0.2, -0.1]), (T, 1))

    # batch formulation
    n_steps = 30 #batch numbers
    # system state: [slider_x, slider_y, slider_theta, pusher_x, pusher_y, pusher_xdot, pusher_ydot]
    cur_state = (state_max - state_min) * torch.rand(n_steps, 5) + state_min # bs x 7
    # control input: {batch_us}-->pusher acceleration, {batch_faces}--> contact face
    cur_action = (vel_max - vel_min) * torch.rand(n_steps, 2) + vel_min # bs x 2
    cur_action = torch.cat([cur_action, face_id[np.random.choice(range(4), n_steps)][:, None]], axis=1) # bs x 3

    # batch_xs = torch.tile(torch.tensor(x0), (n_steps, 1))
    #
    # # control input: {batch_us}-->pusher acceleration, {batch_faces}--> contact face
    # batch_vel = torch.rand(n_steps, 2) # control command
    # batch_faces = torch.ones(n_steps) * -1 # faceid: -1 (left), -2 (bottom), 1 (right), 2 (top)
    # batch_us = torch.cat([batch_vel, batch_faces[:, None]], axis=-1)

    ### This is the forward_simulate function
    # batch_xnext = ps_sys.batch_step(batch_xs, batch_us, batch_faces)
    batch_xnext = ps_sys.dynamics(cur_state, cur_action)
    batch_cost = ps_sys.cost_func(cur_state, cur_action)
    print(batch_xnext)

    #Pybullet visualization
    # ps_sys.pybullet_init(p_target)
    # ps_sys.vis_traj(batch_xs[0])
    # time.sleep(15)

    # ### Rollout test
    # for i in range(T):
    #     xs[i+1] = ps_sys.step(xs[i], us[i], confaceid=-1)
    #     print(xs[i+1])


