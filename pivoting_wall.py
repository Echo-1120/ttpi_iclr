'''
    Dynamics for flipping a box against the wall
'''
import torch

import matplotlib.pyplot as plt
from IPython.display import clear_output
import time, os
from scipy import integrate
import math
from celluloid import Camera
import torch.nn.functional as F
torch.set_default_dtype(torch.float64)
import numpy as np

"""
pivoting_sys
"""
class pwall_env():
    def __init__(self,state_min, state_max, dt=0.01,w_goal=100, w_action=0.1, device = 'cpu'):
        #Block size: [0.12, 0.12, 0.04]
        self.device = device
        self.pusher_radius = torch.tensor(0.002).to(self.device) #pusher radius
        self.l1 = torch.tensor(0.06).to(self.device) #half length
        self.l2 = torch.tensor(0.02).to(self.device)  # half height
        self.mass = torch.tensor(0.16).to(self.device)
        self.I = (8 / 3 * self.mass * self.l1 * self.l2).to(self.device)  # momnet of inertia
        self.g = torch.tensor(9.81).to(self.device)
        self.u_ps = torch.tensor(0.3).to(self.device)
        self.dt = torch.tensor(dt).to(self.device)
        self.u1 = 0.3 #friction coefficient between object and the ground
        self.u2 = 0.3 #friction coefficient between object and wall

        self.state_min = state_min
        self.state_max = state_max

        self.w_action = w_action
        self.w_goal = w_goal


    def R_func(self, x):
        R_Matrix = [[np.cos(x), -np.sin(x)], [np.sin(x), np.cos(x)]]
        return np.array(R_Matrix)

    def forward_simulate(self,state,action,dt):
        next_state = self.aug_dynamics(state,action,dt)
        return torch.clip(next_state,self.state_min,self.state_max)
    
    def reward_state_action(self,state,action):
        return -1*self.aug_cost_func(state,action)
    
    def reward_action(self,action):
        return -1*self.cost_action(action)
    
    def cost_action(self,action):
        return self.w_action*torch.linalg.norm(action[:, :2], dim=-1)
    
    def cost_func(self, xs, action, p_target):
        #    xs: [x, y, theta, py, x_dot, y_dot, theta_dot, py_dot]
        #    action:[f_1, f_2, v_y]
        # pos_error = torch.linalg.norm(xs[:,:2]-p_target[:,:2], dim=-1)/(self.l1-self.l2)
        ori_error = (xs[:,2]-p_target[:,2]).abs() #/(torch.pi/2)
        # vel_pos_error = torch.linalg.norm(xs[:,4:6]-p_target[:,4:6], dim=-1)/3#/(self.state_domain[1][4]*2)
        vel_ori_error = (xs[:,6]-p_target[:,6]).abs()/3#/(self.state_domain[1][6]*2)

        error = ori_error #+ vel_ori_error

        return error
    
    def aug_cost_func(self, xs, action):
        #    xs: [theta, theta_dot, py, des_theta]
        #    action:[f_1, f_2, v_y]
        e_ori = xs[:,0]-xs[:,-1]
        ori_error = (e_ori/0.1)**2  #/(torch.pi/2)
        action_error = self.cost_action(action)#/(self.state_domain[1][4]*2)

        error = self.w_goal*ori_error + action_error

        return error
    
    def aug_dynamics(self, state, u, dt):
        """
        In this dynamics, the desired orientation is used as the last element, then cost_func is designed as the error between state[2] and state[-1]

        :param stata: state [theta, theta_dot, py, des_theta] // [theta, theta_dot] is defined in global
            frame {g}, [py] is defined in object frame {o}
        :param u: input [f_1. f_2, v_y], defined in object frame {o}. f_1 is the normal force, and f_2 the tangential
            force. v_y is the magnitude of tengential velocity.
        :return: stata_next

        """
        #state: bs x 4
        theta = state[:, 0][:, None].to(self.device) # bs x 1
        theta_dot = state[:, 1][:, None].to(self.device) # bs x 1
        py = state[:, 2][:, None].to(self.device) # bs x 1
        des_theta = state[:, 3][:, None].to(self.device) # bs x 1

        # input: bs x 3
        f_1 = u[:, 0][:, None].to(self.device) # bs x 1
        f_2 = u[:, 1][:, None].to(self.device)# bs x 1o
        f_2 = torch.clamp(f_2, -self.u_ps * f_1, self.u_ps*f_1)
        v_y = u[:, 2][:, None].to(self.device)# bs x 1

        #LS constraints
        # x = np.cos(theta) * self.l1 + np.sin(theta)* self.l2 - self.l1
        # y = np.sin(theta)*self.l1 + np.cos(theta)*self.l2 - self.l2

        tor_max = self.mass*self.g * self.l1
        tor_gravity = self.mass * self.g * torch.sqrt(self.l1**2 + self.l2**2)* \
                  torch.cos(theta + torch.arctan(self.l2/self.l1)) # bs x 1

        tor_vec = torch.cat([torch.ones_like(py)* self.l1*2, py], axis=1) #bs x 2 #torch.tensor([self.l1, py]) # bs x 2
        tor_vec = F.pad(tor_vec, (0, 1)) #pad the last dim to have 3D vector
        tor_mag = F.pad(torch.cat([-f_1, f_2], axis=1), (0, 1))
        torque = torch.cross(tor_vec, tor_mag, dim=1)[:, 2] - tor_gravity[:, 0]

        theta_ddot = torch.div(torque, self.I)[:, None]
        # theta_dot = theta_ddot*self.dt 

        
        x_dot = -torch.sin(theta) * theta_dot *self.l1 + torch.cos(theta) * theta_dot * self.l2
        y_dot = torch.cos(theta)* theta_dot * self.l1 - torch.sin(theta) * theta_dot* self.l2

        ### Finger modes
        cond_stick = 1* (torch.abs(f_2) <= torch.abs(self.u_ps*f_1)) # bs x 1
        cond_slide = 1 - cond_stick # bs x 1

        # if torch.abs(f_2) <= torch.abs(self.u_ps*f_1): # Sticking
        #     py_dot = torch.zeros_like(py)
        # else:
        #     py_dot = v_y*torch.sign(f_2)

        py_dot = cond_stick * torch.zeros_like(py) + cond_slide * v_y*torch.sign(f_2)

        state_dot = torch.cat([theta_dot, theta_ddot, py_dot], axis=1)

        next_state = state[:, :3] + state_dot*dt
        next_state = next_state.clamp(self.state_min[:3], self.state_max[:3])

        # next_state[:, 0] = torch.cos(next_state[:, 2]) * self.l1 + torch.sin(next_state[:, 2]) * self.l2 - self.l1
        # next_state[:, 1] = torch.sin(next_state[:, 2]) * self.l1 + torch.cos(next_state[:, 2]) * self.l2 - self.l2


        next_state = torch.cat([next_state, des_theta], dim=1)


        return next_state # bs x 3


    def dynamics(self, state, u):
        """
        In this dynamics, Limit Surface (LS) is used, inspired by planar pushing model.

        :param stata: state [x, y, theta, py] // [x, y, theta] is defined in global
            frame {g}, [py] is defined in object frame {o}
        :param u: input [f_1. f_2, v_y], defined in object frame {o}. f_1 is the normal force, and f_2 the tangential
            force. v_y is the magnitude of tengential velocity.
        :return: stata_next

        """
        #state: bs x 4
        x = state[:, 0][:, None].to(self.device) # bs x 1
        y = state[:, 1][:, None].to(self.device) # bs x 1
        theta = state[:, 2][:, None].to(self.device) # bs x 1
        py = state[:, 3][:, None].to(self.device) # bs x 1

        # input: bs x 3
        f_1 = u[:, 0][:, None].to(self.device) # bs x 1
        f_2 = u[:, 1][:, None].to(self.device)# bs x 1
        v_y = u[:, 2][:, None].to(self.device)# bs x 1
        f_2 = f_2.clamp(-self.u_ps * f_1, self.u_ps*f_1)

        #LS constraints
        # x = np.cos(theta) * self.l1 + np.sin(theta)* self.l2 - self.l1
        # y = np.sin(theta)*self.l1 + np.cos(theta)*self.l2 - self.l2

        tor_max = self.mass*self.g * self.l1
        tor_gravity = self.mass * self.g * torch.sqrt(self.l1**2 + self.l2**2)* \
                  torch.cos(theta + torch.arctan(self.l2/self.l1)) # bs x 1

        tor_vec = torch.cat([torch.ones_like(py)* self.l1*2, py], axis=1) #bs x 2 #torch.tensor([self.l1, py]) # bs x 2
        tor_vec = F.pad(tor_vec, (0, 1)) #pad the last dim to have 3D vector
        tor_mag = F.pad(torch.cat([-f_1, f_2], axis=1), (0, 1))
        torque = torch.cross(tor_vec, tor_mag, dim=1)[:, 2] - tor_gravity[:, 0]

        theta_ddot = torch.div(torque, self.I)[:, None]
        theta_dot = theta_ddot*self.dt 


        # torque = torch.einsum('ti, ti -> ti', tor_vec, torch.cat([-f_1, f_2], axis=1)) #bs x 2 #torch.cross(tor_vec, torch.tensor([-f_1, f_2]))
        # theta_dot = torch.pow(tor_max, -2) * torque[:, None]
        
        x_dot = -torch.sin(theta) * theta_dot *self.l1 + torch.cos(theta) * theta_dot * self.l2
        y_dot = torch.cos(theta)* theta_dot * self.l1 - torch.sin(theta) * theta_dot* self.l2

        ### Finger modes
        cond_stick = 1* (torch.abs(f_2) <= torch.abs(self.u_ps*f_1)) # bs x 1
        cond_slide = 1 - cond_stick # bs x 1

        # if torch.abs(f_2) <= torch.abs(self.u_ps*f_1): # Sticking
        #     py_dot = torch.zeros_like(py)
        # else:
        #     py_dot = v_y*torch.sign(f_2)

        py_dot = cond_stick * torch.zeros_like(py) + cond_slide * v_y*torch.sign(f_2)

        state_dot = torch.cat([x_dot, y_dot, theta_dot, py_dot], axis=1)

        next_state = state[:, :4] + state_dot*self.dt
        next_state = next_state.clamp(self.state_domain[0][:4], self.state_domain[1][:4])

        next_state[:, 0] = torch.cos(next_state[:, 2]) * self.l1 + torch.sin(next_state[:, 2]) * self.l2 - self.l1
        next_state[:, 1] = torch.sin(next_state[:, 2]) * self.l1 + torch.cos(next_state[:, 2]) * self.l2 - self.l2

        next_state = torch.cat([next_state, torch.cat([x_dot, y_dot, theta_dot, py_dot], dim=1)], dim=1)
        # next_state = next_state.clamp(self.state_domain[0], self.state_domain[1])


        return next_state # bs x 8


    def plot_pivot(self, sys_x):
        '''
        :param sys_x: state: [x, y, theta, px, py, x_dot, y_dot, theta_dot]
        :return:
        '''

        # %%
        sys_x = np.array(sys_x)

        cx = np.cos(sys_x[:, 0]) * self.l1 + np.sin(sys_x[:, 0]) * self.l2 - self.l1
        cy = np.sin(sys_x[:, 0]) * self.l1 + np.cos(sys_x[:, 0]) * self.l2 - self.l2
        theta = sys_x[:, 0]
        tehta_dot = sys_x[:, 1]
        py = sys_x[:, 2]
        px = np.ones_like(py) * self.l1.cpu().numpy()

        R = self.R_func(theta).transpose(2, 0, 1)  # Tx2x2
        # slider plot
        rec = np.array(
            [[-self.l1, -self.l1, self.l1, self.l1, -self.l1], [-self.l2, self.l2, self.l2, -self.l2, -self.l2]])
        # %%
        msh = np.einsum('tij,jl -> til', R, rec) + np.repeat(np.stack([cx, cy], axis=1)[:, :, None], 5, axis=2)  # Tx2x5
        plt_msh = msh.transpose(1, 2, 0)  # 2x5xT
        # %%
        pxy = np.vstack([px, py]).transpose(1, 0)
        pusher_xys_rel = np.einsum('til, tlj -> tij', R, pxy[:, :, None])  # Tx2x1
        pusher_xys = pusher_xys_rel + torch.cat([cx, cy], dim=1)[:, :, None]  # Tx2x1
        # %%
        fig = plt.figure(edgecolor=[0.1, 0.1, 0.1])
        fig.set_size_inches(8, 8)

        # fig.patch.set_facecolor('white')
        # fig.patch.set_alpha(0.9)
        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-0.2, 0.1), ylim=(-0.1, 0.2))

        plt.plot(plt_msh[0, :, ::1], plt_msh[1, :, ::1], color='k', alpha=0.5)  ##plt_msh: 2x5xT
        l1, = plt.plot(cx, cy, color='darkblue', linewidth=5, label='center')
        # ax.plot(plt_msh[0,0:2, -1], plt_msh[1,0:2, -1], 'black', linewidth=3)
        l2 = plt.scatter(pusher_xys[:, 0, 0], pusher_xys[:, 1, 0], c='firebrick', edgecolors='firebrick', marker='.',
                         s=1000, alpha=0.4)
        # # wall1
        # plt.plot([-self.l1, -self.l1], [-self.l2, 1], color='k', linewidth=5)

        # # wall2
        # plt.plot([-self.l1, 1], [-self.l2, -self.l2], color='k', linewidth=5)

        plt.legend((l1, l2), ('slider', 'pusher'), loc='lower right')
        plt.grid("True")
        plt.xlabel('x(cm)')
        plt.ylabel('y(cm)')
        plt.show()

    def animate_pivot(self, sys_x, sys_u, step_skip=1, dt=0.01, xmax=1, x_obst=[], r_obst=[], batch=False,
                      title=None,
                      save_as=None, figsize=3, scale=0, slider_r=0.06, animation=False, file_name='pivoting'):
        # x_t: (x,y,theta, py, x_dot, y_dot, theta_dot, py_dot)
        fig = plt.figure(edgecolor=[0.1, 0.1, 0.1])
        fig.set_size_inches(figsize, figsize)

        ax = fig.add_subplot(111, aspect='equal', autoscale_on=False,
                             xlim=(-0.06, 0.10), ylim=(-0.02, 0.14))

        if not x_obst is None:
            for i, x in enumerate(x_obst):
                circ = plt.Circle(x, r_obst[i], color='grey', alpha=0.5)
                ax.add_patch(circ)

        # color_list = ['y', 'g', 'b', 'm', 'orange', 'r', 'k', 'c', 'bisque', 'blueviolet', 'brown', 'darkblue', ]
        import matplotlib.colors as colors
        color_list = list(colors._colors_full_map.values())
        print()

        l1 = self.l1.cpu().numpy()
        l2 = self.l2.cpu().numpy()



        theta = sys_x[:, :, 0]
        cx = np.cos(theta) * l1 + np.sin(theta) * l2 - l1
        cy = np.sin(theta) * l1 + np.cos(theta) * l2 - l2
        theta_dot = sys_x[:, :, 1]
        py = sys_x[:, :, 2]
        px = np.ones_like(py) * (l1+self.pusher_radius.cpu().numpy())

        R = self.R_func(theta).transpose(2, 3, 0, 1)  # bsxTx2x2

        # force
        # f_12 = sys_u[:, , 0:2]  # T X 2
        # f_1_vec = np.zeros_like(f_12) + np.array([-1, 0])
        # f_2_vec = np.zeros_like(f_12) + np.array([0, 1])
        # f_1_global = np.einsum('tij,tjl -> til', R[1:], f_1_vec[:, :, None])  # T X 2 X 1
        # f_2_global = np.einsum('tij,tjl -> til', R[1:], f_2_vec[:, :, None])

        # slider plot
        rec = np.array(
            [[-l1, -l1, l1, l1, -l1], [-l2, l2, l2, -l2, -l2]])
        # %%
        msh = np.einsum('ktij,jl -> ktil', R, rec) + np.repeat(np.stack([cx, cy], axis=2)[:, :, :, None], 5,
                                                                axis=3)  # bsxTx2x5
        plt_msh = msh.transpose(2, 3, 0, 1)  # 2x5xbsxT
        # %%
        pxy = np.stack([px[:, :, None], py[:, :, None]], axis=2) #bs x T x 2 x1
        pusher_xys_rel = np.einsum('ktil, ktlj -> ktij', R, pxy)  # bsxTx2x1
        print("cx_dim", cx.shape)
        pusher_xys = pusher_xys_rel + np.concatenate([cx[:, :, None], cy[:, :, None]], axis=2)[:, :, :, None]  # bsxTx2x1

        # wall1
        plt.plot([-l1, -l1], [-l2, 1], color='k', linewidth=2)

        # wall2
        plt.plot([-l1, 1], [-l2, -l2], color='k', linewidth=2)

        camera = Camera(fig)
        if animation:
            dt = dt * step_skip
            interval = (1 / dt)  # *10**-3 # in ms

        T = plt_msh.shape[3]
        if animation:  # for loop takes long time
            for bt in range(len(plt_msh[0][0])):
                for i in range(0, T, step_skip):
                    # wall1
                    plt.plot([-l1, -l1], [-l2, 1], color='k', linewidth=2)

                    # wall2
                    plt.plot([-l1, 1], [-l2, -l2], color='k', linewidth=2)
                    # print("plot_step", i)
                    alpha_ = np.clip(0.1 + i * 1.0 / T, 0, 1) if (not animation) else 1.0
                    # plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], color=((random.random(),random.random(),random.random(), 1)))
                    plt.plot(plt_msh[0, :, bt, i], plt_msh[1, :, bt, i], color=color_list[bt], alpha=alpha_)
                    # plt.plot(plt_msh[0, 0:2, bt, i], plt_msh[1, 0:2, bt, i], 'black', linewidth=3, alpha=alpha_)
                    ax.scatter(pusher_xys[bt, i, 0, 0], pusher_xys[bt, i, 1, 0], c=color_list[bt], edgecolors='r',
                               marker='o', s=50, alpha=alpha_)
                    if i < T - 1:
                        # finger force
                        plot_scale = 0.03
                        ft_force = sys_u[bt, i, 1].clip(-self.u_ps.cpu().numpy() * sys_u[bt, i, 0], self.u_ps.cpu().numpy() * sys_u[bt, i, 0])

                        quil_1 = sys_u[bt, i, 0] * plot_scale  # quiver length
                        quil_2 = ft_force * plot_scale  # quiver length
                        ax.quiver(pusher_xys[bt, i, 0, 0], pusher_xys[bt, i, 1, 0], -quil_1 * np.cos(theta[bt, i]),
                                  -quil_1 * np.sin(theta[bt, i]), color='r',
                                  scale=0.5, width=0.01)
                        ax.quiver(pusher_xys[bt, i, 0, 0], pusher_xys[bt, i, 1, 0], quil_2 * np.cos(theta[bt, i] + np.pi / 2),
                                  quil_2 * np.sin(theta[bt, i] + np.pi / 2), color='r',
                                  scale=0.5, width=0.01)
                        
                        # friction force
                        
                        ## ground force
                        f1n = self.mass.cpu().numpy() * self.g.cpu().numpy() + sys_u[bt, i, 0]*np.sin(theta[bt, i]) - ft_force*np.cos(theta[bt, i])
                        f1t = self.u1*f1n
                        
                        ## wall force
                        f2n = sys_u[bt, i, 0]*np.cos(theta[bt, i]) + ft_force*np.sin(theta[bt, i])
                        f2t = self.u2*f2n

                        #plt_msh: 2x5xbsxT

                        ax.quiver(plt_msh[0, 0, bt, i], plt_msh[1, 0, bt, i], -f1t* plot_scale,
                                  f1n* plot_scale, color='green',
                                  scale=0.5, width=0.01)
                        
                        # ax.quiver(plt_msh[0, 0, bt, i], plt_msh[1, 0, bt, i], 0,
                        #           f1n* plot_scale, color='yellow',
                        #           scale=0.5, width=0.01)

                        ax.quiver(plt_msh[0, 1, bt, i], plt_msh[1, 1, bt, i], f2n* plot_scale,
                                f2t* plot_scale, color='green',
                                scale=0.5, width=0.01)
                        
                        # ax.quiver(plt_msh[0, 1, bt, i], plt_msh[1, 1, bt, i], 0,
                        #           f2t* plot_scale, color='yellow',
                        #           scale=0.5, width=0.01)
                       


                    camera.snap()
        else:
            for bt in range(len(plt_msh[0][0])):
                plt.plot(plt_msh[0,:,bt,::step_skip], plt_msh[1,:,bt, ::step_skip], linewidth=2, color='k', alpha=0.2)
                plt.plot(cx[bt], cy[bt], color='darkblue', linewidth=1, label='center')
                plt.scatter(pusher_xys[bt, ::step_skip, 0, 0], pusher_xys[bt, ::step_skip, 1, 0], c='r', edgecolors='r', marker='o', s=30)
                plt.fill(plt_msh[0,:,bt,0], plt_msh[1,:,bt, 0], facecolor='orange', alpha=0.5) #initialization
                plt.fill(plt_msh[0,:,bt,-1], plt_msh[1,:,bt, -1], facecolor='green', alpha=0.5) #Target

        if animation:
            # animation = camera.animate(interval=interval, repeat=False)
            animation = camera.animate()
            animation.save(file_name+'.mp4')

        # ax.legend(["target","init","obstacle"])
        plt.grid("True")

        if not title is None:
            plt.title(title)
        if not save_as is None:
            fig.savefig(save_as + ".jpeg", bbox_inches='tight', pad_inches=0.01, dpi=300)

        return plt



if __name__ == "__main__":
    pwall_sys = pwall_env()
    n_state = 8 #[x, y, theta, py, x_dot, y_dot, theta_dot, py_dot]
    n_input = 3 #[f_1, f_2, v_y]
    bs = 50 #batch_size
    # cur_state = np.random.rand(n_state)*(pivot_sys.state_domain[1]-pivot_sys.state_domain[0]) + pivot_sys.state_domain[0]
    cur_state = torch.zeros(bs, n_state) # bs x 8
    action = torch.zeros(bs, n_input) # bs x 3
    # input = np.zeros(n_input)
    states = [cur_state]
    actions = []
    traj = cur_state[:, :].clone()[:, None, :]  # bsx(T+1)x8
    traj_actions = torch.empty(cur_state.shape[0], pwall_sys.T, 3).to(pwall_sys.device)  # bsxTx3

    for i in range(100):
    # input[0:3]=[10,0, 10]
    #     input = np.random.rand(n_input)*(pwall_sys.input_domain[1]-pwall_sys.input_domain[0]) + pwall_sys.input_domain[0]
        action[:, 0:3] = torch.tensor([3, 0.5, 0.5]).repeat(bs, 1) #bs x 3
        # print('input', input)
        next_state = pwall_sys.dynamics(cur_state, action)
        cur_state = next_state.clone()
        traj = torch.concat((traj, cur_state[:, None, :]), dim=1)
        traj_actions[:, i, :] = action

        # pivot_sys.plot_pivot(np.array(states))
        if torch.any(cur_state[:, 2] == torch.pi / 2):
            print("final step", i)
            break

        # print(next_state)

    print('finished!')
    import numpy as np
    pwall_sys.plot_pivot(traj[0].numpy())
    # plt = pwall_sys.animate_pivot(traj[-10:].numpy(), traj_actions[-10:].numpy(), np.array([-pwall_sys.l2, pwall_sys.l1, np.pi/2]), animation=False)
    # plt.show()
