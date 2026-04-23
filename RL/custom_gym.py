import torch
import gym as gym
import numpy as np
from gym import spaces
import sys
sys.path.append('../examples_ttpi')

class CreateGymEnv(gym.Env):
    '''
    The single integrator (order=1) and double integrator system
    '''
    def __init__(self, agent, reset_bound=None, dt=0.01, T=10):
        # T is max episode length in seconds
        super().__init__()
        self.agent = agent
        # Define the state and action spaces
        self.observation_space = spaces.Box(low=self.agent.state_min.cpu().numpy(), high=self.agent.state_max.cpu().numpy(), shape=(self.agent.dim_state,), dtype=np.float32)
        self.action_space = spaces.Box(low=self.agent.action_min.cpu().numpy(), high=self.agent.action_max.cpu().numpy(), shape=(self.agent.dim_action,), dtype=np.float32)
        
        # time step for forward simulation
        self.dt = dt
        self.max_episode_step = int(T/self.dt)
        self.dim_state = self.agent.dim_state

        if reset_bound is None:
            self.reset_bound = np.float32(self.agent.state_max.cpu().numpy())
        else:
            self.reset_bound = np.float32(reset_bound)
        self.reset()
        
        
    def reset(self):
        # Reset the current state and time

        self.current_state = self.reset_bound*np.float32(2*(-0.5+np.random.rand(self.dim_state)))

        self.current_time = 0
        self.current_step_count = 0
        # Return the initial observation
        return self.current_state
    
    def step(self, action):
        
        action = torch.from_numpy(action).view(1,-1)
        # Update the current state using your system dynamics

        current_state = torch.from_numpy(self.current_state).view(1,-1)
        # Calculate the reward for the current state and action
        reward = self.agent.reward_state_action(current_state, action).view(-1).item()
        self.current_state = self.agent.forward_simulate(current_state, action, self.dt).view(-1).numpy()
        
        # Update the current time
        self.current_time += self.dt
        self.current_step_count += 1        
        done = False #True if self.current_step_count>self.max_episode_step else False # check if the episode is done
        
        # Return the observation, reward, done flag, and any additional information
        return self.current_state, reward, done, {}
    
    def render(self, mode='human'):
        # Implement this method if you want to render the environment
        self.agent.plot(self.current_state)
        
    def close(self):
        # Implement this method if you want to close the environment
        pass
