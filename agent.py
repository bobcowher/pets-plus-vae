import gymnasium as gym
from model import VAE 
import torch
import cv2

class Agent:

    def __init__(self, human=False):
        if(human):
            render_mode = "human"
        else:
            render_mode = "rgb_array"

        self.env = gym.make("CarRacing-v3", render_mode=render_mode, lap_complete_percent=0.95, domain_randomize=False, continuous=False)
        
        self.VAE = VAE(observation_shape=(64, 64, 3))


    def process_observation(self, obs):
        # obs = obs.cpu().numpy()

        obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_NEAREST)

        obs = torch.from_numpy(obs).float().permute(2, 0, 1)

        return obs 


    def train(self, episodes=1):

        for episode in range(episodes):
            done = False
            obs, info = self.env.reset()

            
            while not done:
                action = self.env.action_space.sample()
                obs, reward, done, truncated, info  = self.env.step(action)

                done = done or truncated    
                print(obs)
            
                obs = self.process_observation(obs)
                pred, enc = self.VAE(obs)


