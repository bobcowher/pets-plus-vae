import gymnasium as gym
from model import VAE 
import torch
from buffer import ReplayBuffer
import cv2

class Agent:

    def __init__(self, human=False, max_buffer_size=100000):
        if(human):
            render_mode = "human"
        else:
            render_mode = "rgb_array"

        self.env = gym.make("CarRacing-v3", render_mode=render_mode, lap_complete_percent=0.95, domain_randomize=False, continuous=False)

        obs, info = self.env.reset()
        obs = self.process_observation(obs)
        
        self.VAE = VAE(observation_shape=obs.shape)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        self.memory = ReplayBuffer(max_size=max_buffer_size, input_shape=obs.shape, n_actions=self.env.action_space.n, input_device=self.device, output_device=self.device)


    def process_observation(self, obs):
        # obs = obs.cpu().numpy()

        obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_NEAREST)

        obs = torch.from_numpy(obs).float().permute(2, 0, 1)

        return obs 


    def collect_dataset(self, 
                        episodes=0):

        for episode in range(episodes):
            done = False
            obs, info = self.env.reset()
            obs = self.process_observation(obs)
            episode_reward = 0.0

            
            while not done:
                action = self.env.action_space.sample()
                
                next_obs, reward, done, truncated, info  = self.env.step(action)
                next_obs = self.process_observation(next_obs)
                
                done = done or truncated    

                self.memory.store_transition(obs, action, reward, next_obs, done)

                obs = next_obs

                episode_reward = episode_reward + float(reward)

            print(f"Completed episode {episode} with score {episode_reward}")

            self.memory.print_stats()




            

                

                # pred, enc = self.VAE(obs)


