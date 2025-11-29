import gymnasium as gym
from model import VAE 
import torch
import torch.nn.functional as F
from buffer import ReplayBuffer, device
import matplotlib.pyplot as plt
import cv2
import os

class Agent:

    def __init__(self, human=False, max_buffer_size=100000, learning_rate=0.0001):
        if(human):
            render_mode = "human"
        else:
            render_mode = "rgb_array"

        self.env = gym.make("CarRacing-v3", render_mode=render_mode, lap_complete_percent=0.95, domain_randomize=False, continuous=False)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        obs, info = self.env.reset()
        obs = self.process_observation(obs)
        
        self.vae = VAE(observation_shape=obs.shape).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), learning_rate) 

        self.memory = ReplayBuffer(max_size=max_buffer_size, input_shape=obs.shape, n_actions=self.env.action_space.n, input_device=self.device, output_device=self.device)

        os.makedirs('models', exist_ok=True)


    def load_models(self):
        self.vae.load_the_model(filename="models/latest_vae.pt")


    def process_observation(self, obs):
        # obs = obs.cpu().numpy()

        obs = cv2.resize(obs, (64, 64), interpolation=cv2.INTER_NEAREST)

        obs = torch.from_numpy(obs).float().permute(2, 0, 1).to(self.device)

        return obs 


    def collect_dataset(self, 
                        episodes : int):

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
    

    def show(self, x, recon):
        img = torch.cat([x, recon.clamp(0,1)], dim=2)   # side-by-side
        plt.imshow(img.permute(1,2,0).cpu().detach().numpy())
        plt.axis("off")
        plt.pause(0.001)   # ← non-blocking


    def train_vae(self,
                  epochs : int,
                  batch_size: int):

        for epoch in range(epochs):

            # 1 — sample & reshape
            observations, _, _, _, _ = self.memory.sample_buffer(batch_size)

            # 2 — Q(s,a) with the online network
            predicted_observations, encoded_observations = self.vae(observations)

            # print("Obs:", type(observations))
            # print("Pred:", type(predicted_observations))
            #
            # 4 — loss & optimise
            loss = F.mse_loss(observations, predicted_observations)
            print(f"VAE Loss: {loss.item()}")
            # writer.add_scalar("Stats/model_loss", loss.item(), total_steps)

            self.vae_optimizer.zero_grad()
            loss.backward()
            self.vae_optimizer.step()

        self.vae.save_the_model(filename="models/latest_vae.pt")


    def test_vae(self):
        done = False
        obs, info = self.env.reset()
        obs = self.process_observation(obs)
        
        while not done:

            predicted_obs, encoded_obs = self.vae(obs.unsqueeze(0))
            self.show(obs / 255.0, predicted_obs[0])

            action = self.env.action_space.sample()
            
            next_obs, reward, done, truncated, info  = self.env.step(action)
            next_obs = self.process_observation(next_obs)
            
            done = done or truncated    

            obs = next_obs




            

                

                # pred, enc = self.VAE(obs)


