import gymnasium as gym
from model import VAE, EnsembleModel 
import torch
import torch.nn.functional as F
import numpy as np
from buffer import ReplayBuffer, device
import matplotlib.pyplot as plt
import cv2
import os
import sys
from torch.utils.tensorboard import SummaryWriter
import datetime

class Agent:

    def __init__(self, human=False, max_buffer_size=100000, learning_rate=0.0001):
        if(human):
            render_mode = "human"
        else:
            render_mode = "rgb_array"

        self.env = gym.make("CarRacing-v3", render_mode=render_mode, lap_complete_percent=0.95, domain_randomize=False, continuous=True)

        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        
        obs, info = self.env.reset()
        action = self.env.action_space.sample()

        print(f"Sample Action: {action}")

        obs = self.process_observation(obs)
        
        self.vae = VAE(observation_shape=obs.shape).to(self.device)
        self.vae_optimizer = torch.optim.Adam(self.vae.parameters(), learning_rate) 

        self.ensemble = EnsembleModel(obs_shape=obs.shape[0], action_shape=action.shape[0], device=self.device)


        self.memory = ReplayBuffer(max_size=max_buffer_size, input_shape=obs.shape, n_actions=self.env.action_space.shape[0], input_device=self.device, output_device=self.device)

        os.makedirs('models', exist_ok=True)

        self.left_bias = True


    def load_models(self):
        self.vae.load_the_model(filename="models/latest_vae.pt")


    def heuristic_action(self):
        
        if(self.left_bias):
            left_steer = 0.15
            right_steer = 0.1
        else:
            left_steer = 0.1
            right_steer = 0.15

        steer = np.random.normal(left_steer, right_steer)     # small right bias
        gas   = 1.0
        brake = 0.0
        return np.array([steer, gas, brake], dtype=np.float32)


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
                action = self.heuristic_action()
                
                next_obs, reward, done, truncated, info  = self.env.step(action)
                next_obs = self.process_observation(next_obs)
                
                done = done or truncated    

                self.memory.store_transition(obs, action, reward, next_obs, done)

                obs = next_obs

                episode_reward = episode_reward + float(reward)
            
            self.left_bias = not self.left_bias

            print(f"Completed episode {episode} with score {episode_reward}")

            self.memory.print_stats()


    def show_cv(self, img_tensor, win_name="VAE", delay=1, scale=6):
        """
        scale: how much to enlarge the image (4 → 64x64 becomes 256x256)
        """
        img = img_tensor.detach().cpu()

        if img.max() <= 1.0:
            img = img * 255.0

        img = img.permute(1, 2, 0).numpy().astype(np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # ----- resize here -----
        h, w = img.shape[:2]
        img = cv2.resize(img, (w * scale, h * scale), interpolation=cv2.INTER_NEAREST)

        cv2.imshow(win_name, img)
        key = cv2.waitKey(delay) & 0xFF
        return key != ord('q')

    
    def plan_action(self, current_state, horizon=10, num_samples=100):
        current_state = torch.tensor(current_state, dtype=torch.float32).to(self.device)

        action_sequences = torch.rand(num_samples, horizon, 2).to(self.device) * 2 - 1
        
        # Vectorized rollouts
        states = current_state.unsqueeze(0).expand(num_samples, -1)  # [num_samples, state_dim]
        total_returns = torch.zeros(num_samples, device=self.device)
        
        for t in range(horizon):
            actions = action_sequences[:, t, :]  # [num_samples, action_dim]
            
            # Batch forward pass
            delta_states, rewards, delta_uncertainty, reward_uncertainty = self.model.predict(states, actions) 
            states = states + delta_states
                
            uncertainty_penalty = delta_uncertainty.sum(-1) + reward_uncertainty.squeeze(-1)

            total_returns += rewards.squeeze(-1)
            total_returns -= 25.0 * uncertainty_penalty  # Reduced uncertainty penalty
        # 
        best_idx = torch.argmax(total_returns)
        return action_sequences[best_idx, 0].cpu().numpy()

    
    def train(self, episodes):

        total_steps = 0
        best_score = -1000

        summary_writer_name = f'runs/{datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")}'
        writer = SummaryWriter(summary_writer_name)

        for episode in range(episodes):
            done = False
            episode_reward = 0.0
            obs, info = self.env.reset()

            while not done:
                if episode < 10:
                    action = self.env.action_space.sample()
                else:
                    _, _, _, z_obs = self.vae(obs)
                    action = self.plan_action(current_state=obs)

                next_obs, reward, done, truncated, info = self.env.step(action)
                self.memory.store_transition(obs, action, reward, next_obs, done)
                obs = next_obs
                episode_reward = episode_reward + float(reward)
                # self.env.render()
  
                if(done or truncated):
                    break

            if(episode_reward > best_score):
                best_score = episode_reward
                self.ensemble.save_the_model('best.pt')

            self.ensemble.save_the_model('latest.pt')
            
            writer.add_scalar("Score/Episode Reward", episode_reward, episode)
            print(f"Episode {episode} finished. Reward: {episode_reward}")

            
            if(episode % 10 == 0):
    
                self.train_vae(epochs=50, batch_size=64)

                for _ in range(100):
                    if(self.memory.can_sample(batch_size=self.batch_size)):
                        states, actions, rewards, next_states, dones = self.memory.sample_buffer(batch_size=self.batch_size)

                        _, _, _, z_states = self.vae(states)
                        _, _, _, z_next_states = self.vae(next_states)
                       
                        # actions = actions.unsqueeze(1).long()
                        rewards = rewards.unsqueeze(1)
                        dones = dones.unsqueeze(1).float()

        #                    predicted_obs_diffs, predicated_rewards = self.model.predict(states, actions)

                        loss = self.ensemble.train_step(states=z_states,
                                                        next_states=z_next_states,
                                                        actions=actions,
                                                        rewards=rewards)

                        writer.add_scalar("Loss/model", loss, total_steps)

                        total_steps += 1


    def train_vae(self,
                  epochs : int,
                  batch_size: int):

        for epoch in range(epochs):

            # 1 — sample & reshape
            observations, _, _, _, _ = self.memory.sample_buffer(batch_size)

            # 2 — Q(s,a) with the online network
            recon, mu, logvar, z = self.vae(observations)

            # print("Obs:", type(observations))
            # print("Pred:", type(predicted_observations))
            #
            # 4 — loss & optimise
            recon_loss = F.mse_loss(observations, recon)
            # writer.add_scalar("Stats/model_loss", loss.item(), total_steps)
            kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1).mean()

            beta = 1e-4

            loss = recon_loss + beta * kl

            self.vae_optimizer.zero_grad()
            loss.backward()
            self.vae_optimizer.step()

            print(f"VAE Recon Loss: {recon_loss.item()}")
            print(f"VAE Loss {loss.item()}")

        self.vae.save_the_model(filename="models/latest_vae.pt")


    def test_vae(self, max_steps: int = 500):
        self.vae.eval()
        obs, info = self.env.reset()
        obs = self.process_observation(obs)
        step = 0
        try:
            while step < max_steps:
                inp = (obs / 255.0).unsqueeze(0)
                with torch.no_grad():
                    recon, _, _, _ = self.vae(inp)

                # side-by-side
                pair = torch.cat([obs / 255.0, recon[0].clamp(0,1)], dim=2)  # (3, H, 2W)

                if not self.show_cv(pair, delay=20):  # 10ms = ~100 FPS cap
                    break

                action = self.env.action_space.sample()
                next_obs, reward, done, truncated, info = self.env.step(action)
                next_obs = self.process_observation(next_obs)
                if done or truncated:
                    break
                obs = next_obs
                step += 1
        except KeyboardInterrupt:
            print("Interrupted by user.")
        finally:
            cv2.destroyAllWindows()
            self.vae.train()



