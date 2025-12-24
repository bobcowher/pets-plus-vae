from numpy import int32
import torch.nn as nn
import torch.nn.functional as F
import torch
import os

class BaseModel(nn.Module):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def save_the_model(self, filename='models/latest.pt'):
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        torch.save(self.state_dict(), filename)
        print(f"Saved model to {filename}")


    def load_the_model(self, filename='models/latest.pt', device='cuda'):
        try:
            self.load_state_dict(torch.load(filename, map_location=device))
            print(f"Loaded weights from {filename}")
        except FileNotFoundError:
            print(f"No weights file found at {filename}")
        except Exception as e:
            print(f"Error loading model from {filename}: {e}")


class VAE(BaseModel):

    def __init__(self, observation_shape=()):
        super().__init__()

        # print(observation_shape[-1])
        # conv_output_dim = 64

        self.conv1 = nn.Conv2d(3, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1)

        self.flatten = torch.nn.Flatten()

        with torch.no_grad():
            dummy = torch.zeros(1, *observation_shape, dtype=torch.uint8)
            feats = self._conv_features(dummy)         # (1, C_enc, H_enc, W_enc)
            self.conv_output_shape = feats.shape[1:]   # (C_enc, H_enc, W_enc)
            self.flattened_dim = feats.numel() // 1    # C_enc * H_enc * W_enc
            print(f"Conv output shape: {feats.shape}, flattened dim: {self.flattened_dim}")

        # conv_output = self._conv_forward(torch.zeros(1, *observation_shape))
        # conv_output_dim = conv_output.shape[-1]

        latent_dim = 128  # or whatever you pick

        # self.fc_enc = nn.Linear(self.flattened_dim, latent_dim)
        # self.fc_dec = nn.Linear(latent_dim, self.flattened_dim)
        self.fc_mu = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_logvar = nn.Linear(self.flattened_dim, latent_dim)
        self.fc_dec = nn.Linear(latent_dim, self.flattened_dim)

        self.deconv1 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1)
        self.deconv4 = nn.ConvTranspose2d(32, observation_shape[0], kernel_size=4, stride=2, padding=1)
        # self.conv3 = nn.Conv2d()

        print(f"VAE network initialized. Input shape: {observation_shape}")


    def _reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _conv_features(self, x):
        # Convert uint8 to float if needed (for initialization)
        if x.dtype == torch.uint8:
            x = x.float() / 255.0
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        return x  # (B, C_enc, H_enc, W_enc)

    def _conv_forward(self, x):
        x = self._conv_features(x)
        x = self.flatten(x)
        return x

    def _deconv_forward(self, x):
        x = x.view(-1, *self.conv_output_shape)
        x = F.relu(self.deconv1(x))
        x = F.relu(self.deconv2(x))
        x = F.relu(self.deconv3(x))
        x = F.relu(self.deconv4(x))
       
        return x
        
    def encode(self, x):
        # x: (B,3,H,W) in [0,1]
        with torch.no_grad():
            x = self._conv_forward(x)
            mu = self.fc_mu(x)
            logvar = self.fc_logvar(x)
            z = self._reparameterize(mu, logvar)
        return z
    
    def forward(self, x):
        # Ensure input is uint8 tensor in [0,255] range
        assert x.dtype == torch.uint8, f"Expected uint8 input, got {x.dtype}"
        
        # Add batch dimension if missing
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        x = x.float() / 255.0
        x = self._conv_forward(x)

        mu = self.fc_mu(x)
        logvar = self.fc_logvar(x)

        z = self._reparameterize(mu, logvar)

        x = F.relu(self.fc_dec(z))

        x = self._deconv_forward(x)

        recon = torch.sigmoid(x)

        return recon, mu, logvar, z



class DynamicsModel(nn.Module):

    def __init__(self, hidden_dim=256, obs_shape=None, action_shape=None) -> None:
        super(DynamicsModel, self).__init__()

        self.fc1 = nn.Linear(obs_shape + action_shape, hidden_dim) # Accept the state concatenated with reward.
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.obs_diff_output = nn.Linear(hidden_dim, obs_shape) # Return the state dimensions + the reward
        self.reward_output = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):

        x = torch.cat([state, action], dim=-1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        obs_diff = self.obs_diff_output(x)
        reward = self.reward_output(x)

        return obs_diff, reward


class EnsembleModel:
    def __init__(self, num_models=5, hidden_dim=256, obs_shape=None, action_shape=None, device=None, learning_rate=0.0001):
        self.models = [DynamicsModel(hidden_dim=hidden_dim, obs_shape=obs_shape, action_shape=action_shape).to(device) for _ in range(num_models)]
        self.optimizers = [torch.optim.Adam(m.parameters(), lr=learning_rate) for m in self.models]
        self.model_save_dir = 'models'
    
    def train_step(self, states, actions, next_states, rewards):
        total_loss = 0
        for i, (model, optimizer) in enumerate(zip(self.models, self.optimizers)):
            # Bootstrap sampling - different batch per model
            indices = torch.randint(0, len(states), (len(states),))
            batch_states = states[indices]
            batch_next_states = next_states[indices]
            batch_actions = actions[indices]
            batch_rewards = rewards[indices]

            obs_diffs = batch_next_states - batch_states
            
            predicted_obs_diffs, predicted_rewards = model(batch_states, batch_actions)

            loss = F.mse_loss(torch.cat([predicted_obs_diffs, predicted_rewards], dim=-1), 
                              torch.cat([obs_diffs, batch_rewards], dim=-1))
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        return total_loss / len(self.models)
    
    def predict(self, states, actions, return_uncertainty=False):
        with torch.no_grad():
            obs_diffs = []
            rewards = []
            for model in self.models:
                obs_diff, reward = model(states, actions)
                obs_diffs.append(obs_diff)
                rewards.append(reward)
            
            # Return averaged predictions
            obs_diffs = torch.stack(obs_diffs)
            rewards = torch.stack(rewards)
            
            obs_uncertainty = obs_diffs.var(0)
            reward_uncertainty = rewards.var(0)

            avg_obs_diff = obs_diffs.mean(0)
            avg_reward = rewards.mean(0)

            return avg_obs_diff, avg_reward, obs_uncertainty, reward_uncertainty
    
    def save_the_model(self, filename='latest.pt'):
        os.makedirs(self.model_save_dir, exist_ok=True) 
        
        # Remove .pt extension if present to maintain consistency
        base_filename = filename.replace('.pt', '')

        for i, model in enumerate(self.models):
            torch.save(model.state_dict(), f"{self.model_save_dir}/{base_filename}_{i}.pt")
            
        print(f"Saved ensemble models to {self.model_save_dir}/{base_filename}_*.pt")

    def load_the_model(self, filename='latest.pt', device='cuda'):
        # Remove .pt extension if present to maintain consistency  
        base_filename = filename.replace('.pt', '')
        
        loaded_count = 0
        for i, model in enumerate(self.models):
            file_path = f"{self.model_save_dir}/{base_filename}_{i}.pt"

            try:
                model.load_state_dict(torch.load(file_path, map_location=device))
                print(f"Loaded weights from {file_path}")
                loaded_count += 1
            except FileNotFoundError:
                print(f"No weights file found at {file_path}")
                
        if loaded_count == len(self.models):
            print(f"Successfully loaded all {loaded_count} ensemble models")
        else:
            print(f"Warning: Only loaded {loaded_count}/{len(self.models)} ensemble models")






        

