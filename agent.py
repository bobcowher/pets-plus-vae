import gymnasium as gym

class Agent:

    def __init__(self, human=False):
        if(human):
            render_mode = "human"
        else:
            render_mode = "rgb_array"

        self.env = gym.make("CarRacing-v3", render_mode=render_mode, lap_complete_percent=0.95, domain_randomize=False, continuous=False)


    def train(self, episodes=1):

        for episode in range(episodes):
            done = False
            obs, info = self.env.reset()
            
            while not done:
                action = self.env.action_space.sample()
                response = self.env.step(action)


