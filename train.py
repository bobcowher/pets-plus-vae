from agent import Agent

agent = Agent(human=False)

# agent.collect_dataset(episodes=12)

# agent.train_vae(epochs=1000, batch_size=64)

agent.train(episodes=10000)

