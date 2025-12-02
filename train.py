from agent import Agent

agent = Agent(human=False)

agent.collect_dataset(episodes=100)

agent.train_vae(epochs=1000, batch_size=32)

