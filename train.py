from agent import Agent

agent = Agent(human=False)

agent.collect_dataset(episodes=20)

agent.train_vae(epochs=100, batch_size=32)

