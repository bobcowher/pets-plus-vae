from agent import Agent

agent = Agent(human=False)

agent.collect_dataset(episodes=1)

agent.train_vae(epochs=10, batch_size=32)

