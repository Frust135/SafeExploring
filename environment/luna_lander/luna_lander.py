import gym


class LunaLanderEnviorment:
    def __init__(self):
        self.env = gym.make("LunarLander-v2", render_mode="human")
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def run(self, episode):
        observation = self.env.reset()
        rewards = 0
        actions = []
        for step in range(100):
            self.env.render()
            action = self.env.action_space.sample()
            observation, reward, terminated, truncated, info = self.env.step(
                action)
            if terminated or truncated:
                observation, info = self.env.reset()
            rewards += reward
            actions.append(action)
        self.env.close()
        return rewards, actions
