from stable_baselines3.common.callbacks import BaseCallback

class EpisodeLoggerCallback(BaseCallback):
    def __init__(self):
        super().__init__()
        self.ep_rewards = []

    def _on_step(self):
        for info in self.locals.get("infos", []):
            if "episode" in info:
                r = info["episode"]["r"]
                l = info["episode"]["l"]
                #print(info)
                self.ep_rewards.append(r)
                avg = sum(self.ep_rewards[-20:]) / min(len(self.ep_rewards), 20)
                print(f"   Ep {len(self.ep_rewards):4d} | "
                      f"pasos: {l:5d} | "
                      f"reward: {r:+8.1f} | "
                      f"media×20: {avg:+8.1f}")
        return True