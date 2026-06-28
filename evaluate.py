from stable_baselines3 import PPO
from scp_env2 import SCPClassDEnv

env   = SCPClassDEnv()
model = PPO.load("models/best/best_model")

print("Observando al agente jugar...")
obs, _ = env.reset()
ep = 0

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, _ = env.step(action)
    if terminated or truncated:
        ep += 1
        print(f"Episodio {ep} terminado")
        obs, _ = env.reset()
