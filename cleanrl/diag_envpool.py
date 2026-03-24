import sys
print("PYTHON STARTED", flush=True)
print(f"Python: {sys.version}", flush=True)
print("Importing envpool...", flush=True)
import envpool
print("envpool imported", flush=True)
print("Creating env Asterix-v5 seed=1 num_envs=8", flush=True)
env = envpool.make("Asterix-v5", env_type="gym", num_envs=8, episodic_life=True, reward_clip=True, seed=1)
print(f"envpool.make() SUCCESS", flush=True)
obs = env.reset()
print(f"reset() SUCCESS, obs shape: {obs.shape}", flush=True)
print("DIAGNOSTIC COMPLETE", flush=True)
