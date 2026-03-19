# C51 (Categorical DQN) for Atari using envpool
# Adapted from cleanrl/c51_atari.py (distributional RL algorithm) and
# cleanrl/dqn_atari_envpool.py (envpool integration, SimpleReplayBuffer, CSV output)
import csv
import os
import random
import time
from collections import deque
from dataclasses import dataclass

import envpool
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""

    # Algorithm specific arguments
    env_id: str = "Breakout-v5"
    """the id of the environment"""
    total_timesteps: int = 10000000
    """total timesteps of the experiments"""
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    n_atoms: int = 51
    """the number of atoms for the categorical distribution"""
    v_min: float = -10
    """the return lower bound"""
    v_max: float = 10
    """the return upper bound"""
    buffer_size: int = 200000
    """the replay memory buffer size (reduced from 1M for memory efficiency with envpool)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    target_network_frequency: int = 1000
    """the timesteps it takes to update the target network"""
    batch_size: int = 32
    """the batch size of sample from the replay memory"""
    start_e: float = 1
    """the starting epsilon for exploration"""
    end_e: float = 0.01
    """the ending epsilon for exploration"""
    exploration_fraction: float = 0.10
    """the fraction of `total_timesteps` it takes from start_e to end_e"""
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training (in terms of env steps, not global steps)"""

    # results tracking
    hypothesis_id: str = "h000"
    """hypothesis identifier for results tracking"""
    experiment_id: str = ""
    """experiment identifier (auto-generated if empty)"""
    output_dir: str = "/runs"
    """directory to save CSV results"""


class RecordEpisodeStatistics(gym.Wrapper):
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )


class SimpleReplayBuffer:
    """Simple replay buffer for vectorized environments."""
    def __init__(self, buffer_size, obs_shape, device):
        self.buffer_size = buffer_size
        self.device = device
        self.pos = 0
        self.full = False

        # Store on CPU to save GPU memory, move to device on sample
        self.observations = np.zeros((buffer_size,) + obs_shape, dtype=np.uint8)
        self.next_observations = np.zeros((buffer_size,) + obs_shape, dtype=np.uint8)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

    def add(self, obs, next_obs, actions, rewards, dones):
        """Add a batch of transitions from vectorized envs."""
        batch_size = len(obs)
        for i in range(batch_size):
            self.observations[self.pos] = obs[i]
            self.next_observations[self.pos] = next_obs[i]
            self.actions[self.pos] = actions[i]
            self.rewards[self.pos] = rewards[i]
            self.dones[self.pos] = dones[i]
            self.pos = (self.pos + 1) % self.buffer_size
            if self.pos == 0:
                self.full = True

    def sample(self, batch_size):
        max_idx = self.buffer_size if self.full else self.pos
        idxs = np.random.randint(0, max_idx, size=batch_size)
        return (
            torch.tensor(self.observations[idxs], dtype=torch.float32).to(self.device),
            torch.tensor(self.next_observations[idxs], dtype=torch.float32).to(self.device),
            torch.tensor(self.actions[idxs], dtype=torch.long).to(self.device),
            torch.tensor(self.rewards[idxs], dtype=torch.float32).to(self.device),
            torch.tensor(self.dones[idxs], dtype=torch.float32).to(self.device),
        )

    def __len__(self):
        return self.buffer_size if self.full else self.pos


# ALGO LOGIC: C51 distributional Q-network
class QNetwork(nn.Module):
    def __init__(self, envs, n_atoms=51, v_min=-10, v_max=10):
        super().__init__()
        self.n_atoms = n_atoms
        self.register_buffer("atoms", torch.linspace(v_min, v_max, steps=n_atoms))
        self.n = envs.single_action_space.n
        obs_shape = envs.single_observation_space.shape
        self.network = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
            nn.Linear(512, self.n * n_atoms),
        )

    def get_action(self, x, action=None):
        logits = self.network(x / 255.0)
        # probability mass function for each action
        pmfs = torch.softmax(logits.view(len(x), self.n, self.n_atoms), dim=2)
        q_values = (pmfs * self.atoms).sum(2)
        if action is None:
            action = torch.argmax(q_values, 1)
        return action, pmfs[torch.arange(len(x)), action]


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / duration
    return max(slope * t + start_e, end_e)


if __name__ == "__main__":
    args = tyro.cli(Args)
    if not args.experiment_id:
        args.experiment_id = f"{args.env_id}_s{args.seed}"
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    q_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate, eps=0.01 / args.batch_size)
    target_network = QNetwork(envs, n_atoms=args.n_atoms, v_min=args.v_min, v_max=args.v_max).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = SimpleReplayBuffer(
        args.buffer_size,
        envs.single_observation_space.shape,
        device,
    )

    start_time = time.time()
    avg_returns = deque(maxlen=20)
    all_episode_returns = []

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    global_step = 0

    while global_step < args.total_timesteps:
        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

        # Vectorized epsilon-greedy
        random_actions = np.random.randint(0, envs.single_action_space.n, size=args.num_envs)
        with torch.no_grad():
            actions_t, _ = q_network.get_action(torch.tensor(obs, dtype=torch.float32).to(device))
            greedy_actions = actions_t.cpu().numpy()

        explore_mask = np.random.random(args.num_envs) < epsilon
        actions = np.where(explore_mask, random_actions, greedy_actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, info = envs.step(actions)
        global_step += args.num_envs

        # Add transitions to replay buffer
        rb.add(obs, next_obs, actions, rewards, dones)

        # Log completed episodes
        for idx, d in enumerate(dones):
            if d and info["lives"][idx] == 0:
                ep_return = info["r"][idx]
                avg_returns.append(ep_return)
                all_episode_returns.append(ep_return)
                if len(all_episode_returns) % 10 == 0:
                    print(f"global_step={global_step}, episodic_return={ep_return:.1f}")
                    writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                    writer.add_scalar("charts/episodic_return", ep_return, global_step)
                    writer.add_scalar("charts/episodic_length", info["l"][idx], global_step)

        obs = next_obs

        # ALGO LOGIC: training.
        if global_step > args.learning_starts:
            if (global_step // args.num_envs) % args.train_frequency == 0:
                s_obs, s_next_obs, s_actions, s_rewards, s_dones = rb.sample(args.batch_size)

                with torch.no_grad():
                    _, next_pmfs = target_network.get_action(s_next_obs)
                    # C51 categorical projection
                    next_atoms = s_rewards.unsqueeze(1) + args.gamma * target_network.atoms * (1 - s_dones.unsqueeze(1))
                    # projection onto the support
                    delta_z = target_network.atoms[1] - target_network.atoms[0]
                    tz = next_atoms.clamp(args.v_min, args.v_max)

                    b = (tz - args.v_min) / delta_z
                    l = b.floor().clamp(0, args.n_atoms - 1)
                    u = b.ceil().clamp(0, args.n_atoms - 1)
                    # (l == u).float() handles the case where bj is exactly an integer
                    # example bj = 1, then the upper ceiling should be uj = 2, and lj = 1
                    d_m_l = (u + (l == u).float() - b) * next_pmfs
                    d_m_u = (b - l) * next_pmfs
                    target_pmfs = torch.zeros_like(next_pmfs)
                    for i in range(target_pmfs.size(0)):
                        target_pmfs[i].index_add_(0, l[i].long(), d_m_l[i])
                        target_pmfs[i].index_add_(0, u[i].long(), d_m_u[i])

                _, old_pmfs = q_network.get_action(s_obs, s_actions)
                loss = (-(target_pmfs * old_pmfs.clamp(min=1e-5, max=1 - 1e-5).log()).sum(-1)).mean()

                if (global_step // args.num_envs) % 100 == 0:
                    writer.add_scalar("losses/loss", loss.item(), global_step)
                    old_val = (old_pmfs * q_network.atoms).sum(1)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if (global_step // args.num_envs) % args.target_network_frequency == 0:
                target_network.load_state_dict(q_network.state_dict())

    # Compute robust metrics and save CSV
    if all_episode_returns:
        returns_arr = np.array(all_episode_returns)
        n_episodes = len(returns_arr)
        q4_start = int(n_episodes * 0.75)
        q4_metric = float(np.mean(returns_arr[q4_start:])) if q4_start < n_episodes else float(np.mean(returns_arr))
        auc_metric = float(np.sum(returns_arr))
        final_avg = float(np.mean(returns_arr[-20:])) if n_episodes >= 20 else float(np.mean(returns_arr))
        mean_return = float(np.mean(returns_arr))

        csv_filename = f"{args.hypothesis_id}__{args.experiment_id}.csv"
        csv_path = os.path.join(args.output_dir, csv_filename)
        os.makedirs(args.output_dir, exist_ok=True)
        with open(csv_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["env_id", "seed", "hypothesis_id", "experiment_id", "algorithm",
                         "total_timesteps", "n_episodes", "mean_return", "q4_return", "auc", "final_avg20"])
            w.writerow([args.env_id, args.seed, args.hypothesis_id, args.experiment_id, "c51",
                         args.total_timesteps, n_episodes, mean_return, q4_metric, auc_metric, final_avg])
        print(f"Results saved to {csv_path}")

    envs.close()
    writer.close()
