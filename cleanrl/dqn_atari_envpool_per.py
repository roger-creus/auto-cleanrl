# DQN with Prioritized Experience Replay (PER) for Atari using envpool
# Based on dqn_atari_envpool.py with PER from rainbow_atari.py
# Key difference from rainbow: simple 1-step DQN + PER (no n-step, no distributional, no dueling, no noisy nets)
import collections
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
import torch.nn.functional as F
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
    total_timesteps: int = 40000000
    """total timesteps of the experiments"""
    learning_rate: float = 1e-4
    """the learning rate of the optimizer"""
    num_envs: int = 16
    """the number of parallel game environments"""
    buffer_size: int = 200000
    """the replay memory buffer size (reduced from 1M for memory efficiency with envpool)"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 1.0
    """the target network update rate"""
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

    # PER-specific hyperparameters
    prioritized_replay_alpha: float = 0.6
    """alpha parameter for prioritized replay buffer (how much prioritization to use)"""
    prioritized_replay_beta: float = 0.4
    """initial beta parameter for importance sampling correction (annealed to 1.0)"""
    prioritized_replay_eps: float = 1e-6
    """small constant added to priorities to prevent zero probabilities"""

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


# ── Segment trees for PER ────────────────────────────────────────────────────
# Adapted from https://github.com/openai/baselines/blob/master/baselines/common/segment_tree.py

class SumSegmentTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.zeros(self.tree_size, dtype=np.float64)

    def _propagate(self, idx):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = self.tree[parent * 2 + 1] + self.tree[parent * 2 + 2]
            parent = (parent - 1) // 2

    def update(self, idx, value):
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = value
        self._propagate(tree_idx)

    def total(self):
        return self.tree[0]

    def retrieve(self, value):
        idx = 0
        while idx * 2 + 1 < self.tree_size:
            left = idx * 2 + 1
            right = left + 1
            if value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = right
        return idx - (self.capacity - 1)


class MinSegmentTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree_size = 2 * capacity - 1
        self.tree = np.full(self.tree_size, float("inf"), dtype=np.float64)

    def _propagate(self, idx):
        parent = (idx - 1) // 2
        while parent >= 0:
            self.tree[parent] = min(self.tree[parent * 2 + 1], self.tree[parent * 2 + 2])
            parent = (parent - 1) // 2

    def update(self, idx, value):
        tree_idx = idx + self.capacity - 1
        self.tree[tree_idx] = value
        self._propagate(tree_idx)

    def min(self):
        return self.tree[0]


def _next_power_of_2(n):
    """Return the smallest power of 2 >= n."""
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


# ── Prioritized Replay Buffer for vectorized environments ─────────────────────

PrioritizedBatch = collections.namedtuple(
    "PrioritizedBatch",
    ["observations", "next_observations", "actions", "rewards", "dones", "indices", "weights"],
)


class PrioritizedReplayBuffer:
    """Prioritized Experience Replay buffer adapted for vectorized environments.

    The segment tree capacity must be a power of 2, so we round up buffer_size.
    Vectorized env transitions (batch of num_envs) are added one at a time in a loop.
    """

    def __init__(self, buffer_size, obs_shape, device, alpha=0.6, beta=0.4, eps=1e-6):
        # Round up to next power of 2 for segment tree
        self.tree_capacity = _next_power_of_2(buffer_size)
        self.buffer_size = buffer_size
        self.device = device
        self.alpha = alpha
        self.beta = beta
        self.eps = eps

        # Data storage (actual buffer_size, not tree_capacity)
        self.observations = np.zeros((buffer_size,) + obs_shape, dtype=np.uint8)
        self.next_observations = np.zeros((buffer_size,) + obs_shape, dtype=np.uint8)
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.float32)

        self.pos = 0
        self.size = 0
        self.max_priority = 1.0

        # Segment trees use tree_capacity (power of 2)
        self.sum_tree = SumSegmentTree(self.tree_capacity)
        self.min_tree = MinSegmentTree(self.tree_capacity)

    def add(self, obs, next_obs, actions, rewards, dones):
        """Add a batch of transitions from vectorized envs.

        Args:
            obs: (num_envs, *obs_shape) observations
            next_obs: (num_envs, *obs_shape) next observations
            actions: (num_envs,) actions
            rewards: (num_envs,) rewards
            dones: (num_envs,) done flags
        """
        batch_size = len(obs)
        for i in range(batch_size):
            idx = self.pos

            self.observations[idx] = obs[i]
            self.next_observations[idx] = next_obs[i]
            self.actions[idx] = actions[i]
            self.rewards[idx] = rewards[i]
            self.dones[idx] = dones[i]

            # New transitions get max priority so they are sampled at least once
            priority = self.max_priority ** self.alpha
            self.sum_tree.update(idx, priority)
            self.min_tree.update(idx, priority)

            self.pos = (self.pos + 1) % self.buffer_size
            self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        """Sample a batch of transitions with prioritized sampling."""
        indices = []
        p_total = self.sum_tree.total()
        segment = p_total / batch_size

        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            upperbound = np.random.uniform(a, b)
            idx = self.sum_tree.retrieve(upperbound)
            # Clamp index to valid range (safety against floating point edge cases)
            idx = min(idx, self.size - 1)
            indices.append(idx)

        # Compute importance sampling weights
        probs = np.array(
            [self.sum_tree.tree[idx + self.tree_capacity - 1] for idx in indices],
            dtype=np.float64,
        )
        # Avoid division by zero
        probs = np.clip(probs, 1e-10, None)
        weights = (self.size * probs / p_total) ** (-self.beta)
        weights = weights / weights.max()

        return PrioritizedBatch(
            observations=torch.tensor(self.observations[indices], dtype=torch.float32).to(self.device),
            next_observations=torch.tensor(self.next_observations[indices], dtype=torch.float32).to(self.device),
            actions=torch.tensor(self.actions[indices], dtype=torch.long).to(self.device),
            rewards=torch.tensor(self.rewards[indices], dtype=torch.float32).to(self.device),
            dones=torch.tensor(self.dones[indices], dtype=torch.float32).to(self.device),
            indices=indices,
            weights=torch.tensor(weights, dtype=torch.float32).to(self.device),
        )

    def update_priorities(self, indices, td_errors):
        """Update priorities based on TD errors.

        Args:
            indices: list of buffer indices
            td_errors: numpy array of absolute TD errors
        """
        priorities = np.abs(td_errors) + self.eps
        self.max_priority = max(self.max_priority, priorities.max())

        for idx, priority in zip(indices, priorities):
            self.sum_tree.update(idx, priority ** self.alpha)
            self.min_tree.update(idx, priority ** self.alpha)

    def __len__(self):
        return self.size


# ── Q-Network (NatureCNN — same as base DQN) ─────────────────────────────────

class QNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
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
            nn.Linear(512, envs.single_action_space.n),
        )

    def forward(self, x):
        return self.network(x / 255.0)


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

    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = PrioritizedReplayBuffer(
        args.buffer_size,
        envs.single_observation_space.shape,
        device,
        alpha=args.prioritized_replay_alpha,
        beta=args.prioritized_replay_beta,
        eps=args.prioritized_replay_eps,
    )

    start_time = time.time()
    avg_returns = deque(maxlen=20)
    all_episode_returns = []

    # TRY NOT TO MODIFY: start the game
    obs = envs.reset()
    global_step = 0

    while global_step < args.total_timesteps:
        # Anneal PER beta from initial value to 1.0 over training
        beta_fraction = min(1.0, global_step / args.total_timesteps)
        rb.beta = args.prioritized_replay_beta + beta_fraction * (1.0 - args.prioritized_replay_beta)

        # ALGO LOGIC: put action logic here
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

        # Vectorized epsilon-greedy
        random_actions = np.random.randint(0, envs.single_action_space.n, size=args.num_envs)
        with torch.no_grad():
            q_values = q_network(torch.tensor(obs, dtype=torch.float32).to(device))
            greedy_actions = torch.argmax(q_values, dim=1).cpu().numpy()

        explore_mask = np.random.random(args.num_envs) < epsilon
        actions = np.where(explore_mask, random_actions, greedy_actions)

        # TRY NOT TO MODIFY: execute the game and log data.
        next_obs, rewards, dones, info = envs.step(actions)
        global_step += args.num_envs

        # Add transitions to prioritized replay buffer
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
                data = rb.sample(args.batch_size)

                with torch.no_grad():
                    target_max, _ = target_network(data.next_observations).max(dim=1)
                    td_target = data.rewards + args.gamma * target_max * (1 - data.dones)

                old_val = q_network(data.observations).gather(1, data.actions.unsqueeze(1)).squeeze()

                # Per-sample TD errors for priority updates
                td_errors = td_target - old_val

                # Weighted MSE loss using importance sampling weights
                # loss = mean(w_i * (td_target_i - q_i)^2)
                loss = (data.weights * (td_errors ** 2)).mean()

                # Update priorities in the replay buffer
                rb.update_priorities(data.indices, td_errors.detach().cpu().numpy())

                if (global_step // args.num_envs) % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss.item(), global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    writer.add_scalar("charts/beta", rb.beta, global_step)
                    sps = int(global_step / (time.time() - start_time))
                    print("SPS:", sps)
                    writer.add_scalar("charts/SPS", sps, global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            # update target network
            if (global_step // args.num_envs) % args.target_network_frequency == 0:
                for target_param, q_param in zip(target_network.parameters(), q_network.parameters()):
                    target_param.data.copy_(
                        args.tau * q_param.data + (1.0 - args.tau) * target_param.data
                    )

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
            w.writerow([args.env_id, args.seed, args.hypothesis_id, args.experiment_id, "dqn_per",
                         args.total_timesteps, n_episodes, mean_return, q4_metric, auc_metric, final_avg])
        print(f"Results saved to {csv_path}")

    envs.close()
    writer.close()
