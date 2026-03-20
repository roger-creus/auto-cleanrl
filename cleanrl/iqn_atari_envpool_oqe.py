# IQN + Optimistic Quantile Exploration (OQE) for Atari using envpool
# Novel exploration technique: during action selection, sample τ from U(β_t, 1.0)
# where β_t anneals from oqe_start (e.g. 0.5) to 0.0 over training.
# Early: upper-half quantiles → optimistic, directed exploration toward high-upside actions.
# Late: full distribution → standard mean-value exploitation.
# Training loss remains unchanged: standard IQN quantile regression with τ ~ U(0,1).
# This uses IQN's distributional information for principled UCB-like exploration
# without external bonuses (RND, ICM) or random epsilon-greedy.
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
    """the replay memory buffer size"""
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

    # IQN specific arguments
    n_quantiles: int = 64
    """number of quantile samples for training (N in the paper)"""
    n_quantiles_policy: int = 32
    """number of quantile samples for action selection (K in the paper)"""
    cosine_embedding_dim: int = 64
    """dimension of the cosine embedding (n in the paper)"""
    huber_kappa: float = 1.0
    """threshold for the Huber loss used in quantile regression"""

    # OQE (Optimistic Quantile Exploration) arguments
    oqe_start: float = 0.5
    """initial lower bound for τ sampling during action selection (0.5 = upper-half only)"""
    oqe_end: float = 0.0
    """final lower bound for τ sampling (0.0 = standard full-distribution mean)"""
    oqe_fraction: float = 0.5
    """fraction of training over which to anneal oqe_start → oqe_end"""

    # results tracking
    hypothesis_id: str = "h066"
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

        self.observations = np.zeros((buffer_size,) + obs_shape, dtype=np.uint8)
        self.next_observations = np.zeros((buffer_size,) + obs_shape, dtype=np.uint8)
        self.actions = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards = np.zeros((buffer_size,), dtype=np.float32)
        self.dones = np.zeros((buffer_size,), dtype=np.float32)

    def add(self, obs, next_obs, actions, rewards, dones):
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


class IQNNetwork(nn.Module):
    """Implicit Quantile Network.

    Architecture:
    1. CNN encoder: obs -> 512-dim features (psi)
    2. Cosine embedding: tau -> cosine_embedding_dim -> 512-dim (phi)
    3. Element-wise multiply: psi * phi -> 512-dim
    4. Linear head: 512 -> n_actions

    Forward pass takes (obs, n_quantiles) and returns:
    - quantile_values: [batch, n_quantiles, n_actions]
    - taus: [batch, n_quantiles] (the sampled quantile fractions)
    """
    def __init__(self, envs, cosine_embedding_dim=64):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        n_actions = envs.single_action_space.n
        self.n_actions = n_actions
        self.cosine_embedding_dim = cosine_embedding_dim

        # State encoder (NatureCNN)
        self.encoder = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3136, 512),
            nn.ReLU(),
        )

        # Cosine embedding for quantile fractions
        # cos(pi * i * tau) for i = 0, 1, ..., cosine_embedding_dim-1
        # Precompute i * pi
        self.register_buffer(
            "i_pi",
            torch.arange(1, cosine_embedding_dim + 1, dtype=torch.float32).unsqueeze(0) * torch.pi
        )  # [1, cosine_embedding_dim]

        # Linear layer to project cosine embedding to feature dim
        self.cosine_linear = nn.Linear(cosine_embedding_dim, 512)

        # Value head
        self.value_head = nn.Linear(512, n_actions)

    def forward(self, x, n_quantiles, taus=None):
        """
        Args:
            x: observations [batch, C, H, W]
            n_quantiles: number of quantile samples
            taus: optional precomputed tau values [batch, n_quantiles]
        Returns:
            quantile_values: [batch, n_quantiles, n_actions]
            taus: [batch, n_quantiles]
        """
        batch_size = x.shape[0]

        # Encode state
        features = self.encoder(x / 255.0)  # [batch, 512]

        # Sample or use provided quantile fractions
        if taus is None:
            taus = torch.rand(batch_size, n_quantiles, device=x.device)  # [batch, n_quantiles]

        # Cosine embedding: cos(pi * i * tau) for i=1..n
        # taus: [batch, n_quantiles] -> [batch, n_quantiles, 1]
        # i_pi: [1, cosine_embedding_dim] -> broadcast
        tau_expanded = taus.unsqueeze(2)  # [batch, n_quantiles, 1]
        cos_embedding = torch.cos(tau_expanded * self.i_pi)  # [batch, n_quantiles, cosine_embedding_dim]

        # Project cosine embedding to feature dim
        tau_features = F.relu(self.cosine_linear(cos_embedding))  # [batch, n_quantiles, 512]

        # Element-wise multiply with state features
        # features: [batch, 512] -> [batch, 1, 512]
        combined = features.unsqueeze(1) * tau_features  # [batch, n_quantiles, 512]

        # Compute quantile values
        quantile_values = self.value_head(combined)  # [batch, n_quantiles, n_actions]

        return quantile_values, taus


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

    # Seeding
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

    # IQN networks
    q_network = IQNNetwork(envs, cosine_embedding_dim=args.cosine_embedding_dim).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = IQNNetwork(envs, cosine_embedding_dim=args.cosine_embedding_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = SimpleReplayBuffer(
        args.buffer_size,
        envs.single_observation_space.shape,
        device,
    )

    start_time = time.time()
    avg_returns = deque(maxlen=20)
    all_episode_returns = []

    obs = envs.reset()
    global_step = 0

    while global_step < args.total_timesteps:
        # Epsilon-greedy action selection with OQE (Optimistic Quantile Exploration)
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

        # OQE: anneal the lower bound for τ sampling during action selection
        oqe_beta = linear_schedule(args.oqe_start, args.oqe_end, args.oqe_fraction * args.total_timesteps, global_step)

        random_actions = np.random.randint(0, envs.single_action_space.n, size=args.num_envs)
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, dtype=torch.float32).to(device)
            K = args.n_quantiles_policy

            # OQE: sample τ from U(oqe_beta, 1.0) instead of U(0, 1)
            # When oqe_beta=0.5: only upper-half quantiles → optimistic exploration
            # When oqe_beta=0.0: standard full-distribution mean → exploitation
            oqe_taus = torch.rand(args.num_envs, K, device=device) * (1.0 - oqe_beta) + oqe_beta

            quantile_values, _ = q_network(obs_tensor, n_quantiles=K, taus=oqe_taus)
            # quantile_values: [num_envs, K, n_actions]
            # Mean across biased quantiles gives optimistic Q-value estimate
            mean_q = quantile_values.mean(dim=1)  # [num_envs, n_actions]
            greedy_actions = torch.argmax(mean_q, dim=1).cpu().numpy()

        explore_mask = np.random.random(args.num_envs) < epsilon
        actions = np.where(explore_mask, random_actions, greedy_actions)

        # Execute actions
        next_obs, rewards, dones, info = envs.step(actions)
        global_step += args.num_envs

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

        # Training
        if global_step > args.learning_starts:
            if (global_step // args.num_envs) % args.train_frequency == 0:
                s_obs, s_next_obs, s_actions, s_rewards, s_dones = rb.sample(args.batch_size)
                B = args.batch_size
                N = args.n_quantiles
                kappa = args.huber_kappa

                with torch.no_grad():
                    # DDQN-style: online net selects actions, target net evaluates
                    online_qv, _ = q_network(s_next_obs, n_quantiles=N)  # [B, N, n_actions]
                    online_mean_q = online_qv.mean(dim=1)  # [B, n_actions]
                    next_actions = online_mean_q.argmax(dim=1)  # [B]

                    # Target quantile values for selected actions
                    # Sample fresh tau values for target (N' = N)
                    target_qv, target_taus = target_network(s_next_obs, n_quantiles=N)  # [B, N, n_actions]
                    next_actions_expanded = next_actions.unsqueeze(1).unsqueeze(2).expand(-1, N, 1)
                    target_q = target_qv.gather(2, next_actions_expanded).squeeze(2)  # [B, N]

                    # TD target
                    td_target = s_rewards.unsqueeze(1) + args.gamma * (1 - s_dones.unsqueeze(1)) * target_q  # [B, N]

                # Current quantile estimates for taken actions
                # Sample fresh tau values for online network
                current_qv, current_taus = q_network(s_obs, n_quantiles=N)  # [B, N, n_actions]
                s_actions_expanded = s_actions.unsqueeze(1).unsqueeze(2).expand(-1, N, 1)
                current_q = current_qv.gather(2, s_actions_expanded).squeeze(2)  # [B, N]

                # --- Quantile Huber Loss ---
                # Pairwise TD errors: current_q[:, i] vs td_target[:, j]
                pred = current_q.unsqueeze(2)    # [B, N, 1] (i dimension)
                target = td_target.unsqueeze(1)  # [B, 1, N'] (j dimension)
                delta = target - pred            # [B, N, N'] pairwise TD errors

                # Huber loss element-wise
                abs_delta = delta.abs()
                huber = torch.where(
                    abs_delta <= kappa,
                    0.5 * delta ** 2,
                    kappa * (abs_delta - 0.5 * kappa),
                )  # [B, N, N']

                # Asymmetric weighting: |tau_i - I(delta < 0)|
                # current_taus: [B, N] -> [B, N, 1]
                taus_for_loss = current_taus.unsqueeze(2)  # [B, N, 1]
                quantile_weight = torch.abs(taus_for_loss - (delta < 0).float())  # [B, N, N']

                # Quantile Huber loss
                quantile_huber_loss = quantile_weight * huber / kappa  # [B, N, N']

                # Sum over target quantiles (j), mean over sample quantiles (i) and batch
                loss = quantile_huber_loss.sum(dim=2).mean()

                if (global_step // args.num_envs) % 100 == 0:
                    writer.add_scalar("losses/quantile_loss", loss.item(), global_step)
                    writer.add_scalar("losses/mean_q_values", current_q.mean().item(), global_step)
                    writer.add_scalar("charts/oqe_beta", oqe_beta, global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

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
            w.writerow([args.env_id, args.seed, args.hypothesis_id, args.experiment_id, "iqn_oqe",
                         args.total_timesteps, n_episodes, mean_return, q4_metric, auc_metric, final_avg])
        print(f"Results saved to {csv_path}")

    envs.close()
    writer.close()
