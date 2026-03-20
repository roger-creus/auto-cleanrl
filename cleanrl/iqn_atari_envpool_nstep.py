# IQN + N-step returns for Atari using envpool
# Combines IQN (best single DQN component, +0.0011 IQM vs DQN) with
# N-step returns (complementary improvement, +0.0006 IQM vs DQN).
# IQN already uses DDQN-style action selection internally.
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
    n_step: int = 3
    """number of steps for multi-step returns"""

    # IQN specific arguments
    n_quantiles: int = 64
    """number of quantile samples for training (N in the paper)"""
    n_quantiles_policy: int = 32
    """number of quantile samples for action selection (K in the paper)"""
    cosine_embedding_dim: int = 64
    """dimension of the cosine embedding"""
    huber_kappa: float = 1.0
    """threshold for the Huber loss used in quantile regression"""

    # results tracking
    hypothesis_id: str = "h065"
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
    """Simple replay buffer with single-transition add (for n-step compatibility)."""
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

    def add_single(self, obs, next_obs, action, reward, done):
        """Add a single transition (used by n-step logic)."""
        self.observations[self.pos] = obs
        self.next_observations[self.pos] = next_obs
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done
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


class NStepBuffer:
    """Per-environment n-step transition buffer."""
    def __init__(self, num_envs, n_step, gamma):
        self.num_envs = num_envs
        self.n_step = n_step
        self.gamma = gamma
        self.buffers = [deque(maxlen=n_step) for _ in range(num_envs)]

    def _compute_nstep_return(self, transitions):
        R = 0.0
        done_any = False
        for i in reversed(range(len(transitions))):
            _, _, r, d = transitions[i]
            if d:
                done_any = True
            R = r + self.gamma * R * (1.0 - float(d))
        return R, done_any

    def add(self, env_idx, obs, action, reward, done, next_obs, replay_buffer):
        self.buffers[env_idx].append((obs, action, reward, done))

        if done:
            self._flush(env_idx, next_obs, replay_buffer)
        elif len(self.buffers[env_idx]) == self.n_step:
            transitions = list(self.buffers[env_idx])
            R, done_any = self._compute_nstep_return(transitions)
            obs_0 = transitions[0][0]
            action_0 = transitions[0][1]
            replay_buffer.add_single(obs_0, next_obs, action_0, R, float(done_any))

    def _flush(self, env_idx, final_next_obs, replay_buffer):
        buf = self.buffers[env_idx]
        while len(buf) > 0:
            transitions = list(buf)
            R, done_any = self._compute_nstep_return(transitions)
            obs_0 = transitions[0][0]
            action_0 = transitions[0][1]
            replay_buffer.add_single(obs_0, final_next_obs, action_0, R, float(done_any))
            buf.popleft()


class IQNNetwork(nn.Module):
    """Implicit Quantile Network."""
    def __init__(self, envs, cosine_embedding_dim=64):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        n_actions = envs.single_action_space.n
        self.n_actions = n_actions
        self.cosine_embedding_dim = cosine_embedding_dim

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

        self.register_buffer(
            "i_pi",
            torch.arange(1, cosine_embedding_dim + 1, dtype=torch.float32).unsqueeze(0) * torch.pi
        )

        self.cosine_linear = nn.Linear(cosine_embedding_dim, 512)
        self.value_head = nn.Linear(512, n_actions)

    def forward(self, x, n_quantiles, taus=None):
        batch_size = x.shape[0]
        features = self.encoder(x / 255.0)

        if taus is None:
            taus = torch.rand(batch_size, n_quantiles, device=x.device)

        tau_expanded = taus.unsqueeze(2)
        cos_embedding = torch.cos(tau_expanded * self.i_pi)
        tau_features = F.relu(self.cosine_linear(cos_embedding))

        combined = features.unsqueeze(1) * tau_features
        quantile_values = self.value_head(combined)

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

    q_network = IQNNetwork(envs, cosine_embedding_dim=args.cosine_embedding_dim).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = IQNNetwork(envs, cosine_embedding_dim=args.cosine_embedding_dim).to(device)
    target_network.load_state_dict(q_network.state_dict())

    rb = SimpleReplayBuffer(
        args.buffer_size,
        envs.single_observation_space.shape,
        device,
    )

    # N-step transition buffer
    nstep_buffer = NStepBuffer(args.num_envs, args.n_step, args.gamma)
    gamma_n = args.gamma ** args.n_step

    start_time = time.time()
    avg_returns = deque(maxlen=20)
    all_episode_returns = []

    obs = envs.reset()
    global_step = 0

    while global_step < args.total_timesteps:
        # Epsilon-greedy action selection with IQN mean Q
        epsilon = linear_schedule(args.start_e, args.end_e, args.exploration_fraction * args.total_timesteps, global_step)

        random_actions = np.random.randint(0, envs.single_action_space.n, size=args.num_envs)
        with torch.no_grad():
            quantile_values, _ = q_network(
                torch.tensor(obs, dtype=torch.float32).to(device),
                n_quantiles=args.n_quantiles_policy,
            )
            mean_q = quantile_values.mean(dim=1)
            greedy_actions = torch.argmax(mean_q, dim=1).cpu().numpy()

        explore_mask = np.random.random(args.num_envs) < epsilon
        actions = np.where(explore_mask, random_actions, greedy_actions)

        # Execute actions
        next_obs, rewards, dones, info = envs.step(actions)
        global_step += args.num_envs

        # Add transitions through n-step buffer
        for env_idx in range(args.num_envs):
            nstep_buffer.add(
                env_idx,
                obs[env_idx],
                actions[env_idx],
                rewards[env_idx],
                dones[env_idx],
                next_obs[env_idx],
                rb,
            )

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
                    online_qv, _ = q_network(s_next_obs, n_quantiles=N)
                    online_mean_q = online_qv.mean(dim=1)
                    next_actions = online_mean_q.argmax(dim=1)

                    # Target quantile values for selected actions
                    target_qv, target_taus = target_network(s_next_obs, n_quantiles=N)
                    next_actions_expanded = next_actions.unsqueeze(1).unsqueeze(2).expand(-1, N, 1)
                    target_q = target_qv.gather(2, next_actions_expanded).squeeze(2)

                    # N-step TD target: R_n + gamma^n * Q_target(s_n, a)
                    td_target = s_rewards.unsqueeze(1) + gamma_n * (1 - s_dones.unsqueeze(1)) * target_q

                # Current quantile estimates for taken actions
                current_qv, current_taus = q_network(s_obs, n_quantiles=N)
                s_actions_expanded = s_actions.unsqueeze(1).unsqueeze(2).expand(-1, N, 1)
                current_q = current_qv.gather(2, s_actions_expanded).squeeze(2)

                # Quantile Huber Loss
                pred = current_q.unsqueeze(2)
                target = td_target.unsqueeze(1)
                delta = target - pred

                abs_delta = delta.abs()
                huber = torch.where(
                    abs_delta <= kappa,
                    0.5 * delta ** 2,
                    kappa * (abs_delta - 0.5 * kappa),
                )

                taus_for_loss = current_taus.unsqueeze(2)
                quantile_weight = torch.abs(taus_for_loss - (delta < 0).float())
                quantile_huber_loss = quantile_weight * huber / kappa

                loss = quantile_huber_loss.sum(dim=2).mean()

                if (global_step // args.num_envs) % 100 == 0:
                    writer.add_scalar("losses/quantile_loss", loss.item(), global_step)
                    writer.add_scalar("losses/mean_q_values", current_q.mean().item(), global_step)
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
            w.writerow([args.env_id, args.seed, args.hypothesis_id, args.experiment_id, "iqn_nstep",
                         args.total_timesteps, n_episodes, mean_return, q4_metric, auc_metric, final_avg])
        print(f"Results saved to {csv_path}")

    envs.close()
    writer.close()
