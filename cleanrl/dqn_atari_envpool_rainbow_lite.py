# Rainbow-lite DQN for Atari using envpool
# Combines the two best orthogonal DQN improvements:
#   1. NoisyNet (Fortunato et al. 2018) — parametric exploration, replaces epsilon-greedy
#   2. N-step returns (Sutton 1988, Rainbow paper n=3) — better credit assignment
# These are complementary: N-step wins on BattleZone/Breakout/Enduro/SpaceInvaders,
# NoisyNet wins on Amidar/MsPacman/Phoenix/Solaris.
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
    learning_starts: int = 80000
    """timestep to start learning"""
    train_frequency: int = 4
    """the frequency of training (in terms of env steps, not global steps)"""
    n_step: int = 3
    """number of steps for multi-step returns (Rainbow paper default: 3)"""

    # results tracking
    hypothesis_id: str = "h064"
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
    """Per-environment n-step transition buffer.

    For each environment, maintains a deque of (obs, action, reward, done) tuples.
    When the deque reaches n_step length or an episode ends, computes the
    discounted n-step return and emits a transition to the replay buffer.
    """
    def __init__(self, num_envs, n_step, gamma):
        self.num_envs = num_envs
        self.n_step = n_step
        self.gamma = gamma
        self.buffers = [deque(maxlen=n_step) for _ in range(num_envs)]

    def _compute_nstep_return(self, transitions):
        """Compute discounted n-step return from a list of (obs, action, reward, done) tuples."""
        R = 0.0
        done_any = False
        for i in reversed(range(len(transitions))):
            _, _, r, d = transitions[i]
            if d:
                done_any = True
            R = r + self.gamma * R * (1.0 - float(d))
        return R, done_any

    def add(self, env_idx, obs, action, reward, done, next_obs, replay_buffer):
        """Add a transition for a specific environment and emit n-step transitions when ready."""
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
        """Flush all remaining transitions in the deque for an environment."""
        buf = self.buffers[env_idx]
        while len(buf) > 0:
            transitions = list(buf)
            R, done_any = self._compute_nstep_return(transitions)
            obs_0 = transitions[0][0]
            action_0 = transitions[0][1]
            replay_buffer.add_single(obs_0, final_next_obs, action_0, R, float(done_any))
            buf.popleft()


class NoisyLinear(nn.Module):
    """Factorized Gaussian NoisyLinear layer (Fortunato et al. 2018)."""

    def __init__(self, in_features: int, out_features: int, sigma_init: float = 0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.sigma_init = sigma_init

        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))

        self._init_parameters()
        self.reset_noise()

    def _init_parameters(self):
        bound = 1.0 / self.in_features ** 0.5
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma_init / self.in_features ** 0.5)
        nn.init.constant_(self.bias_sigma, self.sigma_init / self.out_features ** 0.5)

    @staticmethod
    def _f(x: torch.Tensor) -> torch.Tensor:
        """Factorized noise transformation: sign(x) * sqrt(|x|)."""
        return x.sign() * x.abs().sqrt()

    def reset_noise(self):
        """Resample factorized Gaussian noise for weights and biases."""
        eps_i = self._f(torch.randn(self.in_features, device=self.weight_epsilon.device))
        eps_j = self._f(torch.randn(self.out_features, device=self.weight_epsilon.device))
        self.weight_epsilon.copy_(eps_j.outer(eps_i))
        self.bias_epsilon.copy_(eps_j)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class QNetwork(nn.Module):
    def __init__(self, envs):
        super().__init__()
        obs_shape = envs.single_observation_space.shape
        n_actions = envs.single_action_space.n

        # CNN encoder (regular Conv2d layers)
        self.conv = nn.Sequential(
            nn.Conv2d(obs_shape[0], 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Noisy fully-connected layers (NoisyNet exploration)
        self.noisy_fc1 = NoisyLinear(3136, 512)
        self.noisy_fc2 = NoisyLinear(512, n_actions)

    def forward(self, x):
        x = self.conv(x / 255.0)
        x = F.relu(self.noisy_fc1(x))
        return self.noisy_fc2(x)

    def reset_noise(self):
        """Reset noise on all NoisyLinear layers."""
        self.noisy_fc1.reset_noise()
        self.noisy_fc2.reset_noise()


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
    print(f"[DIAG] device={device}, cuda_available={torch.cuda.is_available()}", flush=True)

    # env setup
    print(f"[DIAG] Creating envpool env: {args.env_id}, num_envs={args.num_envs}, seed={args.seed}", flush=True)
    envs = envpool.make(
        args.env_id,
        env_type="gym",
        num_envs=args.num_envs,
        episodic_life=True,
        reward_clip=True,
        seed=args.seed,
    )
    print(f"[DIAG] envpool.make() done", flush=True)
    envs.num_envs = args.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)
    assert isinstance(envs.action_space, gym.spaces.Discrete), "only discrete action space is supported"

    print(f"[DIAG] Creating Q-network", flush=True)
    q_network = QNetwork(envs).to(device)
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)
    target_network = QNetwork(envs).to(device)
    target_network.load_state_dict(q_network.state_dict())
    print(f"[DIAG] Q-network created, allocating replay buffer ({args.buffer_size} transitions)", flush=True)

    rb = SimpleReplayBuffer(
        args.buffer_size,
        envs.single_observation_space.shape,
        device,
    )
    print(f"[DIAG] Replay buffer allocated", flush=True)

    # N-step transition buffer
    nstep_buffer = NStepBuffer(args.num_envs, args.n_step, args.gamma)
    gamma_n = args.gamma ** args.n_step

    start_time = time.time()
    avg_returns = deque(maxlen=20)
    all_episode_returns = []

    # TRY NOT TO MODIFY: start the game
    print(f"[DIAG] Calling envs.reset()", flush=True)
    obs = envs.reset()
    print(f"[DIAG] envs.reset() done, starting training loop", flush=True)
    global_step = 0

    while global_step < args.total_timesteps:
        # NoisyNet greedy action selection (noise provides exploration — no epsilon needed)
        q_network.train()
        with torch.no_grad():
            q_values = q_network(torch.tensor(obs, dtype=torch.float32).to(device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        # Execute the game and log data
        next_obs, rewards, dones, info = envs.step(actions)
        global_step += args.num_envs

        # Add transitions through n-step buffer (emits to replay buffer when ready)
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

        # ALGO LOGIC: training
        if global_step > args.learning_starts:
            if (global_step // args.num_envs) % args.train_frequency == 0:
                s_obs, s_next_obs, s_actions, s_rewards, s_dones = rb.sample(args.batch_size)
                with torch.no_grad():
                    target_max, _ = target_network(s_next_obs).max(dim=1)
                    # N-step TD target: R_n + gamma^n * max_a Q_target(s_n, a) * (1 - done)
                    td_target = s_rewards + gamma_n * target_max * (1 - s_dones)
                q_network.train()
                old_val = q_network(s_obs).gather(1, s_actions.unsqueeze(1)).squeeze()
                loss = F.mse_loss(td_target, old_val)

                if (global_step // args.num_envs) % 100 == 0:
                    writer.add_scalar("losses/td_loss", loss, global_step)
                    writer.add_scalar("losses/q_values", old_val.mean().item(), global_step)
                    print("SPS:", int(global_step / (time.time() - start_time)))
                    writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

                # optimize the model
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # Reset noise after each gradient update (NoisyNet)
                q_network.reset_noise()
                target_network.reset_noise()

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
            w.writerow([args.env_id, args.seed, args.hypothesis_id, args.experiment_id, "dqn_rainbow_lite",
                         args.total_timesteps, n_episodes, mean_return, q4_metric, auc_metric, final_avg])
        print(f"Results saved to {csv_path}")

    envs.close()
    writer.close()
