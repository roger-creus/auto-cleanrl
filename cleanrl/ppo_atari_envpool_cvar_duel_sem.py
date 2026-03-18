# PPO + CVaR + Dueling + SEM + DrQ (h036)
# Triple combination: CVaR advantage (h029) + Dueling (h020) + Simplicial Embeddings (h030) + DrQ
# SEM constrains the shared representation via product-of-softmax groups before dueling split
# Dueling separates value (distributional CVaR quantiles) and advantage (policy logits) streams
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
from torch.distributions.categorical import Categorical
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
    learning_rate: float = 2.5e-4
    """the learning rate of the optimizer"""
    num_envs: int = 128
    """the number of parallel game environments"""
    num_steps: int = 32
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 4
    """the number of mini-batches"""
    update_epochs: int = 4
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.1
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.01
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Quantile Regression Value parameters
    num_quantiles: int = 32
    """number of quantile levels for the distributional value function"""
    huber_kappa: float = 1.0
    """threshold for quantile Huber loss (1.0 = standard Huber)"""

    # CVaR advantage parameters
    cvar_alpha_start: float = 0.25
    """initial CVaR alpha (fraction of lower quantiles to use). 0.25 = pessimistic baseline → optimistic advantages"""
    cvar_alpha_end: float = 1.0
    """final CVaR alpha (1.0 = use all quantiles = standard mean)"""

    # DrQ augmentation parameters
    aug_pad: int = 4
    """random shift padding (pixels) for DrQ-style augmentation"""

    # SEM parameters
    sem_groups: int = 32
    """number of softmax groups for simplicial embedding"""
    sem_vertices: int = 16
    """number of vertices per group (sem_groups * sem_vertices = hidden dim)"""
    sem_temperature: float = 0.1
    """temperature for per-group softmax"""

    # results tracking
    hypothesis_id: str = "h000"
    """hypothesis identifier for results tracking"""
    experiment_id: str = ""
    """experiment identifier (auto-generated if empty)"""
    output_dir: str = "/runs"
    """directory to save CSV results"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


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


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def quantile_huber_loss(quantile_values, target, taus, kappa=1.0):
    """Compute quantile Huber loss.

    Args:
        quantile_values: (batch, num_quantiles) predicted quantile values
        target: (batch,) target return values
        taus: (num_quantiles,) quantile levels in [0, 1]
        kappa: Huber loss threshold
    Returns:
        scalar loss
    """
    # target: (batch, 1) for broadcasting
    target = target.unsqueeze(1)  # (batch, 1)
    # quantile_values: (batch, num_quantiles)

    # TD errors: (batch, num_quantiles)
    td_error = target - quantile_values

    # Huber loss element
    huber = torch.where(
        td_error.abs() <= kappa,
        0.5 * td_error.pow(2),
        kappa * (td_error.abs() - 0.5 * kappa)
    )

    # Asymmetric weighting by quantile level
    # taus: (1, num_quantiles)
    taus = taus.unsqueeze(0)
    quantile_weight = torch.abs(taus - (td_error.detach() < 0).float())

    # Weighted Huber loss, mean over batch and quantiles
    loss = (quantile_weight * huber).mean()
    return loss


def random_shift(x, pad=4):
    """DrQ-style random shift augmentation using grid_sample for efficiency.

    Applies random +-pad pixel shifts independently to each sample in the batch.
    x shape: (batch, C, H, W) — raw uint8 pixel values (0-255) as float tensors.
    """
    n, c, h, w = x.shape
    shift_h = (torch.randint(0, 2 * pad + 1, (n, 1, 1), device=x.device).float() - pad) / (h / 2)
    shift_w = (torch.randint(0, 2 * pad + 1, (n, 1, 1), device=x.device).float() - pad) / (w / 2)
    grid_h = torch.linspace(-1, 1, h, device=x.device).view(1, h, 1).expand(n, h, w) + shift_h
    grid_w = torch.linspace(-1, 1, w, device=x.device).view(1, 1, w).expand(n, h, w) + shift_w
    grid = torch.stack([grid_w, grid_h], dim=-1)
    return torch.nn.functional.grid_sample(x, grid, padding_mode='zeros', align_corners=True)


class Agent(nn.Module):
    def __init__(self, envs, num_quantiles=32, sem_groups=32, sem_vertices=16, sem_temperature=0.1):
        super().__init__()
        self.num_quantiles = num_quantiles
        self.sem_groups = sem_groups
        self.sem_vertices = sem_vertices
        self.sem_temperature = sem_temperature
        sem_dim = sem_groups * sem_vertices  # 32*16 = 512

        self.cnn = nn.Sequential(
            layer_init(nn.Conv2d(4, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = layer_init(nn.Linear(64 * 7 * 7, sem_dim))
        n_actions = envs.single_action_space.n
        # Dueling: value stream outputs quantiles (distributional) from SEM features
        self.value_stream = nn.Sequential(
            layer_init(nn.Linear(sem_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, num_quantiles), std=1),
        )
        # Dueling: advantage stream outputs per-action advantages from SEM features
        self.advantage_stream = nn.Sequential(
            layer_init(nn.Linear(sem_dim, 256)),
            nn.ReLU(),
            layer_init(nn.Linear(256, n_actions), std=0.01),
        )

    def _sem_embed(self, x):
        """CNN → FC → SEM (product-of-softmax groups)."""
        h = self.cnn(x / 255.0)
        h = self.fc(h)
        h = h.view(-1, self.sem_groups, self.sem_vertices)
        h = torch.softmax(h / self.sem_temperature, dim=-1)
        h = h.view(-1, self.sem_groups * self.sem_vertices)
        return h

    def get_value(self, x, cvar_alpha=1.0):
        """Return CVaR value estimate from distributional value stream."""
        quantiles = self.value_stream(self._sem_embed(x))
        sorted_q, _ = torch.sort(quantiles, dim=-1)
        k = max(1, int(cvar_alpha * self.num_quantiles))
        return sorted_q[:, :k].mean(dim=-1, keepdim=True)

    def get_action_and_value(self, x, action=None, cvar_alpha=1.0):
        hidden = self._sem_embed(x)
        # Policy from advantage stream
        advantages = self.advantage_stream(hidden)
        probs = Categorical(logits=advantages)
        if action is None:
            action = probs.sample()
        # Value from distributional value stream with CVaR
        quantiles = self.value_stream(hidden)
        sorted_q, _ = torch.sort(quantiles, dim=-1)
        k = max(1, int(cvar_alpha * self.num_quantiles))
        value = sorted_q[:, :k].mean(dim=-1, keepdim=True)
        return action, probs.log_prob(action), probs.entropy(), value

    def get_action_value_and_quantiles(self, x, action=None):
        """Return action, log_prob, entropy, scalar value (mean), and full quantile predictions."""
        hidden = self._sem_embed(x)
        advantages = self.advantage_stream(hidden)
        probs = Categorical(logits=advantages)
        if action is None:
            action = probs.sample()
        quantiles = self.value_stream(hidden)
        value = quantiles.mean(dim=-1, keepdim=True)
        return action, probs.log_prob(action), probs.entropy(), value, quantiles


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
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

    # Fixed quantile levels (midpoints of uniform intervals)
    taus = torch.arange(0, args.num_quantiles, device=device, dtype=torch.float32)
    taus = (taus + 0.5) / args.num_quantiles  # midpoints: [0.5/N, 1.5/N, ..., (N-0.5)/N]

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

    agent = Agent(envs, num_quantiles=args.num_quantiles,
                  sem_groups=args.sem_groups, sem_vertices=args.sem_vertices,
                  sem_temperature=args.sem_temperature).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)
    avg_returns = deque(maxlen=20)
    all_episode_returns = []

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs = torch.Tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        # Anneal CVaR alpha: start pessimistic (0.25), end at mean (1.0)
        progress = (iteration - 1.0) / args.num_iterations
        cvar_alpha = args.cvar_alpha_start + progress * (args.cvar_alpha_end - args.cvar_alpha_start)

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic — use CVaR value for advantage estimation
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, cvar_alpha=cvar_alpha)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            next_obs, reward, next_done, info = envs.step(action.cpu().numpy())
            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            # Collect all completed episodes at this step
            step_returns = []
            step_lengths = []
            for idx, d in enumerate(next_done):
                if d and info["lives"][idx] == 0:
                    step_returns.append(info["r"][idx])
                    step_lengths.append(info["l"][idx])
            if step_returns:
                mean_ret = np.mean(step_returns)
                mean_len = np.mean(step_lengths)
                for r in step_returns:
                    avg_returns.append(r)
                    all_episode_returns.append(r)
                print(f"global_step={global_step}, episodic_return={mean_ret:.1f} (n={len(step_returns)})")
                writer.add_scalar("charts/avg_episodic_return", np.average(avg_returns), global_step)
                writer.add_scalar("charts/episodic_return", mean_ret, global_step)
                writer.add_scalar("charts/episodic_length", mean_len, global_step)

        # bootstrap value if not done — use CVaR for consistent advantage estimation
        with torch.no_grad():
            next_value = agent.get_value(next_obs, cvar_alpha=cvar_alpha).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                # DrQ-style augmentation: apply random shift to observations during training only
                mb_obs_aug = random_shift(b_obs[mb_inds], pad=args.aug_pad)

                _, newlogprob, entropy, newvalue, new_quantiles = agent.get_action_value_and_quantiles(
                    mb_obs_aug, b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss (standard PPO clipping)
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss: quantile Huber regression
                # new_quantiles: (mb, num_quantiles), b_returns[mb_inds]: (mb,)
                v_loss = quantile_huber_loss(new_quantiles, b_returns[mb_inds], taus, kappa=args.huber_kappa)

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/cvar_alpha", cvar_alpha, global_step)
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

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
            w.writerow([args.env_id, args.seed, args.hypothesis_id, args.experiment_id, "ppo_cvar_duel_sem",
                         args.total_timesteps, n_episodes, mean_return, q4_metric, auc_metric, final_avg])
        print(f"Results saved to {csv_path}")

    envs.close()
    writer.close()
