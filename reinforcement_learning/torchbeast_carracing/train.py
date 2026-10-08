# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# R2AC (real-time recurrent actor-critic) on CarRacing-v3.
# Synchronous single-process version of torchbeast_procgen.polybeast_learner
# (`learn_rtrl`): no libtorchbeast/gRPC, the N environments of a gymnasium
# AsyncVectorEnv play the role of the actors, and every unroll of
# `unroll_length` steps x `num_envs` environments is one learner batch.

import argparse
import collections
import os
import time

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from torchbeast.core import vtrace
from torchbeast_carracing.carracing_wrappers import make_vector_env
from torchbeast_carracing.model import RTRLQuasiLSTMNet


def parse_args():
    parser = argparse.ArgumentParser(description="R2AC on CarRacing-v3")
    parser.add_argument("--savedir", type=str, required=True)
    parser.add_argument("--xpid", type=str, required=True)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_envs", type=int, default=16,
                        help="Number of parallel environments (= learner batch size).")
    parser.add_argument("--unroll_length", type=int, default=50)
    parser.add_argument("--total_steps", type=int, default=5_000_000,
                        help="Total environment steps (agent decisions, after action repeat).")
    parser.add_argument("--hidden_size", type=int, default=256)
    parser.add_argument("--learning_rate", type=float, default=0.0006)
    parser.add_argument("--alpha", type=float, default=0.99, help="RMSProp smoothing constant.")
    parser.add_argument("--momentum", type=float, default=0.0, help="RMSProp momentum.")
    parser.add_argument("--epsilon", type=float, default=0.01, help="RMSProp epsilon.")
    parser.add_argument("--grad_norm_clipping", type=float, default=40.0)
    parser.add_argument("--entropy_cost", type=float, default=0.01)
    parser.add_argument("--baseline_cost", type=float, default=0.5)
    parser.add_argument("--discounting", type=float, default=0.99)
    parser.add_argument("--log_every", type=int, default=10, help="Log every this many updates.")
    return parser.parse_args()


def compute_baseline_loss(advantages):
    return 0.5 * torch.sum(advantages ** 2)


def compute_entropy_loss(logits):
    """Return the entropy loss, i.e., the negative entropy of the policy."""
    policy = F.softmax(logits, dim=-1)
    log_policy = F.log_softmax(logits, dim=-1)
    return torch.sum(policy * log_policy)


def compute_policy_gradient_loss(logits, actions, advantages):
    cross_entropy = F.nll_loss(
        F.log_softmax(torch.flatten(logits, 0, 1), dim=-1),
        target=torch.flatten(actions, 0, 1),
        reduction="none",
    )
    cross_entropy = cross_entropy.view_as(advantages)
    return torch.sum(cross_entropy * advantages.detach())


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    assert torch.cuda.is_available()
    device = torch.device("cuda")

    xpdir = os.path.join(os.path.expanduser(args.savedir), args.xpid)
    os.makedirs(xpdir, exist_ok=True)
    with open(os.path.join(xpdir, "args.txt"), "w") as f:
        f.write(" ".join(f"--{k} {v}" for k, v in vars(args).items()) + "\n")
    episode_log = open(os.path.join(xpdir, "log_episode.tsv"), "w")
    episode_log.write("step\tepisode_return\tepisode_length\n")
    train_log = open(os.path.join(xpdir, "log_train.tsv"), "w")
    train_log.write("step\tsps\tmean_return_last20\ttotal_loss\tpg_loss\tbaseline_loss\tentropy_loss\n")

    B, T = args.num_envs, args.unroll_length
    envs = make_vector_env(B)
    obs, _ = envs.reset(seed=args.seed)

    model = RTRLQuasiLSTMNet(
        envs.single_action_space.n, args.hidden_size, envs.single_observation_space.shape
    ).to(device)

    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        eps=args.epsilon,
        alpha=args.alpha,
    )

    def lr_lambda(epoch):
        return 1 - min(epoch * T * B, args.total_steps) / args.total_steps

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # Torchbeast convention: reward[t] / done[t] are the reward and the episode
    # end that led to frame[t]; done[t] also means "reset the state before frame[t]".
    frame = torch.from_numpy(obs).to(device)
    reward = torch.zeros(B, device=device)
    done = torch.ones(B, dtype=torch.bool, device=device)
    rnn_state = model.initial_rnn_state(B, device)
    rtrl_state = model.initial_rtrl_state(B, device)

    episode_return = np.zeros(B)
    episode_length = np.zeros(B, dtype=np.int64)
    recent_returns = collections.deque(maxlen=20)

    step = 0
    num_updates = 0
    start_time = time.time()
    while step < args.total_steps:
        # 1. Rollout (T steps). The last frame of this unroll is the first of the next one.
        init_rnn_state = rnn_state
        frames, rewards, dones = [frame], [reward], [done]
        actions, behavior_logits = [], []
        for _ in range(T):
            action, logits, rnn_state = model.act(
                frame.unsqueeze(0), reward.unsqueeze(0), done.unsqueeze(0), rnn_state)
            obs, env_reward, terminated, truncated, _ = envs.step(action[0].cpu().numpy())
            env_done = terminated | truncated

            episode_return += env_reward
            episode_length += 1
            for i in np.flatnonzero(env_done):
                step_i = step + len(actions) * B + i + 1
                episode_log.write(f"{step_i}\t{episode_return[i]:.2f}\t{episode_length[i]}\n")
                recent_returns.append(episode_return[i])
            episode_return[env_done] = 0.0
            episode_length[env_done] = 0

            frame = torch.from_numpy(obs).to(device)
            reward = torch.from_numpy(env_reward).float().to(device)
            done = torch.from_numpy(env_done).to(device)
            frames.append(frame)
            rewards.append(reward)
            dones.append(done)
            actions.append(action[0])
            behavior_logits.append(logits[0])
        episode_log.flush()
        step += T * B

        frames = torch.stack(frames)  # (T + 1, B, C, H, W)
        rewards = torch.stack(rewards)  # (T + 1, B)
        dones = torch.stack(dones)  # (T + 1, B)
        actions = torch.stack(actions)  # (T, B)
        behavior_logits = torch.stack(behavior_logits)  # (T, B, A)

        # 2. Learner forward over the T + 1 frames, from the state before frame 0.
        core_input, notdone, core_output, rnn_tm1, z_outs, f_outs, _ = model.forward_core(
            frames, rewards, dones, init_rnn_state)
        core_output.retain_grad()  # top gradients dL / dc_t for RTRL
        policy_logits, baseline = model.heads(core_input, core_output)

        # Take final value function slice for bootstrapping.
        bootstrap_value = baseline[-1]

        discounts = (~dones[1:]).float() * args.discounting
        clipped_rewards = torch.clamp(rewards[1:], -1, 1)

        vtrace_returns = vtrace.from_logits(
            behavior_policy_logits=behavior_logits,
            target_policy_logits=policy_logits[:-1],
            actions=actions,
            discounts=discounts,
            rewards=clipped_rewards,
            values=baseline[:-1],
            bootstrap_value=bootstrap_value,
        )

        pg_loss = compute_policy_gradient_loss(
            policy_logits[:-1], actions, vtrace_returns.pg_advantages)
        baseline_loss = args.baseline_cost * compute_baseline_loss(
            vtrace_returns.vs - baseline[:-1])
        entropy_loss = args.entropy_cost * compute_entropy_loss(policy_logits[:-1])
        total_loss = pg_loss + baseline_loss + entropy_loss

        # 3. Backprop for everything outside the core, then RTRL for the core.
        optimizer.zero_grad()
        model.core.rtrl_zero_grad()
        total_loss.backward()

        # The RTRL state is advanced over frames 0..T-1 only: frame T is
        # frame 0 of the next unroll (its top gradient is zero anyway, as it
        # is only used for the detached bootstrap value).
        rtrl_state = model.compute_grad_rtrl(
            core_input[:T].detach(), notdone[:T], rnn_tm1[:T].detach(),
            z_outs[:T], f_outs[:T], rtrl_state, core_output.grad[:T])

        nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm_clipping)
        optimizer.step()
        scheduler.step()
        num_updates += 1

        if num_updates % args.log_every == 0:
            sps = step / (time.time() - start_time)
            mean_return = np.mean(recent_returns) if recent_returns else float("nan")
            train_log.write(
                f"{step}\t{sps:.1f}\t{mean_return:.2f}\t{total_loss.item():.4f}\t"
                f"{pg_loss.item():.4f}\t{baseline_loss.item():.4f}\t{entropy_loss.item():.4f}\n")
            train_log.flush()
            print(
                f"step {step:>9d} | sps {sps:6.1f} | return(last20) {mean_return:8.2f} | "
                f"loss {total_loss.item():9.4f} | pg {pg_loss.item():9.4f} | "
                f"baseline {baseline_loss.item():9.4f} | entropy {entropy_loss.item():8.4f}",
                flush=True)
            torch.save({"model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "step": step, "args": vars(args)},
                       os.path.join(xpdir, "model.tar"))

    envs.close()
    episode_log.close()
    train_log.close()


if __name__ == "__main__":
    main()
