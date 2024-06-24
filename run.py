import numpy as np
from agent import Agent
from r2d2replaybuffer import r2d2_ReplayMemory
import torch
import gymnasium as gym
import time
import logger
import lahc
import os
import json
import logging
import copy

from envs.dynamic_crossing import DynamicCrossingEnv
from minigrid.core.world_object import Door, Goal, Lava, Wall, Ball, Key


class ActivationFunctionOptimizer(lahc.LateAcceptanceHillClimber):
    def __init__(self, initial_state, agent, env, candidate_activations, memory, args):
        super(ActivationFunctionOptimizer, self).__init__(
            initial_state=initial_state)
        self.agent = agent
        self.memory = memory
        self.args = args
        self.env = env
        self.candidate_activations = candidate_activations

    def move(self):
        self.state = np.random.randint(len(self.candidate_activations))
        self.agent.critic.update_activation_function(
            self.candidate_activations[self.state])
        self.agent.critic_target.update_activation_function(
            self.candidate_activations[self.state])

    def energy(self):
        total_reward = 0
        zeroshot = self.args["zero_shot"]
        episode_range = 5
        total_numsteps = 50000

        if zeroshot:
            for _ in range(episode_range):
                hidden_p = self.agent.get_initial_hidden()
                action = -1
                reward = 0
                state = self.env.reset()[0]["image"].astype(
                    np.float32).reshape(-1)

                while True:
                    action, hidden_p = self.agent.select_action(
                        state, action, reward, hidden_p, EPS_up=False, evaluate=True
                    )
                    next_state, reward, terminated, truncated, _ = self.env.step(
                        action)
                    total_reward += reward
                    state = next_state["image"].astype(np.float32).reshape(-1)
                    if terminated or truncated:
                        break
            avg_reward = total_reward / episode_range
            if avg_reward != 0:
                avg_reward = 1/avg_reward
            return avg_reward

        else:
            for _ in range(episode_range):
                agent_copy = self.agent
                memory_copy = self.memory
                numsteps = 0
                while numsteps <= total_numsteps:
                    hidden_p = agent_copy.get_initial_hidden()
                    action = -1  # placeholder
                    reward = 0
                    state = self.env.reset()[0]["image"].astype(
                        np.float32).reshape(-1)

                    ep_hiddens = [hidden_p]  # z[-1]
                    ep_actions = [action]  # a[-1]
                    ep_rewards = [reward]  # r[-1]
                    ep_states = [state]  # o[0]

                    while True:
                        # never used
                        action, hidden_p = agent_copy.select_action(
                            state,
                            action,
                            reward,
                            hidden_p,
                            EPS_up=True,
                            evaluate=False,
                        )
                        next_state, reward, terminated, truncated, _ = self.env.step(
                            action)  # Step

                        state = next_state["image"].astype(
                            np.float32).reshape(-1)
                        ep_hiddens.append(hidden_p)  # z[t]
                        ep_actions.append(action)  # a[t]
                        ep_rewards.append(reward)  # r[t]
                        ep_states.append(state)  # o[t+1]
                        total_reward += reward
                        if (
                            len(memory_copy) > self.args["batch_size"]
                            and numsteps % self.args["rl_update_every_n_steps"] == 0
                        ):
                            losses = agent_copy.update_parameters(
                                memory_copy, self.args["batch_size"], self.args["rl_updates_per_step"]
                            )
                        numsteps += 1
                        if terminated or truncated:
                            break
                memory_copy.push(
                    ep_states, ep_actions, ep_rewards, ep_hiddens)
            avg_reward = total_reward / episode_range
            if avg_reward != 0:
                avg_reward = 1/avg_reward
            return avg_reward


def run_exp(args):
    # Create env
    if args["env_name"] == "DynamicCrossingEnv":
        env = DynamicCrossingEnv(
            size=9,
            num_crossings=1,
            obstacle_type=Wall,
            num_dists=0,
            seed=args['seed'],
        )
        test_env = DynamicCrossingEnv(
            size=9,
            num_crossings=1,
            obstacle_type=Wall,
            num_dists=0,
            seed=args['seed'],
        )
    else:
        env = gym.make(args["env_name"])
        test_env = gym.make(args["env_name"])

    # only use image obs
    obs_dim = np.prod(env.observation_space["image"].shape)
    act_dim = env.action_space.n  # discrete action
    logger.log(
        env.observation_space["image"],
        f"obs_dim={obs_dim} act_dim={act_dim} max_steps={env.max_steps}",
    )

    # Initialize agent and buffer
    agent = Agent(env, args)
    memory = r2d2_ReplayMemory(args["replay_size"], obs_dim, act_dim, args)
    seed = args["seed"]
    np.random.seed(seed)
    torch.manual_seed(seed)
    env.reset(seed=seed)
    test_env.reset(seed=seed + 1)
    memory.reset(seed)

    # Training

    total_numsteps = 0
    while total_numsteps <= args["num_steps"]:
        hidden_p = agent.get_initial_hidden()
        action = -1  # placeholder
        reward = 0
        state = env.reset()[0]["image"].astype(np.float32).reshape(-1)

        ep_hiddens = [hidden_p]  # z[-1]
        ep_actions = [action]  # a[-1]
        ep_rewards = [reward]  # r[-1]
        ep_states = [state]  # o[0]

        while True:
            if total_numsteps % args["logging_freq"] == 0:
                if total_numsteps > 0:  # except the first evaluation
                    FPS = running_metrics["length"] / (time.time() - time_now)
                    # average the metrics
                    running_metrics = {
                        k: v / k_episode for k, v in running_metrics.items()
                    }
                    running_losses = {
                        k: v / k_updates for k, v in running_losses.items()
                    }
                log_and_test(
                    test_env,
                    agent,
                    total_numsteps,
                    running_metrics if total_numsteps > 0 else None,
                    running_losses if total_numsteps > 0 else None,
                    FPS if total_numsteps > 0 else None,
                )

                # running metrics
                k_episode = 0  # num of env episodes
                k_updates = 0  # num of agent updates
                running_metrics = {
                    k: 0.0
                    for k in [
                        "return",
                        "length",
                        "success",
                    ]
                }
                running_losses = {}
                time_now = time.time()

            if total_numsteps < args["random_actions_until"]:  # never used
                action = env.action_space.sample()
            else:
                action, hidden_p = agent.select_action(
                    state,
                    action,
                    reward,
                    hidden_p,
                    EPS_up=True,
                    evaluate=False,
                )

            next_state, reward, terminated, truncated, _ = env.step(
                action)  # Step
            state = next_state["image"].astype(np.float32).reshape(-1)

            ep_hiddens.append(hidden_p)  # z[t]
            ep_actions.append(action)  # a[t]
            ep_rewards.append(reward)  # r[t]
            ep_states.append(state)  # o[t+1]

            running_metrics["return"] += reward
            running_metrics["length"] += 1

            if (
                len(memory) > args["batch_size"]
                and total_numsteps % args["rl_update_every_n_steps"] == 0
            ):
                losses = agent.update_parameters(
                    memory, args["batch_size"], args["rl_updates_per_step"]
                )
                k_updates += 1
                if running_losses == {}:
                    running_losses = losses
                else:
                    running_losses = {
                        k: running_losses[k] + v for k, v in losses.items()
                    }

            if isinstance(env, DynamicCrossingEnv):
                if total_numsteps == args["env_change"] and total_numsteps > 0:
                    print(f"Increasing_size at {total_numsteps} steps")
                    if args["env_name"] == "DynamicCrossingEnv":
                        env = DynamicCrossingEnv(
                            size=11,
                            num_crossings=1,
                            obstacle_type=Wall,
                            num_dists=0,
                            seed=args['seed'],
                        )
                        test_env = DynamicCrossingEnv(
                            size=11,
                            num_crossings=1,
                            obstacle_type=Wall,
                            num_dists=0,
                            seed=args['seed'],
                        )
                    else:
                        env = gym.make(args["env_name"],
                                       size=11)
                        test_env = gym.make(
                            args["env_name"], size=11)
                    env.reset(seed=seed)
                    test_env.reset(seed=seed+1)

            if total_numsteps == args["lahc"]:
                activation_functions = [
                    'softplus-softplus',
                    'gelu-gelu',
                    'elu-elu',
                    'hardswish-hardswish',
                    'tanh-tanh'
                ]

                initial_state = np.random.randint(len(activation_functions))
                lahc.LateAcceptanceHillClimber.steps_minimum = 20
                lahc.LateAcceptanceHillClimber.history_length = 3
                lahc.LateAcceptanceHillClimber.updates_every = 1
                optimizer = ActivationFunctionOptimizer(
                    initial_state, agent, env, activation_functions, memory, args)
                optimizer.run()
                optimized = optimizer.best_state
                best_activation_function = activation_functions[optimized]

                agent.critic.update_activation_function(
                    best_activation_function)
                agent.critic_target.update_activation_function(
                    best_activation_function)
                args['af'] = best_activation_function

            total_numsteps += 1

            if terminated or truncated:
                break

        # Append transition to memory
        memory.push(ep_states, ep_actions, ep_rewards, ep_hiddens)

        k_episode += 1
        running_metrics["success"] += int(reward > 0.0)  # terminal reward

    config_path = os.path.join(
        args['logdir'], "config_new.json")
    with open(config_path, "w") as fp:
        json.dump(args, fp, indent=4)


def log_and_test(
    env,
    agent,
    total_numsteps,
    running_metrics,
    running_losses,
    FPS,
):
    logger.record_step("env_steps", total_numsteps)
    if total_numsteps > 0:
        for k, v in running_metrics.items():
            logger.record_tabular("train/" + k, v)
        for k, v in running_losses.items():
            logger.record_tabular(k, v)
        logger.record_tabular("FPS", FPS)

    metrics = {
        k: 0.0
        for k in [
            "return",
            "length",
            "success",
        ]
    }
    episodes = 10
    for _ in range(episodes):
        hidden_p = agent.get_initial_hidden()
        action = -1  # placeholder
        reward = 0
        state = env.reset()[0]["image"].astype(np.float32).reshape(-1)

        while True:
            action, hidden_p = agent.select_action(
                state, action, reward, hidden_p, EPS_up=False, evaluate=True
            )
            next_state, reward, terminated, truncated, _ = env.step(action)
            metrics["return"] += reward
            metrics["length"] += 1

            state = next_state["image"].astype(np.float32).reshape(-1)

            if terminated or truncated:
                break

        metrics["success"] += int(reward > 0.0)

    metrics = {k: metrics[k] / episodes for k in metrics.keys()}
    for k, v in metrics.items():
        logger.record_tabular(k, v)
    logger.dump_tabular()
