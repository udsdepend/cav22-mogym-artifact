import sys
from torch import nn
import torch
from torch import optim
import torch.nn.functional as F
import random
from collections import namedtuple, deque
import matplotlib.pyplot as plt
import numpy as np
from rlmate.replay_buffer import ReplayBuffer as RB
from rlmate.time_analyzer import TimeAnalyzer as TA
import rlmate.util as util
from rlmate.result import Result
import importlib
import logging


class Agent:
    # defines both, the local and the target network
    # defines the optimizer. Mostly, Adam is used
    # initializes buffer and update_counter
    # initialize hyper-parameters of learning process
    def __init__(self, env, dqn_args):
        self.dqn_args = dqn_args
        # self.result = Result(self.dqn_args.hermes_name)
        self.seed = dqn_args.seed
        random.seed(dqn_args.seed)
        np.random.seed(dqn_args.seed)
        torch.manual_seed(dqn_args.seed)

        self.device = torch.device("cpu")
        device_string = "cpu"
        if self.dqn_args.gpu:
            if not torch.cuda.is_available():
                logging.warning("GPU option was set, but CUDA is not available")
            else:
                try:
                    device_string = "cuda:" + str(self.dqn_args.gpu_id)
                    self.device = torch.device(device_string)
                except:
                    logging.warning(
                        "Using "
                        + str(device_string)
                        + "did not work, does gpu_id exist?"
                    )
                    self.device_string = torch.device("cpu")
        logging.info("Device set to " + device_string)

        input_length = len(env.explorer.state_vector)
        network_weights = self.dqn_args.neural_network_weights
        network_weights[0] = input_length
        network_weights[-1] = len(env.available_actions)

        network_module = importlib.import_module(self.dqn_args.neural_network_file)
        if self.dqn_args.neural_network_weights == None:
            self.qnetwork_target = network_module.Network(self.device)
            self.qnetwork_local = network_module.Network(self.device)
        else:
            self.qnetwork_target = network_module.Network(
                self.device, self.dqn_args.neural_network_weights
            )
            self.qnetwork_local = network_module.Network(
                self.device, self.dqn_args.neural_network_weights
            )

        self.optimizer = optim.Adam(
            self.qnetwork_local.parameters(), lr=self.dqn_args.learning_rate,
        )
        state_length = env.explorer.num_features
        self.buffer = RB(
            self.dqn_args.buffer_size,
            1,
            [[state_length]],
            self.dqn_args.batch_size,
            seed=self.seed,
            device=self.device,
        )

        self.update_counter = 0
        self.step_counter = 0

        # hyperparameters of learning:
        self.n_episodes = dqn_args.num_episodes
        self.l_episodes = dqn_args.length_episodes
        self.eps_start = dqn_args.eps_start
        self.eps_end = dqn_args.eps_end
        self.eps_decay = dqn_args.eps_decay
        self.positive_reward = self.dqn_args.positive_reward
        self.negative_reward = self.dqn_args.negative_reward

        # store env
        self.env = env
        self.num_actions = len(env.available_actions)

        # variables for learning
        self.best_score = dqn_args.best_network_score

    # used after init to load an existing network
    def load(self):
        raise Exception("not implemented")

    # carries out one step of the agent
    def step(self, state, action, reward, next_state, done):
        # add the sample to the buffer
        state = np.array([state], dtype=object)
        next_state = np.array([next_state], dtype=object)

        self.buffer.add(state, action, reward, next_state, done)

        # increment update counter
        self.update_counter = (self.update_counter + 1) % self.dqn_args.update_every

        # if the update counter mets the requirement of UPDATE_EVERY,
        # sample and start the learning process
        if self.update_counter == 0:
            if (len(self.buffer)) > self.dqn_args.batch_size:
                samples = self.buffer.sample()
                self.learn(samples, self.dqn_args.gamma)

        self.step_counter += 1

    # act epsilon greedy according to the local netowrk
    def act(self, state, eps=0, available_actions=[]):
        state = torch.tensor(state).float().to(self.device)
        with torch.no_grad():
            action_values = self.qnetwork_local(state)

        if random.random() > eps:
            q_vals = action_values.cpu().numpy()
            for i, aa in enumerate(available_actions):
                if not aa:
                    q_vals[i] = -float("inf")

            res = np.argmax(q_vals)
        else:
            available_actions_ids = [
                index for index in range(self.num_actions) if available_actions[index]
            ]
            if len(available_actions_ids) != 0:
                res = random.choice(available_actions_ids)
            else:
                res = random.choice(range(self.num_actions))
        return res

    # learn method
    def learn(self, samples, gamma):
        states, actions, rewards, next_states, dones = samples

        states = states[0]
        next_states = next_states[0]
        # Implementation of dqn algorithm
        q_values_next_states = self.qnetwork_target.forward(next_states).max(dim=1)[0]
        targets = rewards + (gamma * (q_values_next_states) * (1 - dones))
        q_values = self.qnetwork_local.forward(states)

        actions = actions.view(actions.size()[0], 1)
        predictions = torch.gather(q_values, 1, actions).view(actions.size()[0])

        # calculate loss between targets and predictions
        loss = F.mse_loss(predictions, targets)

        # make backward step
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # perform a soft-update to the network
        for target_weight, local_weight in zip(
            self.qnetwork_target.parameters(), self.qnetwork_local.parameters()
        ):
            target_weight.data.copy_(
                self.dqn_args.tau * local_weight.data
                + (1.0 - self.dqn_args.tau) * target_weight.data
            )

    def save_network(self, post_sign=None):
        path = self.dqn_args.hermes_name
        if post_sign != None:
            path = path + "_" + post_sign

        util.save_network(self.qnetwork_local, path=path + ".pth")
        util.save_network_as_json(self.qnetwork_local, path=path + ".json")

    def train(self):

        # initialize arrays and values
        means = []
        scores_window = deque(maxlen=100)
        eps = self.eps_start

        # iterate for initialized number of episodes
        for i_episode in range(1, self.n_episodes + 1):
            # reset state and score

            # reset to a random state:
            state = self.env.reset()
            score = 0
            # make at most max_t steps
            states = []
            actions = []
            rewards = []
            dones = []

            for t in range(self.l_episodes):
                available_actions = self.env.available_actions
                action = self.act(state, eps, available_actions)
                next_state, reward, done, _ = self.env.step(
                    action
                )  # send the action to the environment and observe
                # extract to check with python version

                if reward < 0:
                    reward = self.negative_reward
                elif reward > 0:
                    reward = self.positive_reward
                score += reward * np.power(self.dqn_args.gamma, t)
                self.step(state, action, reward, next_state, done)

                states.append(state)
                actions.append(action)
                rewards.append(reward)
                dones.append(done)
                next_state = next_state
                state = next_state

                if done:
                    break

            # save the very last state
            states.append(next_state)

            if self.dqn_args.extract_all_states:
                f = open(self.dqn_args.hermes_name + ".states", "a")
                for position in self.env.path:
                    f.write(str(position[0]) + " " + str(position[1]) + ";")
                f.write("\n")
                f.close()
            scores_window.append(score)

            eps = max(self.eps_end, self.eps_decay * eps)

            if i_episode % self.dqn_args.checkpoint_episodes == (
                -1 % self.dqn_args.checkpoint_episodes
            ):

                score = np.mean(scores_window)
                means.append(score)

                # if current score is better, save the network weights and update best seen score
                if score > self.best_score:
                    self.best_score = score
                    self.save_network("best")

                print(
                    "\rEpisode {}\tAverage Score: {:.2f}\tBest Score: {:.2f}".format(
                        i_episode, score, self.best_score
                    )
                )

                f = open(self.dqn_args.hermes_name + ".scores", "a")
                for score in scores_window:
                    f.write(str(score) + "\n")
                f.close()

            if self.dqn_args.policy_extraction_frequency != 0 and (
                i_episode % self.dqn_args.policy_extraction_frequency
            ) == (-1 % self.dqn_args.policy_extraction_frequency):
                self.save_network(str(i_episode))

        self.save_network("end")

        if self.dqn_args.extract_replay_buffer:
            f = open(self.dqn_args.hermes_name + ".rpb_table", "w")
            for entry in self.lookup:
                f.write(str(self.lookup[entry]) + " " + str(entry) + "\n")
            f.close()

        return self
