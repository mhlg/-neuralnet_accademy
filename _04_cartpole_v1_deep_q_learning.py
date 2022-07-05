from __future__ import annotations

import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import gym
import joblib
import numpy as np
import tensorflow as tf
from tqdm import tqdm
import math

from replay_buffer import ReplayBuffer


class DQNetwork(tf.keras.Model):
    def __init__(self, num_actions, num_hidden) -> None:
        super(DQNetwork, self).__init__()

        self.dense = tf.keras.layers.Dense(
            num_hidden, activation="relu"
        )
       
        self.actions = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, states):
        x = self.dense(states)
        x = self.actions(x)
        return x


class DQLearningAgent:
    def __init__(
        self,
        alpha: float,
        gamma: float,
        epsilon_range: Tuple[float],
        state_space,
        action_space,
        update_frequency: int,
        checkpoint_dir: str,
        replay_buffer_size: int,
    ):

        self.alpha = alpha  # learning rate.
        self.gamma = gamma  # discount rate.
        self.action_space = action_space
        self.state_space = state_space
        self.checkpoint_dir = checkpoint_dir
        self.replay_buffer = ReplayBuffer(
            size=replay_buffer_size, state_space=self.state_space.shape
        )
        self.update_frequency = update_frequency
        self.steps = 0  # number of calls to the learn function.
        self.epsilon_range = epsilon_range
        self.epsilon = self.epsilon_range[0]

        self.network = DQNetwork(
            num_actions=action_space.n,
            num_hidden=128,
        )
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha)
        )

        self.target_network = DQNetwork(
            num_actions=action_space.n,
            num_hidden=128,
        )
        self.target_network.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha)
        )

        # self.copy_network_weights()

        self.loss = tf.keras.losses.MeanSquaredError()

    def __getstate__(self) -> Dict:
        """This allows the class to be pickled
        No other piece of state should leak outside of these variables.
        """

        data = {}
        data["alpha"] = self.__dict__["alpha"]
        data["gamma"] = self.__dict__["gamma"]
        data["epsilon"] = self.__dict__["epsilon"]
        data["epsilon_range"] = self.__dict__["epsilon_range"]
        data["action_space"] = self.__dict__["action_space"]
        data["state_space"] = self.__dict__["state_space"]
        data["checkpoint_dir"] = self.__dict__["checkpoint_dir"]
        data["replay_buffer"] = self.__dict__["replay_buffer"]
        data["update_frequency"] = self.__dict__["update_frequency"]
        data["steps"] = self.__dict__["steps"]
        data["loss"] = self.__dict__["loss"]

        return data

    def __setstate__(self, state: dict) -> None:
        for k in iter(state):
            self.__setattr__(k, state[k])

    def save(self)->None:
        path = self.checkpoint_dir
        joblib.dump(self, os.path.join(path, f"agent.pkl"))
        tf.keras.models.save_model(
            self.network, os.path.join(path, f"network"))
        tf.keras.models.save_model(
            self.target_network, os.path.join(path, f"target_network")
        )

    @staticmethod
    def load(path: str)->DQLearningAgent:
        agent = joblib.load(os.path.join(path, "agent.pkl"))
        assert isinstance(agent, DQLearningAgent)
        agent.network = tf.keras.models.load_model(
            os.path.join(path, "network"))
        agent.target_network = tf.keras.models.load_model(
            os.path.join(path, "target_network")
        )
        return agent

    def select_action(self, state) -> int:
        if random.random() < self.epsilon:
            action = self.action_space.sample()
        else:
            Q_values = self.network(
                tf.expand_dims(tf.convert_to_tensor(state), 0)
            ).numpy()[0]
            action = int(np.argmax(Q_values))

        return action

    def decrement_epsilon(self, decay:float) -> None:
        self.epsilon = max(self.epsilon - decay, self.epsilon_range[1])

    def copy_network_weights(self):
        self.target_network.set_weights(self.network.get_weights())

    def learn(self, batch_size: int):
        with tf.GradientTape() as tape:
            states, actions, rewards, next_states, dones = self.replay_buffer.sample(
                batch_size)

            indices = tf.range(batch_size, dtype=tf.int32)
            actions = tf.convert_to_tensor(actions)
            actions_indices = tf.stack([indices, actions], axis=1)
            # compute Q(s, a) for all a in the action space.
            q_value = self.network(tf.convert_to_tensor(states))
            # compute Q(s,a) by selecting the Q value for the action taken during simulation.
            q_value = tf.gather_nd(q_value, actions_indices)
            # in DQL the Q(s', a') for all a in the action space.
            # is computed by the target newtwork.
            q_next = self.target_network(tf.convert_to_tensor(next_states))
            # select the action that maximize the Q value.
            q_next = tf.reduce_max(q_next, axis=1)
            is_terminal = 1.0 - tf.convert_to_tensor(dones, dtype=tf.float32)

            q_next = q_next * is_terminal
            q_target = tf.convert_to_tensor(rewards) + self.gamma * q_next

            loss_value = self.loss(q_value, q_target)
            grads = tape.gradient(loss_value, self.network.trainable_variables)
            self.network.optimizer.apply_gradients(
                zip(grads, self.network.trainable_variables)
            )

        # update target network weights.
        self.steps += 1
        if self.steps % self.update_frequency == 0:
            self.copy_network_weights()

        return loss_value


if __name__ == "__main__":
    now = datetime.now()
    env_name = "CartPole-v1"
    algo_name = "DQL"
    checkpoint_dir = os.path.join(
        "./runs", env_name, algo_name, datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
    )

    print(f"checkpoint_dir={checkpoint_dir}")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    # tensorboard
    tensorboard_log_dir = os.path.join(checkpoint_dir, "logs")
    print(f"tensorboard_log_dir={tensorboard_log_dir}")
    tensorboard_writer = tf.summary.create_file_writer(tensorboard_log_dir)

    env = gym.make(env_name)

    number_of_episodes = 10000
    alpha = 0.0001
    gamma = 0.99
    epsilon_decay = 1e-5
    update_frequency = 500
    batch_size = 128
    replay_buffer_size = 12800 
    epsilon_range = (1.0, 0.01)

    agent = DQLearningAgent(
        alpha=alpha,
        gamma=gamma,
        epsilon_range=epsilon_range,
        update_frequency=update_frequency,
        action_space=env.action_space,
        state_space=env.observation_space,
        checkpoint_dir=checkpoint_dir,
        replay_buffer_size=replay_buffer_size,
    )


    best = -math.inf
    episodic_return = []
    
    for i in tqdm(range(number_of_episodes)):
        done = False
        state = env.reset()
        rewards = []
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action=action)
            rewards.append(reward)
            agent.replay_buffer.append(state, action, reward, next_state, done)
            state = next_state
            if agent.replay_buffer.population_size > batch_size:
                loss = agent.learn(batch_size)
                with tensorboard_writer.as_default():
                    tf.summary.scalar("epsilon", 
                                      agent.epsilon, step=agent.steps)
                    tf.summary.scalar("loss", loss, step=agent.steps)
                agent.decrement_epsilon(epsilon_decay)

        episodic_return.append(sum(rewards))
        rolling_episodic_return = np.mean(episodic_return[-100:])
       
        with tensorboard_writer.as_default():
            tf.summary.scalar("episodic_return",
                              data=episodic_return[-1], step=i)
            tf.summary.scalar("rolling_episodic_return",
                              data=rolling_episodic_return, step=i)

       
        if rolling_episodic_return > best and agent.steps > 0:
            best = rolling_episodic_return
            agent.save()
