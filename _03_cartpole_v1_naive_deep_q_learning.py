from __future__ import annotations

import math
import os
import random
from datetime import datetime
from typing import Dict, Tuple

import gym
import joblib
import numpy as np
import tensorflow as tf
from tqdm import tqdm


class DQNetwork(tf.keras.Model):
    def __init__(self, num_actions, num_hidden) -> None:
        super(DQNetwork, self).__init__()
        self.dense = tf.keras.layers.Dense(num_hidden, activation="relu")
        self.actions = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, state):
        x = self.dense(state)
        x = self.actions(x)
        return x


class DQLearningAgent():

    def __init__(self, alpha: float, gamma: float,
                 epsilon_bounds: Tuple[float], action_space, checkpoint_dir: str) -> None:

        self.alpha = alpha  # learning rate.
        self.gamma = gamma  # discount rate.
        self.epsilon_bounds = epsilon_bounds  # (max, min).
        self.epsilon = epsilon_bounds[0]  # exploration exploitation rate.
        self.action_space = action_space
        self.checkpoint_dir = checkpoint_dir
        self.steps = 0 # track the number of call to learn function. 
        self.network = DQNetwork(num_actions=action_space.n,
                                 num_hidden=128)
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha))
        self.loss = tf.keras.losses.MeanSquaredError()

    def __getstate__(self) -> Dict:
        """This allows the class to be pickled
        No other piece of state should leak outside of these variables.
        """
        data = {}
        data["alpha"] = self.__dict__["alpha"]
        data["gamma"] = self.__dict__["gamma"]
        data["epsilon_bounds"] = self.__dict__["epsilon_bounds"]
        data["epsilon"] = self.__dict__["epsilon"]
        data["action_space"] = self.__dict__["action_space"]
        data["checkpoint_dir"] = self.__dict__["checkpoint_dir"]
        data["steps"] = self.__dict__["steps"]
        data["loss"] = self.__dict__["loss"]

        return data

    def __setstate__(self, state: dict) -> None:
        for k in iter(state):
            self.__setattr__(k, state[k])

    def save(self):
        path = self.checkpoint_dir
        joblib.dump(self, os.path.join(path, f"agent.pkl"))
        tf.keras.models.save_model(
            self.network, os.path.join(path, f"network"))

    @staticmethod
    def load(path: str) -> DQLearningAgent:
        agent = joblib.load(os.path.join(path, "agent.pkl"))
        assert isinstance(agent, DQLearningAgent)
        agent.network = tf.keras.models.load_model(
            os.path.join(path, "network"))
        return agent

    def select_action(self, state) -> int:
        if random.random() < self.epsilon:
            action = self.action_space.sample()
        else:
            Q_values = self.network(tf.expand_dims(
                tf.convert_to_tensor(state), 0)).numpy()[0]
            action = int(np.argmax(Q_values))

        return action

    def decrement_epsilon(self, decay: float) -> None:
        self.epsilon = max(self.epsilon - decay, self.epsilon_bounds[1])

    def learn(self, state, action, reward, next_state):

        indices = tf.range(1, dtype=tf.int32)
        actions = tf.expand_dims(tf.convert_to_tensor(action), 0)
        actions_indices = tf.stack([indices, actions], axis=1)

        with tf.GradientTape() as tape:
            q = tf.gather_nd(self.network(tf.expand_dims(tf.convert_to_tensor(state), 0)),
                             actions_indices)

            q_next = tf.reduce_max(self.network(tf.expand_dims(tf.convert_to_tensor(next_state), 0)),
                                   axis=1)

            q_target = tf.expand_dims(tf.convert_to_tensor(reward), 0) + self.gamma * q_next

            loss_value = self.loss(q, q_target)

            grads = tape.gradient(loss_value, self.network.trainable_variables)
            self.network.optimizer.apply_gradients(
                zip(grads, self.network.trainable_variables))

        self.steps += 1
        return loss_value.numpy()


if __name__ == '__main__':

    now = datetime.now()
    env_name = "CartPole-v1"
    algo_name = "NaiveDQL"
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

    agent = DQLearningAgent(alpha=alpha, 
                            gamma=gamma,
                            epsilon_bounds=(1, 0.01),
                            action_space=env.action_space, 
                            checkpoint_dir=checkpoint_dir)

    best_score = -math.inf

    episodic_return = []
    for i in tqdm(range(number_of_episodes)):
        done = False
        state = env.reset()
        rewards = []
        while not done:
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action=action)
            rewards.append(reward)
            loss_value = agent.learn(state, action, reward, next_state)

            with tensorboard_writer.as_default():
                tf.summary.scalar("epsilon", agent.epsilon, step=agent.steps)
                tf.summary.scalar("loss", loss_value, step=agent.steps)

            state = next_state
            agent.decrement_epsilon(epsilon_decay)

        episodic_return.append(sum(rewards))
        with tensorboard_writer.as_default():
            tf.summary.scalar("episodic_return",
                              data=episodic_return[-1], step=i)

        if i % 100 == 0:
            score = np.mean(episodic_return[-100:])
            with tensorboard_writer.as_default():
                tf.summary.scalar("average_score",
                                  data=score, step=i)
            print(f"Episode {i} average score {score:.2f}")
            if score > best_score:
                best_score = score
                agent.save()
