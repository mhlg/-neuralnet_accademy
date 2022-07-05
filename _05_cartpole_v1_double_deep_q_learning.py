import os
import random
from datetime import datetime
from typing import Dict, List, Tuple

import gym
import joblib

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from replay_buffer import ReplayBuffer


class QNetwork(tf.keras.Model):
    def __init__(self, input_shape, num_actions, num_hidden) -> None:
        super(QNetwork, self).__init__()

        self.dense = [
            tf.keras.layers.Dense(
                num_hidden[0], activation="relu", input_shape=input_shape
            )
        ]

        if len(num_hidden) > 0:
            for n in num_hidden[1:]:
                self.dense.append(tf.keras.layers.Dense(n, activation="relu"))

        self.actions = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, state):
        x = state
        for l in self.dense:
            x = l(x)
        x = self.actions(x)
        return x


class DDQLearningAgent:
    def __init__(
        self,
        alpha: float,
        gamma: float,
        epsilon_decay: float,
        state_space,
        action_space,
        update_frequency: int,
        checkpoint_dir: str,
        replay_buffer_size: int,
    ):

        self.alpha = alpha  # learning rate.
        self.gamma = gamma  # discount rate.
        self.epsilon = 1.0  # exploration exploitation rate.
        self.epsilon_decay = epsilon_decay
        self.action_space = action_space
        self.state_space = state_space
        self.checkpoint_dir = checkpoint_dir
        self.replay_buffer = ReplayBuffer(
            size=replay_buffer_size, state_space=self.state_space.shape
        )
        self.update_frequency = update_frequency
        self.steps = 0

        self.network = QNetwork(
            input_shape=(None, *self.state_space.shape),
            num_actions=action_space.n,
            num_hidden=[128, 96],
        )
        self.network.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha)
        )

        self.target_network = QNetwork(
            input_shape=(None, *self.state_space.shape),
            num_actions=action_space.n,
            num_hidden=[128,96]
        )
        self.target_network.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.alpha)
        )

        self.copy_network_weights()

        self.loss = tf.keras.losses.MeanSquaredError()
        self.metric = tf.keras.metrics.Mean()

    def __getstate__(self) -> Dict:
        """This allows the class to be pickled
        No other piece of state should leak outside of these variables.
        """

        data = {}
        data["alpha"] = self.__dict__["alpha"]
        data["gamma"] = self.__dict__["gamma"]
        data["epsilon"] = self.__dict__["epsilon"]
        data["epsilon_decay"] = self.__dict__["epsilon_decay"]
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

    def save(self):
        path = self.checkpoint_dir
        joblib.dump(self, os.path.join(path, f"agent.pkl"))
        tf.keras.models.save_model(self.network, os.path.join(path, f"network"))
        tf.keras.models.save_model(
            self.target_network, os.path.join(path, f"target_network")
        )

    @staticmethod
    def load(path: str):
        data = joblib.load(os.path.join(path, "agent.pkl"))
        data.network = tf.keras.models.load_model(os.path.join(path, "network"))
        data.target_network = tf.keras.models.load_model(
            os.path.join(path, "target_network")
        )
        return data

    def select_action(self, state) -> int:
        if random.random() < self.epsilon:
            action = self.action_space.sample()
        else:
            Q_values = self.network(
                tf.expand_dims(tf.convert_to_tensor(state), 0)
            ).numpy()[0]
            action = int(np.argmax(Q_values))

        return action

    def copy_network_weights(self):
        self.target_network.set_weights(self.network.get_weights())

    def decrement_epsilon(self) -> None:
        self.epsilon = max(self.epsilon - self.epsilon_decay, 0.01)

    def learn(self, batch_size: int):

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(
            batch_size
        )

        indices = tf.range(batch_size, dtype=tf.int32)
        actions = tf.convert_to_tensor(actions)
        actions_indices = tf.stack([indices, actions], axis=1)

        with tf.GradientTape() as tape:
            # compute Q(s, a) for all a in the action space.
            q_pred = self.network(tf.convert_to_tensor(states))
            # compute Q(s,a) by selecting the Q value for the action taken during simulation.
            q_pred = tf.gather_nd(q_pred, actions_indices)

            # in DDQL the Q(s', a') for all a in the action space
            # is computed by the target newtwork like in DQL but
            # the action a' is selected according tho the
            # maximum Q value of the regular network for s'.
            _q_next = self.network(tf.convert_to_tensor(next_states))
            max_actions = tf.argmax(_q_next, axis=1, output_type=tf.int32)
            max_actions_indices = tf.stack([indices, max_actions], axis=1)

            q_next = self.target_network(tf.convert_to_tensor(next_states))
            max_q_next = tf.gather_nd(q_next, max_actions_indices)

            is_terminal = 1 - tf.convert_to_tensor(dones, dtype=tf.float32)

            q_target = tf.convert_to_tensor(rewards) + self.gamma * (
                max_q_next * is_terminal
            )

            y_true = q_pred
            y_pred = q_target
            loss_value = self.loss(y_true, y_pred)

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
    algo_name = "DDQL"
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

    n_episodes = 10000
    replay_buffer_size = 10000
    alpha = 0.0001
    gamma = 0.99
    epsilon_decay = 1e-5
    update_frequency = 1000

    agent = DDQLearningAgent(
        alpha=alpha,
        gamma=gamma,
        epsilon_decay=epsilon_decay,
        update_frequency=update_frequency,
        state_space=env.observation_space,
        action_space=env.action_space,
        checkpoint_dir=checkpoint_dir,
        replay_buffer_size=replay_buffer_size,
    )

    score_history = []
    returns = []
    batch_size = 32
    for i in tqdm(range(n_episodes)):
        done = False
        state = env.reset()
        # env.render()
        rewards = []
        while not done:
            # select a random action.
            action = agent.select_action(state)
            next_state, reward, done, info = env.step(action=action)
            # env.render()
            rewards.append(reward)
            agent.replay_buffer.append(state, action, reward, next_state, done)
            state = next_state
            if agent.replay_buffer.population_size > batch_size:
                loss = agent.learn(batch_size)
                with tensorboard_writer.as_default():
                    tf.summary.scalar("epsilon", agent.epsilon, step=agent.steps)
                    tf.summary.scalar("loss", loss, step=agent.steps)
                # decrement epsilon.
                agent.decrement_epsilon()

        returns.append(sum(rewards))
        with tensorboard_writer.as_default():
            tf.summary.scalar("reward", data=returns[-1], step=i)
            tf.summary.scalar("steps", data=len(rewards), step=i)

        # save checkpoint.
        if i > 0 and i % 100 == 0:
            agent.save()
