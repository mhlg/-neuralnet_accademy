from typing import Tuple
import numpy as np

class ReplayBuffer:
    def __init__(self, size: int, state_space: Tuple[int]) -> None:

        self._population_size = 0
        self._size = size
        self._states = np.zeros((size, *state_space), dtype=np.float32)
        self._actions = np.zeros(size, dtype=np.int32)
        self._rewards = np.zeros(size, dtype=np.float32)
        self._next_states = np.zeros((size, *state_space), dtype=np.float32)
        self._dones = np.zeros(size, dtype=np.uint8)

    @property
    def size(self) -> int:
        return self._size

    @property
    def is_full(self)->bool:
        return self._population_size == self._size

    @property
    def population_size(self) -> int:
        return self._population_size

    def append(self, state, action, reward, next_state, done):
        i = self._population_size % self._size
        self._actions[i] = action
        self._states[i] = state
        self._rewards[i] = reward
        self._next_states[i] = next_state
        self._dones[i] = done

        if self._population_size < self.size:
            self._population_size += 1

    def sample(self, size: int):
        if size > self._population_size:
            raise ValueError("sample size is greater than the population size.")
        indices = np.random.choice(
            self.population_size, size=size, replace=False, p=None
        )
        states = self._states[indices]
        actions = self._actions[indices]
        rewards = self._rewards[indices]
        next_states = self._next_states[indices]
        dones = self._dones[indices]

        return states, actions, rewards, next_states, dones
