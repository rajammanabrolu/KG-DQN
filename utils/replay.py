from collections import deque
import numpy as np
import random


class ReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)

        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, rho=0):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.concatenate(state), action, reward, np.concatenate(next_state), done

    def __len__(self):
        return len(self.buffer)


class PriorityReplayBuffer(object):
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
        self.priority_buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.expand_dims(state, 0)
        next_state = np.expand_dims(next_state, 0)
        #state = state.unsqueeze(0)
        #next_state = next_state.unsqueeze(0)

        if reward > 0:
            self.priority_buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size, rho):
        pbatch = int(batch_size * rho)
        batch = int(batch_size * (1 - rho))
        if pbatch > len(self.priority_buffer):
            pbatch = len(self.priority_buffer)
            batch = batch_size - len(self.priority_buffer)
        elif batch > len(self.buffer):
            batch = len(self.buffer)
            pbatch = batch_size - len(self.buffer)

        if pbatch == 0:
            state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch))

            return np.concatenate(list(state)), action, reward, np.concatenate(list(next_state)), done
        if batch == 0:
            pstate, paction, preward, pnext_state, pdone = zip(*random.sample(self.priority_buffer, pbatch))
            return np.concatenate(list(pstate)), paction, preward, np.concatenate(list(pnext_state)), pdone

        pstate, paction, preward, pnext_state, pdone = zip(*random.sample(self.priority_buffer, pbatch))
        pstate, paction, preward, pnext_state, pdone = np.concatenate(list(pstate)), paction, preward, np.concatenate(list(pnext_state)), pdone
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch))
        state, action, reward, next_state, done = np.concatenate(list(state)), action, reward, np.concatenate(list(next_state)), done

        return np.concatenate([pstate, state]), np.concatenate([paction, action]), np.concatenate([preward, reward]), np.concatenate([pnext_state, next_state]), np.concatenate([pdone, done])

    def __len__(self):
        return len(self.buffer)
