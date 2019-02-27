import math, random
import textworld
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F

from collections import deque
from nltk.tokenize import word_tokenize

#from matplotlib import use
#use('Agg')
import matplotlib.pyplot as plt

import logging

from utils.replay import *
from utils.schedule import *

#from memory_profiler import profile

USE_CUDA = torch.cuda.is_available()


class DQN(nn.Module):
    def __init__(self, num_inputs, num_actions, hidden_dim):
        super(DQN, self).__init__()

        self.num_inputs = num_inputs
        self.num_actions = num_actions

        self.layers = nn.Sequential(
            nn.Linear(num_inputs, hidden_dim[0]),
            nn.ReLU(),
            nn.Linear(hidden_dim[0], hidden_dim[1]),
            nn.ReLU(),
            nn.Linear(hidden_dim[1], num_actions)
        )

    def forward(self, x):
        return self.layers(x)

    def act(self, state, epsilon):
        if random.random() > epsilon:
            with torch.no_grad():
                state = (torch.FloatTensor(state).unsqueeze(0)).cuda()
            q_value = self.forward(state)
            action = q_value.max(1)[1].data[0]
        else:
            action = random.randrange(self.num_actions)
        return action


class DQNTrainer(object):
    #@profile()
    def __init__(self, game, params):
        self.num_episodes = params['num_episodes']
        self.update_freq = params['update_frequency']
        self.filename = 'dqn_' + '_'.join([str(v) for k, v in params.items()])
        logging.basicConfig(filename='logs/' + self.filename + '.log', level=logging.WARN, filemode='w')
        logging.warning("Parameters", params)

        self.game = game
        self.env = textworld.start(self.game)
        self.params = params

        if params['replay_buffer_type'] == 'priority':
            self.replay_buffer = PriorityReplayBuffer(params['replay_buffer_size'])
        elif params['replay_buffer_type'] == 'standard':
            self.replay_buffer = ReplayBuffer(params['replay_buffer_size'])

        self.vocab = self.load_vocab()
        self.all_actions = self.load_action_dictionary()

        self.model = DQN(len(self.vocab.items()), len(self.all_actions.items()), params['hidden_dims']).cuda()
        # model = nn.DataParallel(model)

        self.optimizer = optim.Adam(self.model.parameters(), lr=params['lr'])

        self.env.compute_intermediate_reward()
        self.env.activate_state_tracking()

        self.num_frames = params['num_frames']
        self.batch_size = params['batch_size']
        self.gamma = params['gamma']

        self.losses = []
        self.all_rewards = []
        self.completion_steps = []

        self.rho = params['rho']

        if params['scheduler_type'] == 'exponential':
            self.e_scheduler = ExponentialSchedule(self.num_frames, params['e_decay'], params['e_final'])
        elif params['scheduler_type'] == 'linear':
            self.e_scheduler = LinearSchedule(self.num_frames, params['e_final'])

    def load_vocab(self):
        vocab = eval(open('../w2id.txt', 'r').readline())
        return vocab

    def load_action_dictionary(self):
        all_actions = eval(open('../id2act.txt', 'r').readline())
        return all_actions

    def state_rep_generator(self, state_description):
        bag_of_words = np.zeros(len(self.vocab))

        for token in word_tokenize(state_description):
            if token not in self.vocab.keys():
                token = '<UNK>'

            bag_of_words[self.vocab[token]] += 1

        return bag_of_words  # torch.FloatTensor(bag_of_words).cuda()

    def plot(self, frame_idx, rewards, losses, completion_steps):
        fig = plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('frame %s. steps: %s' % (frame_idx, np.mean(completion_steps[-10:])))
        plt.plot(completion_steps)
        plt.subplot(133)
        plt.title('loss-dqn')
        plt.plot(losses)
        #txt = "Gamma:" + str(self.gamma) + ", Num Frames:" + str(self.num_frames) + ", E Decay:" + str(epsilon_decay)
        plt.figtext(0.5, 0.01, self.filename, wrap=True, horizontalalignment='center', fontsize=12)
        #plt.show()
        fig.savefig('plots/' + self.filename + '_' + str(frame_idx) + '.png')

    def compute_td_loss(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.rho)

        state = torch.FloatTensor(state).cuda()
        with torch.no_grad():
            next_state = torch.FloatTensor(next_state).cuda()
        action = torch.LongTensor(action).cuda()
        reward = torch.FloatTensor(reward).cuda()
        done = torch.FloatTensor(1 * done).cuda()

        q_values = self.model(state)
        next_q_values = self.model(next_state)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]

        expected_q_value = reward + self.gamma * next_q_value * (1 - done)
        #print(q_value.size())
        #print(done)
        #print(q_value, next_q_value, expected_q_value)
        loss = (q_value - (expected_q_value.data)).pow(2).mean()
        #clipped_loss = loss.clamp(-1.0, 1.0)
        loss = loss.clamp(-1.0, 1.0)
        #right_gradient = clipped_loss * -1.0
        #print(loss)

        self.optimizer.zero_grad()
        #loss.backward(right_gradient.data.unsqueeze(1)[:, 0])
        loss.backward()

        self.optimizer.step()

        return loss

    #@profile()
    def train(self):
        total_frames = 0
        for e_idx in range(1, self.num_episodes + 1):
            state = self.env.reset()
            state_text = state.description
            state_rep = self.state_rep_generator(state_text)
            episode_reward = 0
            completion_steps = 0
            episode_done = False


            for frame_idx in range(1, self.num_frames + 1):
                epsilon = self.e_scheduler.value(total_frames)
                action = self.model.act(state_rep, epsilon)

                action_text = self.all_actions[int(action)]
                logging.info('-------')
                logging.info(state_text)
                logging.info(action_text)

                next_state, reward, done = self.env.step(action_text)
                reward += next_state.intermediate_reward
                reward = max(-1.0, min(reward, 1.0))

                #if reward != 0:
                logging.warning('--------')
                logging.warning(frame_idx)
                logging.warning(state_text)
                #print(next_state_text)
                logging.warning(action_text)
                logging.warning(reward)

                #print(reward)

                next_state_text = next_state.description
                next_state_rep = self.state_rep_generator(next_state_text)

                self.replay_buffer.push(state_rep, action, reward, next_state_rep, done)

                state = next_state
                state_text = next_state_text
                state_rep = next_state_rep

                episode_reward += reward
                completion_steps += 1
                total_frames += 1

                if len(self.replay_buffer) > self.batch_size:
                    if frame_idx % self.update_freq == 0:
                        loss = self.compute_td_loss()
                        self.losses.append(loss.data[0])

                if done:
                    logging.warning("Done")
                    state = self.env.reset()
                    state_text = state.description
                    state_rep = self.state_rep_generator(state_text)
                    self.all_rewards.append(episode_reward)
                    self.completion_steps.append(completion_steps)
                    episode_reward = 0
                    completion_steps = 0
                    episode_done = True
                elif frame_idx == self.num_frames:

                    self.all_rewards.append(episode_reward)
                    self.completion_steps.append(completion_steps)
                    episode_reward = 0
                    completion_steps = 0

                if episode_done:
                    break

            if e_idx % (int(self.num_episodes / 10)) == 0:
                logging.info("Episode:" + str(e_idx))
                self.plot(e_idx, self.all_rewards, self.losses, self.completion_steps)
                self.plot(e_idx, self.all_rewards, self.losses, self.completion_steps)
                parameters = {
                    'model': self.model,
                    'replay_buffer': self.replay_buffer,
                    'action_dict': self.all_actions,
                    'vocab': self.vocab,
                    'params': self.params,
                    'stats': {
                        'losses': self.losses,
                        'rewards': self.all_rewards,
                        'completion_steps': self.completion_steps
                    }
                }
                torch.save(parameters, 'models/' + self.filename + '_' + str(e_idx) + '.pt')

        parameters = {
            'model': self.model,
            'replay_buffer': self.replay_buffer,
            'action_dict': self.all_actions,
            'vocab': self.vocab,
            'params': self.params,
            'stats': {
                'losses': self.losses,
                'rewards': self.all_rewards,
                'completion_steps': self.completion_steps
            }
        }
        torch.save(parameters, 'models/' + self.filename + '_final.pt')
        self.env.close()




