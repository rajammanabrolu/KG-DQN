import networkx as nx

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
import spacy

import logging
import textworld
import matplotlib.pyplot as plt

from representations import StateNAction
from utils.schedule import *
#from utils.priority_replay import PriorityReplayBuffer
#from utils.replay import ReplayBuffer

from utils.graph_replay import *

from models import KGDQN

import numpy as np
import itertools


class KGDQNTrainer(object):
    
    def __init__(self, game, params):
        self.num_episodes = params['num_episodes']
        self.state = StateNAction()

        self.update_freq = params['update_frequency']
        self.filename = 'kgdqn_' + '_'.join([str(v) for k, v in params.items() if 'file' not in str(k)])
        logging.basicConfig(filename='logs/' + self.filename + '.log', filemode='w')
        logging.warning("Parameters", params)

        self.env = textworld.start(game)
        self.params = params

        if params['replay_buffer_type'] == 'priority':
            self.replay_buffer = GraphPriorityReplayBuffer(params['replay_buffer_size'])
        elif params['replay_buffer_type'] == 'standard':
            self.replay_buffer = GraphReplayBuffer(params['replay_buffer_size'])

        params['vocab_size'] = len(self.state.vocab_drqa)

        self.model = KGDQN(params, self.state.all_actions).cuda()

        if self.params['preload_weights']:
            self.model = torch.load(self.params['preload_file'])['model']
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

        #priority fraction
        self.rho = params['rho']

        if params['scheduler_type'] == 'exponential':
            self.e_scheduler = ExponentialSchedule(self.num_frames, params['e_decay'], params['e_final'])
        elif params['scheduler_type'] == 'linear':
            self.e_scheduler = LinearSchedule(self.num_frames, params['e_final'])
        
    def plot(self, frame_idx, rewards, losses, completion_steps):
        fig = plt.figure(figsize=(20, 5))
        plt.subplot(131)
        plt.title('frame %s. avg reward: %s' % (frame_idx, np.mean(rewards[-10:])))
        plt.plot(rewards)
        plt.subplot(132)
        plt.title('frame %s. avg steps: %s' % (frame_idx, np.mean(completion_steps[-10:])))
        plt.plot(completion_steps)
        plt.subplot(133)
        plt.title('loss-kgdqn')
        plt.plot(losses)
        plt.figtext(0.5, 0.01, self.filename, wrap=True, horizontalalignment='center', fontsize=12)
        fig.savefig('plots/' + self.filename + '_' + str(frame_idx) + '.png')
        #plt.show()

    def compute_td_loss(self):
        state, action, reward, next_state, done = self.replay_buffer.sample(self.batch_size, self.rho)

        reward = torch.FloatTensor(reward).cuda()
        done = torch.FloatTensor(1 * done).cuda()
        action_t = torch.LongTensor(action).cuda()

        q_value = self.model.forward_td_init(state, action_t)[0][0]

        with torch.no_grad():
            #Loop through all feasible actions for fwd
            actions = torch.LongTensor([a.pruned_actions_rep for a in list(next_state)]).cuda()
            fwd_init, sts = self.model.forward_td_init(next_state, actions[:, 0, :])#.unsqueeze_(0)
            next_q_values = fwd_init[0].unsqueeze_(0)
            for i in range(1, actions.size(1)):
                act = actions[:, i, :]#.squeeze()
                sts = sts.new_tensor(sts.data)
                cat_q = self.model.forward_td(sts, next_state, act)[0].unsqueeze_(0)
                next_q_values = torch.cat((next_q_values, cat_q), dim=0)

            next_q_values = next_q_values.transpose(0, 1)

        next_q_value = next_q_values.max(1)[0]
        expected_q_value = reward + self.gamma * next_q_value * (1 - done)

        loss = (q_value - (expected_q_value.data)).pow(2).mean()
        # clipped_loss = loss.clamp(-1.0, 1.0)
        #loss = loss.clamp(-1.0, 1.0)
        # right_gradient = clipped_loss * -1.0

        # loss.backward(right_gradient.data.unsqueeze(1)[:, 0])
        loss.backward()

        self.optimizer.step()
        self.optimizer.zero_grad()

        return loss

    def train(self):
        total_frames = 0
        for e_idx in range(1, self.num_episodes + 1):
            print("Episode:", e_idx)
            logging.info("Episode:" + str(e_idx))
            state = self.env.reset()
            self.state.step(state.description, pruned=self.params['pruned'])
            self.model.train()
            # print(state)

            episode_reward = 0
            completion_steps = 0
            episode_done = False
            prev_action = None

            for frame_idx in range(1, self.num_frames + 1):
                epsilon = self.e_scheduler.value(total_frames)

                action, picked = self.model.act(self.state, epsilon)


                action_text = self.state.get_action_text(action)
                logging.info('-------')
                logging.info(self.state.visible_state)
                logging.info('picked:' + str(picked))
                logging.info(action_text)

                next_state, reward, done = self.env.step(action_text)
                #if next_state.intermediate_reward == 0:
                #    reward += -0.1
                #else:
                #    reward += next_state.intermediate_reward

                reward += next_state.intermediate_reward
                reward = max(-1.0, min(reward, 1.0))
                if reward != 0:
                    print(action_text, reward)

                logging.warning('--------')
                logging.warning(frame_idx)
                logging.warning(self.state.visible_state)
                logging.warning(action_text)
                logging.warning(reward)

                episode_reward += reward
                completion_steps += 1
                total_frames += 1

                if done:
                    logging.warning("Done")

                    self.all_rewards.append(episode_reward)
                    self.completion_steps.append(completion_steps)
                    episode_reward = 0
                    completion_steps = 0
                    break
                elif frame_idx == self.num_frames:

                    self.all_rewards.append(episode_reward)
                    self.completion_steps.append(completion_steps)
                    episode_reward = 0
                    completion_steps = 0

                state = self.state
                self.state.step(next_state.description, prev_action=prev_action, pruned=self.params['pruned'])
                prev_action = action_text
                self.replay_buffer.push(state, action, reward, self.state, done)

                if len(self.replay_buffer) > self.batch_size:
                    if frame_idx % self.update_freq == 0:
                        loss = self.compute_td_loss()
                        self.losses.append(loss.item())

                # """
            self.plot(e_idx, self.all_rewards, self.losses, self.completion_steps)
            if e_idx % (int(self.num_episodes / 500)) == 0:
                logging.info("Episode:" + str(e_idx))
                # self.plot(frame_idx, self.all_rewards, self.losses, self.completion_steps)
                parameters = {
                    'model': self.model,
                    'replay_buffer': self.replay_buffer,
                    'action_dict': self.state.all_actions,
                    'vocab_drqa': self.state.vocab_drqa,
                    'vocab_kge': self.state.vocab_kge,
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
            'action_dict': self.state.all_actions,
            'vocab_drqa': self.state.vocab_drqa,
            'vocab_kge': self.state.vocab_kge,
            'params': self.params,
            'stats': {
                'losses': self.losses,
                'rewards': self.all_rewards,
                'completion_steps': self.completion_steps
            }
        }
        torch.save(parameters, 'models/' + self.filename + '_final.pt')
        self.env.close()

