import networkx as nx
import requests
from nltk import sent_tokenize, word_tokenize
import json
import numpy as np
import re
import matplotlib.pyplot as plt
import itertools
import random


def call_stanford_openie(sentence):
    url = "http://localhost:9000/"
    querystring = {
        "properties": "%7B%22annotators%22%3A%20%22openie%22%7D",
        "pipelineLanguage": "en"}
    response = requests.request("POST", url, data=sentence, params=querystring)
    response = json.JSONDecoder().decode(response.text)
    return response


class StateNAction(object):
    
    def __init__(self):
        self.graph_state = nx.DiGraph()

        self.graph_state_rep = []
        self.visible_state = ""
        self.drqa_input = ""
        self.vis_pruned_actions = []
        self.pruned_actions_rep = []
        self.vocab_drqa = self.load_vocab()
        self.rev_vocab_drqa = {v: k for k, v in self.vocab_drqa.items()}
        self.all_actions = self.load_action_dictionary()

        self.vocab_kge = self.load_vocab_kge()
        self.adj_matrix = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
        self.all_actions_rep = [self.get_action_rep_drqa(x) for x in list(self.all_actions.keys())]
        self.room = ""

    def visualize(self):
        pos = nx.spring_layout(self.graph_state)
        edge_labels = {e: self.graph_state.edges[e]['rel'] for e in self.graph_state.edges}
        nx.draw_networkx_edge_labels(self.graph_state, pos, edge_labels)
        nx.draw(self.graph_state, pos=pos, with_labels=True, node_size=200, font_size=10)
        plt.show()

    def load_vocab_kge(self):
        ent = {}
        with open('initialize/state/entity2id.tsv', 'r') as f:
            for line in f:
                e, eid = line.split('\t')
                ent[e.strip()] = int(eid.strip())
        rel = {}
        with open('initialize/state/relation2id.tsv', 'r') as f:
            for line in f:
                r, rid = line.split('\t')
                rel[r.strip()] = int(rid.strip())

        return {'entity': ent, 'relation': rel}

    def load_vocab(self):
        vocab = eval(open('../w2id.txt', 'r').readline())
        return vocab

    def load_action_dictionary(self):
        all_actions = eval(open('../act2id.txt', 'r').readline())
        return all_actions

    def update_state_base(self, visible_state):
        visible_state = visible_state.split('-')
        if len(visible_state) > 1:
            visible_state = visible_state[2]
        self.visible_state = visible_state
        try:
            sents = call_stanford_openie(self.visible_state)['sentences']
            for ov in sents:
                triple = ov['openie']
                for tr in triple:
                    h, r, t = tr['subject'], tr['relation'], tr['object']
                    self.graph_state.add_edge(h, t, rel=r)

        except:
            print(self.visible_state)
        return

    def update_state(self, visible_state, prev_action=None):
        remove = []
        prev_remove = []
        link = []
        visible_state = visible_state.split('-')
        if len(visible_state) > 1:
            visible_state = visible_state[2]
        dirs = ['north', 'south', 'east', 'west']

        self.visible_state = str(visible_state)
        rules = []
        
        sents = call_stanford_openie(self.visible_state)['sentences']

        for ov in sents:
            triple = ov['openie']
            for tr in triple:
                h, r, t = tr['subject'].lower(), tr['relation'].lower(), tr['object'].lower()

                if h == 'we':
                    h = 'you'
                    if r == 'are in':
                        r = "'ve entered"

                if h == 'it':
                    break
                rules.append((h, r, t))

        room = ""
        room_set = False
        for rule in rules:
            h, r, t = rule
            if 'entered' in r or 'are in' in r:
                prev_remove.append(r)
                if not room_set:
                    room = t
                    room_set = True
            if 'should' in r:
                prev_remove.append(r)
            if 'see' in r or 'make out' in r:
                link.append((r, t))
                remove.append(r)
            #else:
            #    link.append((r, t))

        prev_room = self.room
        self.room = room
        add_rules = []
        if prev_action is not None:
            for d in dirs:
                if d in prev_action and self.room != "":
                    add_rules.append((prev_room, d + ' of', room))

        prev_room_subgraph = None
        prev_you_subgraph = None

        for sent in sent_tokenize(self.visible_state):
            if 'exit' in sent or 'entranceway' in sent:
                for d in dirs:
                    if d in sent:
                        rules.append((self.room, 'has', 'exit to ' + d))
                    if prev_room != "":
                        graph_copy = self.graph_state.copy()
                        graph_copy.remove_edge('you', prev_room)
                        con_cs = [graph_copy.subgraph(c) for c in nx.weakly_connected_components(graph_copy)]

                        for con_c in con_cs:
                            if prev_room in con_c.nodes:
                                prev_room_subgraph = nx.induced_subgraph(graph_copy, con_c.nodes)
                            if 'you' in con_c.nodes:
                                prev_you_subgraph = nx.induced_subgraph(graph_copy, con_c.nodes)

        for l in link:
            add_rules.append((room, l[0], l[1]))

        for rule in rules:
            h, r, t = rule
            if r not in remove:
                add_rules.append(rule)
        edges = list(self.graph_state.edges)
        print("add", add_rules)
        for edge in edges:
            r = self.graph_state[edge[0]][edge[1]]['rel']
            if r in prev_remove:
                self.graph_state.remove_edge(*edge)
                
        if prev_you_subgraph is not None:
            self.graph_state.remove_edges_from(prev_you_subgraph.edges)
        
        for rule in add_rules:
            u = '_'.join(str(rule[0]).split())
            v = '_'.join(str(rule[2]).split())
            if u in self.vocab_kge['entity'].keys() and v in self.vocab_kge['entity'].keys():
                if u != 'it' and v != 'it':
                    self.graph_state.add_edge(rule[0], rule[2], rel=rule[1])
        print("pre", self.graph_state.edges)
        if prev_room_subgraph is not None:
            self.graph_state.add_edges_from(prev_room_subgraph.edges)
        print(self.graph_state.edges)

        return
    
    def get_state_rep_kge(self):
        ret = []
        self.adj_matrix = np.zeros((len(self.vocab_kge['entity']), len(self.vocab_kge['entity'])))
        #adj = []
        #for g in self.graph_state.nodes:
        #    ret.append(self.vocab_kge['entity']['_'.join(str(g).split())])

        for u, v in self.graph_state.edges:
            u = '_'.join(str(u).split())
            v = '_'.join(str(v).split())

            if u not in self.vocab_kge['entity'].keys() or v not in self.vocab_kge['entity'].keys():
                break

            u_idx = self.vocab_kge['entity'][u]
            v_idx = self.vocab_kge['entity'][v]
            self.adj_matrix[u_idx][v_idx] = 1

            ret.append(self.vocab_kge['entity'][u])
            ret.append(self.vocab_kge['entity'][v])

        return list(set(ret))

    def get_visible_state_rep_drqa(self, state_description):
        state_desc_num = []#120 * [0]

        for i, token in enumerate(word_tokenize(state_description)[:80]):
            if token not in self.vocab_drqa.keys():
                token = '<UNK>'
            state_desc_num.append(self.vocab_drqa[token])

        return state_desc_num

    def get_action_rep_drqa(self, action):
        action_desc_num = 20 * [0]

        for i, token in enumerate(word_tokenize(action)[:20]):
            if token not in self.vocab_drqa.keys():
                token = '<UNK>'

            action_desc_num[i] = self.vocab_drqa[token]

        return action_desc_num

    def get_cur_actions(self):
        return list(self.all_actions.keys())

    def get_cur_actions_pruned(self):
        action_ents = {a:[] for a in self.all_actions.keys()}
        action_scores = {a:0 for a in self.all_actions.keys()}
        for action in self.all_actions.keys():

            for n in self.graph_state.nodes:
                if str(n) in action:
                    action_ents[action] += [n]
                    action_scores[action] += 1

        for a in action_ents.keys():
            if len(action_ents[a]) < 2:
                continue
            ent_pairs = itertools.combinations(action_ents, 2)
            try:
                for pair in ent_pairs:
                    if nx.has_path(self.graph_state, pair[0], pair[1]):
                        action_scores[a] += 1
                    if nx.has_path(self.graph_state, pair[1], pair[0]):
                        action_scores[a] += 1
            except nx.NodeNotFound:
                continue

        sorted_scores = sorted(action_scores.items(), key=lambda kv: kv[1], reverse=True)
        max_score = max([a[1] for a in sorted_scores])
        max_actions = 36

        if max_score == 0:
            ret = random.sample(list(self.all_actions.keys()), max_actions)
            return ret

        partitions = {s: [] for s in range(0, max_score + 1)}

        for act, score in sorted_scores:
            partitions[score] += [act]

        ret = []
        left = max_actions
        for s in range(max_score, 0, -1):
            sample_no = min(left, len(partitions[s]))
            left -= sample_no
            to_add = random.sample(partitions[s], sample_no)
            ret += to_add
            if len(ret) > max_actions:
                ret = ret[:max_actions]
                break
        for dir in ['north', 'south', 'east', 'west']:
            ret.append('go ' + dir)

        return ret

    def get_action_text(self, action_ids):
        ret = ""
        for ids in action_ids:
            if ids != 0:
                if self.rev_vocab_drqa[ids] != "'s":
                    ret += ' ' + self.rev_vocab_drqa[ids]
                else:
                    ret += self.rev_vocab_drqa[ids]
        #ret = " ".join([self.rev_vocab_drqa[i] for i in action_ids if i != 0])
        ret = ret.strip()
        return ret
    
    def step_pruned(self, visible_state, prev_action=None):
        self.update_state(visible_state, prev_action)

        self.vis_pruned_actions = self.get_cur_actions_pruned()

        self.pruned_actions_rep = [self.get_action_rep_drqa(a) for a in self.vis_pruned_actions]

        inter = self.visible_state + "The actions are:" + ",".join(self.vis_pruned_actions) + "."
        self.drqa_input = self.get_visible_state_rep_drqa(inter)

        self.graph_state_rep = self.get_state_rep_kge(), self.adj_matrix

    def step(self, visible_state, prev_action=None, pruned=True):
        if pruned:
            self.step_pruned(visible_state, prev_action)
            return
        self.update_state(visible_state, prev_action)

        self.vis_pruned_actions = self.get_cur_actions()

        self.pruned_actions_rep = [self.get_action_rep_drqa(a) for a in self.vis_pruned_actions]

        inter = self.visible_state + "The actions are:" + ",".join(self.vis_pruned_actions) + "."
        self.drqa_input = self.get_visible_state_rep_drqa(inter)

        self.graph_state_rep = self.get_state_rep_kge(), self.adj_matrix

        #return graph_state_rep, vis_feasible_actions, feasible_action_rep, drqa_input
