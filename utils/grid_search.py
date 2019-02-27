import utils.schedule
import itertools
import random


def generate_cartesian_product(dict):
    result = []
    dict_notags = []
    keys = []
    for key in dict:
        keys.append(key)
        dict_notags.append(dict[key])
    for p in itertools.product(*dict_notags):
        pending_object = {}
        for index in range(len(p)):
            pending_object[keys[index]] = p[index]
        result.append(pending_object)
    return result


class RandomGridSearch(object):
    def __init__(self, grid, per_params, seed):
        self.grid = grid
        self.all_configs = generate_cartesian_product(self.grid)
        print(len(self.all_configs))
        self.max_params = int(per_params * len(self.all_configs))
        self.finished_params = 0
        random.seed(seed)

    def get_config(self):
        idx = random.randint(0, len(self.all_configs))
        self.finished_params += 1 
        return self.all_configs.pop(idx)

    def get_configs(self):
        idxs = random.sample(range(len(self.all_configs)), k=self.max_params)
        return [self.all_configs[i] for i in idxs]
    
    def is_done(self):
        return self.finished_params == self.max_params
        


