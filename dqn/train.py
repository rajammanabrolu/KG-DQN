from dqn import DQNTrainer
from utils.grid_search import RandomGridSearch
from joblib import Parallel, delayed
import multiprocessing
import gc
#from guppy import hpy
#from memory_profiler import profile

#@profile
def parallelize(game, params):
    print(params)
    #game = "/home/eilab/Raj/tw-drl/Games/obj_20_qlen_5_room_10/train/game_" + str(10) + ".ulx"
    trainer = DQNTrainer(game, params)
    trainer.train()
    #del trainer.model
    #del trainer
    #gc.collect()

    """

    while not grid_search.is_done():
        params = grid_search.get_config()
        #trainer = DQNTrainer(game, params)
        #trainer.train()
    """


if __name__ == "__main__":
    param_grid = {
        'num_episodes': [1000, 5000],
        'num_frames': [500, 1000, 5000],
        'replay_buffer_type': ['priority', 'standard'],
        'replay_buffer_size': [10000, 50000],
        #'num_frames': [100000, 500000],
        'batch_size': [64],
        'lr': [0.01, 0.001],
        'gamma': [0.5, 0.2, 0.05],
        'rho': [0.25],
        'scheduler_type': ['exponential', 'linear'],
        'e_decay': [500, 100],
        'e_final': [0.01, 0.1, 0.2],
        'hidden_dims': [[64, 32], [128, 64], [256, 128]],
        'update_frequency': [1, 4, 10]
    }

    grid_search = RandomGridSearch(param_grid, 0.2, 21)
    game = "/home/eilab/Raj/tw-drl/Games/obj_20_qlen_5_room_10/train/game_" + str(10) + ".ulx"

    all_params = grid_search.get_configs()#[:4]
    #print(len(all_params))
    #pool = multiprocessing.Pool(processes=4)
    #pool.map(parallelize, all_params)
    #pool.close()
    #pool.join()
    #@profile
    #def run():
    Parallel(n_jobs=2, prefer='processes')(delayed(parallelize)(game, params) for params in all_params)
    #run()
