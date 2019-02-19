
import random
import numpy as np
random.seed(123)
np.random.seed(456)

import sys
import numpy as np
from collections import defaultdict

import ConfigSpace as CS
from hpbandster.optimizers.config_generators.bohb import BOHB as CG_BOHB

class MyWorker:
    def __init__(self):
        self.rng  = np.random.RandomState(456)
        
    def compute(self, config_id, config, budget, **kwargs):
        
        print('MyWorker.compute:')
        print('\tconfig_id', config_id)
        print('\tconfig   ', config)
        print('\tbudget   ', budget)
        
        res = np.clip(config['x'] + self.rng.randn() / budget, config['x']/2, 1.5*config['x'])
        
        return {
            'loss': float(res),
            'info': res
        }
    
    @staticmethod
    def get_configspace():
        config_space = CS.ConfigurationSpace()
        config_space.seed(123)
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))
        return config_space



def make_config_generator(
        configspace=None,
        min_points_in_model=None,
        top_n_percent=15,
        num_samples=64,
        random_fraction=1/3,
        bandwidth_factor=3,
        min_bandwidth=1e-3,
    ):
    
    return CG_BOHB(
        configspace         = configspace,
        min_points_in_model = min_points_in_model,
        top_n_percent       = top_n_percent,
        num_samples         = num_samples,
        random_fraction     = random_fraction,
        bandwidth_factor    = bandwidth_factor,
        min_bandwidth       = min_bandwidth
    )


class SHScheduler:
    def __init__(self, eta=3, min_budget=0.01, max_budget=1):
        
        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget
        
        # precompute some HB stuff
        self.max_SH_iter = -int(np.log(min_budget/max_budget)/np.log(eta)) + 1
        self.budgets     = max_budget * np.power(eta, -np.linspace(self.max_SH_iter-1, 0, self.max_SH_iter))
        
    def get_iter_params(self, iter_num):
        
        print('-' * 100, file=sys.stderr)
        print('get_next_iter_num:')
        
        # number of 'SH rungs'
        s = self.max_SH_iter - 1 - (iter_num % self.max_SH_iter)
        
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
        ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
        
        print({
            "iter_num"    : iter_num,
            "num_configs" : ns,
            "budgets"     : self.budgets[(-s-1):],
        })
        
        return {
            "num_configs" : ns,
            "budgets"     : list(self.budgets[(-s-1):])
        }

class _Job:
    def __init__(self, id, kwargs, result):
        self.id         = id
        self.kwargs     = kwargs
        self.result     = result
        self.exception  = None
        self.timestamps = -1

# -


worker = MyWorker()

max_budget = 9

hp_scheduler     = SHScheduler(eta=3, min_budget=1, max_budget=max_budget)
config_generator = make_config_generator(configspace = worker.get_configspace())

num_iterations = 10
all_history = []
for iter_num in range(num_iterations):
    
    history     = defaultdict(list)
    iter_params = hp_scheduler.get_iter_params(iter_num=iter_num)
    
    # --
    # First stage of SH (sample from space)
    
    num_stages = len(iter_params['num_configs'])
    
    stage       = 0
    num_configs = iter_params['num_configs'][0]
    budget      = iter_params['budgets'][0]
    
    for config_num in range(num_configs):
        config_id = (iter_num, stage, config_num)
        config, _ = config_generator.get_config(budget=budget)
        
        print('next_run', (config_id, config, budget))
        res = worker.compute(config_id=config_id, config=config, budget=budget)
        print(res)
        
        job = Job(id=config_id, kwargs={"budget" : budget, "config" : config}, result=res)
        config_generator.new_result(job)
        history[stage].append({"config_id" : config_id, "config" : config, "budget" : budget, "res" : res})
    
    # --
    # Next stages of SH (finish training)
    
    for stage in range(1, num_stages):
        num_configs = iter_params['num_configs'][stage]
        budget      = iter_params['budgets'][stage]
        
        population = sorted(history[stage - 1], key=lambda x: x['res']['loss'])[:num_configs]
        population = sorted(population, key=lambda x: x['config_id']) # For compatibility w/ original implementation
        
        results = []
        for p in population:
            config_id = p['config_id']
            config    = p['config']
            
            print('next_run', (config_id, config, budget))
            res = worker.compute(config_id=config_id, config=config, budget=budget)
            print(res)
            
            job = Job(id=config_id, kwargs={"budget" : budget, "config" : config}, result=res)
            config_generator.new_result(job)
            history[stage].append({"config_id" : config_id, "config" : config, "budget" : budget, "res" : res})
    
    all_history += sum(history.values(), [])

best_run = sorted([a for a in all_history if a['budget'] == max_budget], key=lambda x: x['res']['loss'])[0]
print(best_run['config'])

