
import random
import numpy as np
random.seed(123)
np.random.seed(456)

import sys
import numpy as np
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


from hpbandster.core.base_iteration import BaseIteration
class SuccessiveHalving(BaseIteration):
    def _advance_to_next_stage(self, config_ids, losses):
        print('SuccessiveHalving._advance_to_next_stage', self.stage)
        ranks = np.argsort(np.argsort(losses))
        return(ranks < self.num_configs[self.stage])


class BOHB:
    def __init__(self,
        configspace=None,
        eta=3,
        min_budget=0.01,
        max_budget=1,
        min_points_in_model=None,
        top_n_percent=15,
        num_samples=64,
        random_fraction=1/3,
        bandwidth_factor=3,
        min_bandwidth=1e-3
    ):
    
        self.config_generator = CG_BOHB(
            configspace = configspace,
            min_points_in_model = min_points_in_model,
            top_n_percent=top_n_percent,
            num_samples = num_samples,
            random_fraction=random_fraction,
            bandwidth_factor=bandwidth_factor,
            min_bandwidth = min_bandwidth
        )
        
        # Hyperband related stuff
        self.eta = eta
        self.min_budget = min_budget
        self.max_budget = max_budget
        
        # precompute some HB stuff
        self.max_SH_iter = -int(np.log(min_budget/max_budget)/np.log(eta)) + 1
        self.budgets     = max_budget * np.power(eta, -np.linspace(self.max_SH_iter-1, 0, self.max_SH_iter))
        
    def get_next_iteration(self, iteration, iteration_kwargs={}):
        
        print('-' * 100, file=sys.stderr)
        print('get_next_iteration:')
        
        # number of 'SH rungs'
        s = self.max_SH_iter - 1 - (iteration % self.max_SH_iter)
        
        # number of configurations in that bracket
        n0 = int(np.floor((self.max_SH_iter)/(s+1)) * self.eta**s)
        ns = [max(int(n0*(self.eta**(-i))), 1) for i in range(s+1)]
        
        print({
            "iteration"   : iteration,
            "num_configs" : ns,
            "budgets"     : self.budgets[(-s-1):],
        })
        
        return SuccessiveHalving(
            HPB_iter=iteration,
            num_configs=ns,
            budgets=self.budgets[(-s-1):],
            config_sampler=self.config_generator.get_config,
            **iteration_kwargs
        )

class Job:
    def __init__(self, id, kwargs, result):
        self.id     = id
        self.kwargs = kwargs
        self.result = result
        
        self.exception  = None
        self.timestamps = -1

# --

worker = MyWorker()

max_budget = 9

bohb = BOHB(
    configspace = worker.get_configspace(),
    min_budget  = 1,
    max_budget  = max_budget,
    eta         = 3,
)

iterations     = 10
all_iterations = []
all_results    = []
for iteration in range(iterations):
    iteration         = bohb.get_next_iteration(iteration=iteration)
    iteration_results = []
    while True:
        next_run = iteration.get_next_run()
        if next_run is None:
            break
        
        print('next_run', next_run)
        config_id, config, budget = next_run
        
        res = worker.compute(config_id=config_id, config=config, budget=budget)
        print(res)
        job = Job(id=config_id, kwargs={"budget" : budget, "config" : config}, result=res)
        
        iteration.register_result(job)
        bohb.config_generator.new_result(job)
        iteration_results.append({"config" : config, "budget" : budget, "res" : res})
    
    all_results += iteration_results
    all_iterations.append(iteration)

best_run = sorted([a for a in all_results if a['budget'] == max_budget], key=lambda x: x['res']['loss'])[0]
print(best_run['config'])






