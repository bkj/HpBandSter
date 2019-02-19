#!/usr/bin/env python

"""
    bohb.py
"""

import numpy as np
from uuid import uuid4

import ConfigSpace as CS
from hpbandster.optimizers.config_generators.bohb import BOHB as CG_BOHB

from rsub import *
from matplotlib import pyplot as plt

np.random.seed(123)

# --
# Helpers

class FiniteIterable:
    """ I'm sure there's some way to do this with itertools """
    def __init__(self, x, max_steps):
        self.x = x
        
        self.max_steps = max_steps
        self.counter   = 0
    
    def __next__(self):
        if self.counter >= self.max_steps:
            raise StopIteration
        else:
            self.counter += 1
            return next(self.x)
    
    def __iter__(self):
        return self


# --
# Sampler

class _Job:
    def __init__(self, id, kwargs, result):
        self.id         = id
        self.kwargs     = kwargs
        self.result     = result
        self.exception  = None
        self.timestamps = -1


class Sampler:
    """ Wrapper around CG_BOHB """
    def __init__(self, configspace):
        self.results = []
        self._config_generator = CG_BOHB(
            configspace         = configspace,
            min_points_in_model = None,
            top_n_percent       = 15,
            num_samples         = 64,
            random_fraction     = 1 / 3,
            bandwidth_factor    = 3,
            min_bandwidth       = 1e-3,
        )
        
    def __next__(self):
        config, config_info = self._config_generator.get_config(budget=-1)
        return Task(config=config, config_info=config_info)
    
    def update(self, task):
        self._config_generator.new_result(
            _Job(
                id=task.id,
                kwargs={
                    "budget" : task.budget,
                    "config" : task.config
                },
                result={
                    "loss" : task.score
                }
            )
        )


# --
# Task

# The thing you're trying to optimize.
# In practice, this would be a wrapper around an NN.

class Task:
    def __init__(self, config, config_info, seed=None):
        
        self.id     = str(uuid4())
        self.config = config
        self.score  = None
        self.budget = None
        
        self.rng = np.random.RandomState(seed) if seed is not None else np.random
        
        self.config      = config
        self.config_info = config_info
    
    def train(self, budget):
        x = self.config['x']
        self.budget = budget
        self.score  = np.clip(x + self.rng.randn() / budget, 0.5 * x, 1.5 * x)
    
    @property
    def summary(self):
        return {
            "id"          : self.id,
            "budget"      : self.budget,
            "score"       : self.score,
            "config"      : self.config,
            "config_info" : self.config_info,
        }


# --
# SHScheduler

# Given a sampler, runs successive halving

class SHScheduler:
    def __init__(self, eta=3, min_budget=1, max_budget=81):
        
        self.eta        = eta
        self.min_budget = min_budget
        self.max_budget = max_budget
        
        self.history = []
        
        self.num_brackets = -int(np.log(min_budget/max_budget)/np.log(eta)) + 1
        self.budgets      = max_budget * np.power(eta, -np.linspace(self.num_brackets-1, 0, self.num_brackets))
    
    def get_pop_sizes(self, bracket_idx):
        s  = self.num_brackets - 1 - (bracket_idx % self.num_brackets)
        n0 = int(np.floor(self.num_brackets / (s + 1)) * self.eta ** s)
        ns = [max(int(n0 * (self.eta ** (-i))), 1) for i in range(s + 1)]
        return ns
    
    def get_budgets(self, bracket_idx):
        s = self.num_brackets - 1 - (bracket_idx % self.num_brackets)
        return list(self.budgets[(-s-1):])
    
    def cull(self, population, k):
        return sorted(list(population), key=lambda x: x.score)[:k]
    
    def run_stage(self, population, budget, callback=None):
        for p in population:
            p.train(budget)
            if callback is not None:
                callback(p)
            
            self.history.append(p.summary)
            yield p
        
    def run_bracket(self, sampler, bracket_idx):
        budgets    = self.get_budgets(bracket_idx)
        pop_sizes  = self.get_pop_sizes(bracket_idx)
        
        for stage in range(len(budgets)):
            if stage == 0:
                # In the first stage, we're using the sampler
                # After each evaluation, we update the sampler's model
                population = FiniteIterable(sampler, pop_sizes[stage])
                population = self.run_stage(population, budget=budgets[stage], callback=sampler.update)
            else:
                # In later stages, don't use the sampler, but we still update the model
                population = self.cull(population, k=pop_sizes[stage])
                population = self.run_stage(population, budget=budgets[stage], callback=sampler.update)
            
            population = list(population)
            print('bracket_idx=%d | stage=%d | population=%d | budget=%d' % (bracket_idx, stage, len(population), budgets[stage]))


def run_hyperband(scheduler, sampler):
    for bracket_idx in range(scheduler.num_brackets):
        scheduler.run_bracket(sampler, bracket_idx=bracket_idx)
    
    return scheduler


config_space = CS.ConfigurationSpace()
config_space.seed(123)
config_space.add_hyperparameter(CS.UniformFloatHyperparameter('x', lower=0, upper=1))

sampler   = Sampler(config_space)
scheduler = SHScheduler()
scheduler = run_hyperband(scheduler, sampler)

_ = plt.plot([h['score'] for h in scheduler.history if h['config_info']['model_based_pick']])
_ = plt.plot([h['score'] for h in scheduler.history if not h['config_info']['model_based_pick']])
_ = plt.yscale('log')
show_plot()

