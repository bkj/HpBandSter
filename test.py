
import random
import numpy as np
random.seed(123)
np.random.seed(456)

import logging
logging.basicConfig(level=logging.WARNING)

import hpbandster.core.nameserver as hpns
import hpbandster.core.result as hpres
from hpbandster.optimizers import BOHB
from hpbandster.examples.commons import MyWorker


NS = hpns.NameServer(run_id='example1', host='127.0.0.1', port=None)
NS.start()

w = MyWorker(sleep_interval=0, nameserver='127.0.0.1',run_id='example1')
w.run(background=True)

result_logger = hpres.json_result_logger(directory='results', overwrite=True)

bohb = BOHB(
    configspace   = w.get_configspace(),
    run_id        = 'example1',
    nameserver    = '127.0.0.1',
    min_budget    = 1,
    max_budget    = 9,
    eta           = 3,
    result_logger = result_logger,
)

res = bohb.run(n_iterations=10)

bohb.shutdown(shutdown_workers=True)
NS.shutdown()

id2config = res.get_id2config_mapping()
incumbent = res.get_incumbent_id()

print(sum([v['config']['x'] for v in id2config.values()]))
print(id2config[incumbent]['config'])