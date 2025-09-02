from pettingzoo.test import api_test
from pettingzoo.test import parallel_api_test
from pettingzoo.test import seed_test, parallel_seed_test
from pettingzoo.test import max_cycles_test
from pettingzoo.test import render_test
from pettingzoo.test import performance_benchmark
from pettingzoo.test import test_save_obs

import multiCarEnv
from multiCarEnv import ParallelCarEnv

env = ParallelCarEnv()
env.reset()  # Reset the environment to initialize agents properly

parallel_api_test(env, num_cycles=1000)


parallel_seed_test(lambda: ParallelCarEnv())

from pettingzoo.utils.conversions import parallel_to_aec
render_test(lambda render_mode=None: parallel_to_aec(ParallelCarEnv(render_mode=render_mode)))

performance_benchmark(parallel_to_aec(ParallelCarEnv()))

