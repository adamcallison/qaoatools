import numpy as np

class RandomFloatSource(object):
    def __init__(self, cache_size=2**14):
        self.cache_size = cache_size
        self._cache = np.random.default_rng().uniform(size=cache_size)
        self._used = 0

    def get(self, n=None):
        if n is None:
            n_use = 1
        else:
            n_use = n

        if n_use > self.cache_size:
            result = np.random.default_rng().uniform(size=n_use)
        elif self._used + n_use <= self.cache_size:
            result = self._cache[self._used: self._used+n_use]
            self._used += n_use
        else:
            resultp1 = self._cache[self._used:]
            n_remaining = n_use - resultp1.shape[0]
            self._cache = np.random.default_rng().uniform(size=self.cache_size)
            resultp2 = self._cache[:n_remaining]
            self._used = n_remaining
            result = np.concatenate((resultp1, resultp2))
        if n is None:
            result = result[0]
        return result

class SimulatedAnnealer(object):
    def __init__(self, cost_function=None, initial_state_generator=None, \
        neighbour_generator=None, acceptance_rule=None, \
        acceptance_parameter_generator=None, parameters=None):

        self.cost_function = cost_function
        self.initial_state_generator = initial_state_generator
        self.neighbour_generator = neighbour_generator
        self.acceptance_rule = acceptance_rule
        self.acceptance_parameter_generator = acceptance_parameter_generator
        if parameters is None:
            self.parameters = {}
        else:
            self.parameters = parameters
        self.parameters_current = dict(self.parameters)

        self.run_state = {}

    def run_multi(self, runs, iterations, parameters=None):
        best_cost = np.inf
        for run in range(runs):
            best_state_curr, best_cost_curr, final_run_state_curr = \
                self.run(iterations, parameters=parameters, verbose=False)
            if best_cost_curr < best_cost:
                best_state, best_cost, final_run_state = best_state_curr, \
                    best_cost_curr, final_run_state_curr
        return best_state, best_cost, final_run_state

    def run(self, iterations, parameters=None, verbose=False):
        self.run_state = {}
        try:
            best_state, best_cost = self._run(iterations, \
                parameters=parameters, verbose=verbose)
        except:
            self.run_state = {}
            raise
        final_run_state = self.run_state
        self.run_state = {}
        return best_state, best_cost, final_run_state

    def _run(self, iterations, parameters=None, verbose=False):
        if not (parameters is None):
            for p, val in parameters.items():
                self.parameters_current[p] = val
        
        self.run_state['iterations'] = iterations
        self.run_state['state'] = \
            self.initial_state_generator(self.parameters_current)
        self.run_state['cost'] = self.cost_function(self.run_state['state'], \
            self.parameters_current, self.run_state)
        self.run_state['costs'] = [self.run_state['cost']]
        self.run_state['best_cost'] = self.run_state['cost']
        self.run_state['best_state'] = self.run_state['state']
        self.run_state['acceptance_history'] = []
        self.run_state['random_float_source'] = RandomFloatSource()

        if verbose: last_print = -np.float('inf')
        last_costsave = 0.0

        for iteration in range(1, iterations):
            self.run_state['iteration'] = iteration
            if verbose:
                pc = 100*(iteration)/iterations
                if pc - last_print >= 1.0-(10e-10):
                    last_print = pc
                    starting_cost = self.run_state['costs'][0]
                    current_cost = self.run_state['cost']
                    print(f"{pc:.2f}% complete. " + \
                        f"Starting_cost={starting_cost}. " + \
                        f"Current cost={current_cost}.", end="\r")

            acceptance_parameter = self.acceptance_parameter_generator(\
                self.parameters_current, self.run_state)
            neighbour = self.neighbour_generator(self.run_state['state'], \
                self.parameters_current, self.run_state)
            neighbour_cost = self.cost_function(neighbour, \
                self.parameters_current, self.run_state)
            accept = self.acceptance_rule(self.run_state['cost'], \
                neighbour_cost, acceptance_parameter, self.parameters_current, \
                self.run_state)
            self.run_state['acceptance_history'].append(accept)
            if accept:
                self.run_state['state'], self.run_state['cost'] = \
                    neighbour, neighbour_cost
                if self.run_state['cost'] < self.run_state['best_cost']:
                    self.run_state['best_state'] = self.run_state['state']
                    self.run_state['best_cost'] = self.run_state['cost']
            pm = (iteration/iterations)*1000
            if pm - last_costsave >= 1.0:
                last_costsave = pm
                self.run_state['costs'].append(self.run_state['cost'])

        return self.run_state['best_state'], self.run_state['best_cost']

def initial_state_generator(sa_params):
    mins, maxs = np.array(sa_params['param_mins']), \
        np.array(sa_params['param_maxs'])
    nparams = len(mins)
    tmp = np.random.default_rng().uniform(size=nparams)
    initial_state = tuple(mins + (tmp*(maxs-mins)))
    return initial_state

def neighbour_generator(state, sa_params, run_state):
    mins, maxs = np.array(sa_params['param_mins']), \
        np.array(sa_params['param_maxs'])
    nparams = len(mins)
    stepsize = sa_params['stepsize']
    step = np.random.default_rng().uniform(size=nparams)
    step = 2.0*(step-0.5)
    step = step / np.sqrt(np.dot(step, step))
    step = step*stepsize
    for j in range(nparams):
        if (state[j] + step[j]) < mins[j]:
            step[j] = mins[j] - state[j]
        elif (state[j] + step[j]) > maxs[j]:
            step[j] = maxs[j] - state[j]
        else:
            pass
    new_state = tuple(np.array(state) + step)
    return new_state

def boltzmann_acceptance_rule(current_cost, candidate_cost, temperature, \
    sa_params, run_state):
    if candidate_cost <= current_cost:
        accept = True
    else:
        acceptance_probability = \
            np.exp(-(candidate_cost-current_cost)/temperature)

        test_probability = np.random.default_rng().uniform()
        accept = test_probability <= acceptance_probability
    return accept

def temperature_schedule(sa_params, run_state):
    T_max = sa_params['max_temperature']
    iteration = run_state['iteration']
    iterations = run_state['iterations']
    scale = np.log((iterations+1)/(iteration+1))/np.log(iterations+1)

    T = T_max*scale
    return T

def minimize(func, bounds, runs, iterations, stepsize, max_temperature):
    sa = SimulatedAnnealer(cost_function=func, \
        initial_state_generator=initial_state_generator, \
        neighbour_generator=neighbour_generator, \
        acceptance_rule=boltzmann_acceptance_rule, \
        acceptance_parameter_generator=temperature_schedule)

    param_mins = tuple(x[0] for x in bounds)
    param_maxs = tuple(x[1] for x in bounds)

    sa_params = {
        'param_mins':param_mins,
        'param_maxs':param_maxs,
        'stepsize':stepsize,
        'max_temperature':max_temperature,
    } 

    best_state, best_cost, final_run_state = sa.run_multi(runs, iterations, \
        parameters=sa_params)

    return best_state, best_cost
