import numpy as np
import mix_util
#import numba 
#numba.set_num_threads(4)
from numba import njit

import optimization


@njit
def fwht(a) -> None:
    """In-place Fast Walsh–Hadamard Transform of array a of size 2^n."""
    N = a.shape[0]
    sqrtN = np.sqrt(N)
    h = 1
    while h < N:
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = (x + y)
                a[j + h] = (x - y)
        h *= 2

    for j in range(len(a)):
        a[j] /= sqrtN

def unitary_propagator(H, state, time):
    tmp = -1.0j*time
    tmp = tmp*H
    tmp = np.exp(tmp)
    return tmp*state

def abstract_qaoa_evolve(Hp_run, mixer_params, problem_params):
    N = Hp_run.shape[0]
    n = int(np.round(np.log2(N)))

    Hd_transformed = mix_util.standard_mixer_eigenvalues(n)
    layers = len(mixer_params)

    if not len(problem_params) == layers:
        raise ValueError

    state = np.ones(N)/np.sqrt(N)
    for layer in range(layers):
        state = unitary_propagator(Hp_run, state, problem_params[layer])
        fwht(state)
        state = unitary_propagator(Hd_transformed, state, mixer_params[layer])
        fwht(state)
    return state

def abstract_qaoa_sample(Hp_run, mixer_params, problem_params, shots=None):
    fstate = abstract_qaoa_evolve(Hp_run, mixer_params, problem_params)
    if shots is None:
        res = fstate
    else:
        try:
            shots, xshots = shots
        except TypeError:
            shots, xshots = shots, 0
        N = len(Hp_run)
        fprobs = np.abs(fstate)**2
        samples = np.random.default_rng().choice(N, size=shots, p=fprobs)
        samples, counts = np.unique(samples, return_counts=True)
        res = samples, counts
        if xshots > 0:
            fstate_x = fstate.copy()
            fwht(fstate_x)
            fprobs_x = np.abs(fstate_x)**2
            samples_x = np.random.default_rng().choice(N, size=xshots, \
                p=fprobs_x)
            samples_x, counts_x = np.unique(samples_x, return_counts=True)
            res = res[0], res[1], samples_x, counts_x
    return res

def abstract_qaoa_objective(Hp_cost, Hp_run, mixer_params, problem_params, \
    shots=None, xweight=0.0, sample_catcher=None):
    if shots is None:
        fstate = abstract_qaoa_sample(Hp_run, mixer_params, problem_params, \
            shots=None)
        fprobs = np.abs(fstate)**2
        if xweight == 0.0:
            obj = np.dot(Hp_cost, fprobs)
        else:
            N = len(Hp_cost)
            n = int(np.round(np.log2(N)))
            Hd_transformed = mix_util.standard_mixer_eigenvalues(n)
            fstate_x = fstate.copy()
            fwht(fstate_x)
            fprobs_x = np.abs(fstate_x)**2
            obj = ((1-xweight)*np.dot(Hp_cost, fprobs)) + (xweight*np.dot(Hd_transformed, fprobs_x))
    else:
        try:
            shots, xshots = shots
        except TypeError:
            shots, xshots = shots, 0
        if xshots == 0:
            samples, counts = abstract_qaoa_sample(Hp_run, mixer_params, \
                problem_params, shots=shots)
        else:
            samples, counts, samples_x, counts_x = abstract_qaoa_sample(\
                Hp_run, mixer_params, problem_params, shots=(shots, xshots))
        
        if not sample_catcher is None:
            for j, sample in enumerate(samples):
                try: 
                    sample_catcher[sample] += counts[j]
                except KeyError:
                    sample_catcher[sample] = counts[j]

        nrgs = Hp_cost[samples]
        obj = np.dot(nrgs, counts)/np.sum(counts)
        if xshots > 0 and xweight > 0.0:
            N = len(Hp_cost)
            n = int(np.round(np.log2(N)))
            Hd_transformed = mix_util.standard_mixer_eigenvalues(n)
            nrgs_x = Hd_transformed[samples_x]
            obj_x = np.dot(nrgs_x, counts_x)/np.sum(counts_x)
            obj = ((1.0-xweight)*obj) + (xweight*obj_x)

    return obj

def process_sample_catcher(Hp, sample_catcher):
    samples, counts, nrgs = [], [], []
    for key, val in sample_catcher.items():
        samples.append(key)
        counts.append(val)
        nrg = Hp[key]
        nrgs.append(nrg)
    samples = np.array(samples, dtype=int)
    counts = np.array(counts, dtype=int)
    nrgs = np.array(nrgs, dtype=float)
    idx = np.argsort(nrgs)
    samples, counts, nrgs = samples[idx], counts[idx], nrgs[idx]
    return (samples, counts, nrgs)

def abstract_qaoa(Hp_cost, Hp_run, layers, shots=None, extra_shots=0, \
    optimizer='bobyqa', optimization_options=None, verbose=False):
    optimizer = optimizer.lower()
    if optimization_options is None:
        optimization_options = {}

    sample_catcher = {}

    extra_output = None

    calls = 0
    best_obj = np.inf
    def func(mixer_params, problem_params):
        nonlocal calls
        nonlocal best_obj
        obj = abstract_qaoa_objective(Hp_cost, Hp_run, mixer_params, \
            problem_params, shots=shots, sample_catcher=sample_catcher)
        calls += 1
        if obj < best_obj:
            best_obj = obj
        if verbose:
            print(f"{calls} completed. "+\
                f"Current best objective: {best_obj}      ", end="\r")
        return obj

    if optimizer in ('spsa', 'spsa_interp', 'spsa_interp2', 'spsa_fourier'):
        opt_alg = 'spsa'
    if optimizer in ('bobyqa', 'bobyqa_interp', 'bobyqa_interp2', 'bobyqa_fourier'):
        opt_alg = 'bobyqa'
    if optimizer in ('sa', 'sa_interp', 'sa_interp2', 'sa_fourier'):
        opt_alg = 'sa'
    
    if optimizer in ('spsa', 'bobyqa', 'sa'):
        param_type = 'standard'
    if optimizer in ('spsa_interp', 'bobyqa_interp', 'sa_interp'):
        param_type = 'interp'
    if optimizer in ('spsa_interp2', 'bobyqa_interp2', 'sa_interp2'):
        param_type = 'interp2'
    if optimizer in ('spsa_fourier', 'bobyqa_fourier', 'sa_fourier'):
        param_type = 'interp2'
    
    if param_type == 'standard':
        mixer_param_init = optimization_options['mixer_param_init']
        problem_param_init = optimization_options['problem_param_init']
    if param_type == 'interp':
        mixer_param_points_init = optimization_options['mixer_param_points_init']
        problem_param_points_init = optimization_options['problem_param_points_init']
    if param_type == 'interp2':
        mixer_param_vals_init = optimization_options['mixer_param_vals_init']
        problem_param_vals_init = optimization_options['problem_param_vals_init']
    if param_type == 'fourier':
        mixer_modes_init = optimization_options['mixer_modes_init']
        problem_modes_init = optimization_options['problem_modes_init']

    if opt_alg == 'spsa':
        spsa_runs = optimization_options.get('runs', 1)
    if opt_alg == 'bobyqa':
        noisy = optimization_options.get('noisy', True)
        max_for_global = optimization_options.get('max_for_global', 1.0)
        maxfun = optimization_options.get('maxfun', None)
        restart = optimization_options.get('restart', False)
    if opt_alg == 'sa':
        param_max = optimization_options.get('param_max', np.pi)
        stepsize = optimization_options.get('stepsize', 0.05)
        iterations = optimization_options.get('iterations', 1000)
        runs = optimization_options.get('runs', 1)
        max_temperature = optimization_options.get('max_temperature', 10)

    if optimizer == 'spsa':
        opt_mixer_params, opt_problem_params, opt_objective, extra_output = optimization.spsa_minimize(\
            func, layers, mixer_param_init, problem_param_init, spsa_runs)
    if optimizer == 'spsa_interp':
        opt_mixer_params, opt_problem_params, opt_objective, extra_output = optimization.spsa_minimize_interp(\
            func, layers, mixer_param_points_init, problem_param_points_init, spsa_runs)
    if optimizer == 'spsa_interp2':
        opt_mixer_params, opt_problem_params, opt_objective, extra_output = optimization.spsa_minimize_interp2(\
            func, layers, mixer_param_vals_init, problem_param_vals_init, spsa_runs)
    if optimizer == 'spsa_fourier':
        opt_mixer_params, opt_problem_params, opt_objective, extra_output = optimization.spsa_minimize_fourier(\
            func, layers, mixer_modes_init, problem_modes_init, spsa_runs)

    if optimizer == 'bobyqa':
        opt_mixer_params, opt_problem_params, opt_objective, extra_output = optimization.bobyqa_minimize(\
            func, layers, mixer_param_init, problem_param_init, noisy, maxfun=maxfun, max_for_global=max_for_global, restart=restart)
    if optimizer == 'bobyqa_interp':
        opt_mixer_params, opt_problem_params, opt_objective, extra_output = optimization.bobyqa_minimize_interp(\
            func, layers, mixer_param_points_init, problem_param_points_init, noisy, maxfun=maxfun, max_for_global=max_for_global, restart=restart)
    if optimizer == 'bobyqa_interp2':
        opt_mixer_params, opt_problem_params, opt_objective, extra_output = optimization.bobyqa_minimize_interp2(\
            func, layers, mixer_param_vals_init, problem_param_vals_init, noisy, maxfun=maxfun, max_for_global=max_for_global, restart=restart)
    if optimizer == 'bobyqa_fourier':
        opt_mixer_params, opt_problem_params, opt_objective, extra_output = optimization.bobyqa_minimize_fourier(\
            func, layers, mixer_modes_init, problem_modes_init, noisy, maxfun=maxfun, max_for_global=max_for_global, restart=restart)

    if optimizer == 'sa':
        opt_mixer_params, opt_problem_params, opt_objective, extra_output = optimization.sa_minimize(\
            func, layers, mixer_param_init, problem_param_init, param_max, stepsize, iterations, runs, max_temperature)
    if optimizer == 'sa_interp':
        opt_mixer_params, opt_problem_params, opt_objective, extra_output = optimization.sa_minimize_interp(\
            func, layers, mixer_param_points_init, problem_param_points_init, param_max, stepsize, iterations, runs, max_temperature)
    if optimizer == 'sa_interp2':
        opt_mixer_params, opt_problem_params, opt_objective, extra_output = optimization.sa_minimize_interp2(\
            func, layers, mixer_param_vals_init, problem_param_vals_init, param_max, stepsize, iterations, runs, max_temperature)
    if optimizer == 'sa_fourier':
        opt_mixer_params, opt_problem_params, opt_objective, extra_output = optimization.sa_minimize_fourier(\
            func, layers, mixer_modes_init, problem_modes_init, param_max, stepsize, iterations, runs, max_temperature)

    if extra_shots > 0:
        abstract_qaoa_objective(Hp_cost, Hp_run, opt_mixer_params, \
            opt_problem_params, shots=extra_shots, \
            sample_catcher=sample_catcher)

    samples = process_sample_catcher(Hp_cost, sample_catcher)

    if extra_output is None:
        extra_output = ()
    extra_output = tuple(extra_output) + (calls,)

    return opt_mixer_params, opt_problem_params, opt_objective, samples, extra_output