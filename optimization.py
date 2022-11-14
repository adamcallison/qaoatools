import skopt
import numpy as np

import optim_spsa
import optim_gridsearch
import optim_sa

def gp_minimize(func, mixer_param_bounds, problem_param_bounds, n_calls, \
    n_initial_points):

    layers = len(mixer_param_bounds)

    def func2(params):
        mixer_params = params[:layers]
        problem_params = params[layers:]
        return func(mixer_params, problem_params)

    param_bounds = list(mixer_param_bounds) + list(problem_param_bounds)

    res = skopt.gp_minimize(func2, param_bounds, n_calls=n_calls, \
        n_initial_points=n_initial_points, initial_point_generator='lhs', \
        acq_func='EI')

    opt_params = np.array(res.x)
    opt_mixer_params = opt_params[:layers]
    opt_problem_params = opt_params[layers:]
    opt_objective = res.fun

    return opt_mixer_params, opt_problem_params, opt_objective

def gp_minimize_lbl(func, mixer_param_bounds, problem_param_bounds, \
    n_calls, n_initial_points):

    opt_mixer_params, opt_problem_params = [], []

    layers = len(mixer_param_bounds)

    for layer in range(layers):
        def func2(params):
            mixer_params = np.array(opt_mixer_params + [params[0]])
            problem_params = np.array(opt_problem_params + [params[1]])
            return func(mixer_params, problem_params)

        param_bounds = [mixer_param_bounds[layer], problem_param_bounds[layer]]

        res = skopt.gp_minimize(func2, param_bounds, n_calls=n_calls, \
            n_initial_points=n_initial_points, initial_point_generator='lhs', \
            acq_func='EI')

        opt_params = np.array(res.x)
        opt_mixer_params.append(opt_params[0])
        opt_problem_params.append(opt_params[1])
        opt_objective = res.fun

    opt_mixer_params = np.array(opt_mixer_params)
    opt_problem_params = np.array(opt_problem_params)

    return opt_mixer_params, opt_problem_params, opt_objective

def gp_minimize_lbl_weighted(func, mixer_param_bounds, problem_param_bounds, \
    n_calls, n_initial_points):

    opt_mixer_params, opt_problem_params = [], []

    layers = len(mixer_param_bounds)

    xweights = 1.0 - ( np.arange(1, layers+1)/(layers) )

    for layer in range(layers):
        def func2(params):
            mixer_params = np.array(opt_mixer_params + [params[0]])
            problem_params = np.array(opt_problem_params + [params[1]])
            return func(mixer_params, problem_params, xweights[layer])

        param_bounds = [mixer_param_bounds[layer], problem_param_bounds[layer]]

        res = skopt.gp_minimize(func2, param_bounds, n_calls=n_calls, \
            n_initial_points=n_initial_points, initial_point_generator='lhs', \
            acq_func='EI')

        opt_params = np.array(res.x)
        opt_mixer_params.append(opt_params[0])
        opt_problem_params.append(opt_params[1])
        opt_objective = res.fun

    opt_mixer_params = np.array(opt_mixer_params)
    opt_problem_params = np.array(opt_problem_params)

    return opt_mixer_params, opt_problem_params, opt_objective

def spsa_minimize(func, mixer_param_init, problem_param_init, runs):
    layers = len(mixer_param_init)
    initial_position = np.array(list(mixer_param_init) + list(problem_param_init))

    def cost_function(params):
        mixer_params = params[:layers]
        problem_params = params[layers:]
        return func(mixer_params, problem_params)

    perturb = 0.1
    lr = 0.1

    final_state = optim_spsa.minimize(cost_function, initial_position, \
        runs=runs, tolerance=1e-8, max_iterations=2000000, alpha=0.602, \
        lr=lr, perturb=perturb, gamma=0.101, blocking=False, \
        allowed_increase=0.5)

    opt_mixer_params = final_state['best_position'][:layers]
    opt_problem_params = final_state['best_position'][layers:]

    opt_objective = final_state['best_objective_value']

    return opt_mixer_params, opt_problem_params, opt_objective

def spsa_minimize_lbl_weighted(func, mixer_param_init, problem_param_init, \
    runs):
    layers = len(mixer_param_init)
    opt_mixer_params, opt_problem_params = [], []

    xweights = 1.0 - ( np.arange(1, layers+1)/(layers) )

    for layer in range(layers):
        def cost_function(params):
            mixer_params = np.array(opt_mixer_params + [params[0]])
            problem_params = np.array(opt_problem_params + [params[1]])
            return func(mixer_params, problem_params, xweights[layer])

        if layer == 0:
            initial_position = initial_position = [mixer_param_init[layer], problem_param_init[layer]]
        else:
            initial_position = list(opt_params)
        initial_position = [mixer_param_init[layer], problem_param_init[layer]]

        perturb = 0.01
        lr = 0.01

        final_state = optim_spsa.minimize(cost_function, initial_position, \
            runs=runs, tolerance=1e-8, max_iterations=2000000, alpha=0.602, \
            lr=lr, perturb=perturb, gamma=0.101, blocking=False, \
            allowed_increase=0.5)

        opt_params = np.array(final_state['best_position'])
        opt_mixer_params.append(opt_params[0])
        opt_problem_params.append(opt_params[1])
        opt_objective = final_state['best_objective_value']

    opt_mixer_params = np.array(opt_mixer_params)
    opt_problem_params = np.array(opt_problem_params)

    return opt_mixer_params, opt_problem_params, opt_objective

def spsa_minimize_mixer(func, mixer_param_init, problem_param_scale_init, runs):
    layers = len(mixer_param_init)
    problem_params_unscaled = (np.arange(1, layers+1)/(layers+1))


    def cost_function(params):
        problem_params = problem_params_unscaled*params[-1]
        mixer_params = params[:-1]
        return func(mixer_params, problem_params)

    initial_position = np.array(list(mixer_param_init) + \
        [problem_param_scale_init])

    perturb = 0.01
    lr = 0.01

    final_state = optim_spsa.minimize(cost_function, initial_position, \
        runs=runs, tolerance=1e-8, max_iterations=2000000, alpha=0.602, \
        lr=lr, perturb=perturb, gamma=0.101, blocking=False, \
        allowed_increase=0.5)

    opt_params = np.array(final_state['best_position'])
    opt_problem_params = problem_params_unscaled*opt_params[-1]
    opt_mixer_params = opt_params[:-1]
    opt_objective = final_state['best_objective_value']

    return opt_mixer_params, opt_problem_params, opt_objective

def spsa_minimize_linear(func, layers, mixer_param_bounds_init, \
    problem_param_bounds_init, runs):
    problem_params_unscaled = (np.arange(layers)/(layers-1))
    mixer_params_unscaled = 1 - problem_params_unscaled
    
    def cost_function(params):
        mixer_param_diff = params[0] - params[1]
        problem_param_diff = params[3] - params[2]
        mixer_params = params[1] + (mixer_params_unscaled*mixer_param_diff)
        problem_params = params[2] + \
            (problem_params_unscaled*problem_param_diff)
        return func(mixer_params, problem_params)

    initial_position = np.array(list(mixer_param_bounds_init) + \
        list(problem_param_bounds_init))

    perturb = 0.01
    lr = 0.01

    final_state = optim_spsa.minimize(cost_function, initial_position, \
        runs=runs, tolerance=1e-8, max_iterations=2000000, alpha=0.602, \
        lr=lr, perturb=perturb, gamma=0.101, blocking=False, \
        allowed_increase=0.5)

    opt_params = np.array(final_state['best_position'])
    mixer_param_diff = opt_params[0] - opt_params[1]
    problem_param_diff = opt_params[3] - opt_params[2]
    opt_mixer_params = opt_params[1] + (mixer_params_unscaled*mixer_param_diff)
    opt_problem_params = opt_params[2] + \
        (problem_params_unscaled*problem_param_diff)
    opt_objective = final_state['best_objective_value']

    return opt_mixer_params, opt_problem_params, opt_objective

def gs_minimize_lbl_weighted(func, mixer_param_bounds, problem_param_bounds, \
    nsteps, tol):

    tols = [tol]*2

    opt_mixer_params, opt_problem_params = [], []

    layers = len(mixer_param_bounds)

    xweights = 1.0 - ( np.arange(1, layers+1)/(layers) )

    for layer in range(layers):
        def func2(params):
            mixer_params = np.array(opt_mixer_params + [params[0]])
            problem_params = np.array(opt_problem_params + [params[1]])
            return func(mixer_params, problem_params, xweights[layer])

        param_bounds = [mixer_param_bounds[layer], problem_param_bounds[layer]]

        res = optim_gridsearch.minimize(func2, param_bounds, nsteps, tols)

        opt_objective, opt_params = res
        opt_mixer_params.append(opt_params[0])
        opt_problem_params.append(opt_params[1])

    opt_mixer_params = np.array(opt_mixer_params)
    opt_problem_params = np.array(opt_problem_params)

    return opt_mixer_params, opt_problem_params, opt_objective

def sa_minimize_lbl_weighted(func, mixer_param_bounds, problem_param_bounds, \
    runs, iterations, stepsize, max_temperature):

    opt_mixer_params, opt_problem_params = [], []

    layers = len(mixer_param_bounds)

    xweights = 1.0 - ( np.arange(1, layers+1)/(layers) )

    for layer in range(layers):
        def func2(params, sa_params, run_state):
            mixer_params = np.array(opt_mixer_params + [params[0]])
            problem_params = np.array(opt_problem_params + [params[1]])
            return func(mixer_params, problem_params, xweights[layer])

        param_bounds = [mixer_param_bounds[layer], problem_param_bounds[layer]]

        res = optim_sa.minimize(func2, param_bounds, runs, iterations, \
            stepsize, max_temperature)

        opt_params, opt_objective  = res
        opt_mixer_params.append(opt_params[0])
        opt_problem_params.append(opt_params[1])

    opt_mixer_params = np.array(opt_mixer_params)
    opt_problem_params = np.array(opt_problem_params)

    return opt_mixer_params, opt_problem_params, opt_objective