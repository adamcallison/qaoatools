import skopt
import pybobyqa
import numpy as np
import scipy.interpolate as spi

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

    perturb = 0.01
    lr = 0.01

    final_state = optim_spsa.minimize(cost_function, initial_position, \
        runs=runs, tolerance=1e-8, max_iterations=2000000, alpha=0.602, \
        lr=lr, perturb=perturb, gamma=0.101, blocking=False, \
        allowed_increase=0.5)

    opt_mixer_params = final_state['best_position'][:layers]
    opt_problem_params = final_state['best_position'][layers:]

    opt_objective = final_state['best_objective_value']

    return opt_mixer_params, opt_problem_params, opt_objective

def bobyqa_minimize(func, mixer_param_init, problem_param_init, noisy, max_for_global=None):
    layers = len(mixer_param_init)
    initial_position = np.array(list(mixer_param_init) + list(problem_param_init))

    def cost_function(params):
        mixer_params = params[:layers]
        problem_params = params[layers:]
        return func(mixer_params, problem_params)

    if max_for_global is None:
        seek_global = False
    else:
        seek_global = True
        bounds = ( np.array([0.0]*(2*layers)), np.array([max_for_global]*(2*layers)) )

    if seek_global:
        soln = pybobyqa.solve(cost_function, initial_position, bounds=bounds, objfun_has_noise=noisy, seek_global_minimum=seek_global)
    else:
        soln = pybobyqa.solve(cost_function, initial_position, objfun_has_noise=noisy, seek_global_minimum=seek_global)

    opt_params = np.array(soln.x)

    opt_mixer_params = opt_params[:layers]
    opt_problem_params = opt_params[layers:]

    opt_objective = soln.f

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


def interp(points):
    xs, ys = tuple(zip(*tuple(points)))
    sfunc = spi.PchipInterpolator(xs, ys)
    return sfunc
def schedule(points, layers):
    sfunc = interp(points)
    params = sfunc(np.arange(1, layers+1)/(layers+1))
    return params

def move_points(old_points, new_xs):
    old_sfunc = interp(old_points)
    new_points = [(new_x, old_sfunc(new_x)) for new_x in new_xs]
    return new_points

def spsa_minimize_interp(func, layers, mixer_param_points_init, \
    problem_param_points_init, runs):

    n_mixer_points = len(mixer_param_points_init)
    n_problem_points = len(mixer_param_points_init)

    def cost_function(params):
        mixer_points = ()
        for j in range(n_mixer_points):
            mixer_points += ((params[2*j], params[(2*j)+1]),)
        problem_points = ()
        for j in range(n_mixer_points, n_mixer_points+n_problem_points):
            problem_points += ((params[2*j], params[(2*j)+1]),)
        mixer_points = tuple(sorted(mixer_points, key=lambda x: x[0]))
        problem_points = tuple(sorted(problem_points, key=lambda x: x[0]))

        mixer_params = schedule(mixer_points, layers)
        problem_params = schedule(problem_points, layers)
        return func(mixer_params, problem_params)

    initial_position = []
    for j in range(n_mixer_points):
        initial_position += list(mixer_param_points_init[j])
    for j in range(n_problem_points):
        initial_position += list(problem_param_points_init[j])
    initial_position = np.array(initial_position)

    params = initial_position
    mixer_points = ()
    for j in range(n_mixer_points):
        mixer_points += ((params[2*j], params[(2*j)+1]),)
    problem_points = ()
    for j in range(n_mixer_points, n_mixer_points+n_problem_points):
        problem_points += ((params[2*j], params[(2*j)+1]),)
    mixer_points = tuple(sorted(mixer_points, key=lambda x: x[0]))
    problem_points = tuple(sorted(problem_points, key=lambda x: x[0]))
    mixer_params = schedule(mixer_points, layers)
    problem_params = schedule(problem_points, layers)

    perturb = 0.01
    lr = 0.01

    final_state = optim_spsa.minimize(cost_function, initial_position, \
        runs=runs, tolerance=1e-8, max_iterations=2000000, alpha=0.602, \
        lr=lr, perturb=perturb, gamma=0.101, blocking=False, \
        allowed_increase=0.5)

    opt_params = np.array(final_state['best_position'])
    mixer_points = ()
    for j in range(n_mixer_points):
        mixer_points += ((opt_params[2*j], opt_params[(2*j)+1]),)
    problem_points = ()
    for j in range(n_mixer_points, n_mixer_points+n_problem_points):
        problem_points += ((opt_params[2*j], opt_params[(2*j)+1]),)
    mixer_points = tuple(sorted(mixer_points, key=lambda x: x[0]))
    problem_points = tuple(sorted(problem_points, key=lambda x: x[0]))
    opt_mixer_params = schedule(mixer_points, layers)
    opt_problem_params = schedule(problem_points, layers)

    opt_objective = final_state['best_objective_value']

    return opt_mixer_params, opt_problem_params, opt_objective

def bobyqa_minimize_interp(func, layers, mixer_param_points_init, \
    problem_param_points_init, noisy, max_for_global=None):

    n_mixer_points = len(mixer_param_points_init)
    n_problem_points = len(mixer_param_points_init)

    if max_for_global is None:
        seek_global = False
    else:
        seek_global = True
        bounds = ( np.array([0.0]*(2*(n_mixer_points+n_problem_points))), np.array([max_for_global]*(2*(n_mixer_points+n_problem_points))) )


    def cost_function(params):
        mixer_points = ()
        for j in range(n_mixer_points):
            mixer_points += ((params[2*j], params[(2*j)+1]),)
        problem_points = ()
        for j in range(n_mixer_points, n_mixer_points+n_problem_points):
            problem_points += ((params[2*j], params[(2*j)+1]),)
        mixer_points = tuple(sorted(mixer_points, key=lambda x: x[0]))
        problem_points = tuple(sorted(problem_points, key=lambda x: x[0]))

        mixer_params = schedule(mixer_points, layers)
        problem_params = schedule(problem_points, layers)
        return func(mixer_params, problem_params)

    initial_position = []
    for j in range(n_mixer_points):
        initial_position += list(mixer_param_points_init[j])
    for j in range(n_problem_points):
        initial_position += list(problem_param_points_init[j])
    initial_position = np.array(initial_position)

    params = initial_position
    mixer_points = ()
    for j in range(n_mixer_points):
        mixer_points += ((params[2*j], params[(2*j)+1]),)
    problem_points = ()
    for j in range(n_mixer_points, n_mixer_points+n_problem_points):
        problem_points += ((params[2*j], params[(2*j)+1]),)
    mixer_points = tuple(sorted(mixer_points, key=lambda x: x[0]))
    problem_points = tuple(sorted(problem_points, key=lambda x: x[0]))
    mixer_params = schedule(mixer_points, layers)
    problem_params = schedule(problem_points, layers)

    if seek_global:
        soln = pybobyqa.solve(cost_function, initial_position, bounds=bounds, objfun_has_noise=noisy, seek_global_minimum=seek_global)
    else:
        soln = pybobyqa.solve(cost_function, initial_position, objfun_has_noise=noisy, seek_global_minimum=seek_global)

    opt_params = np.array(soln.x)
    mixer_points = ()
    for j in range(n_mixer_points):
        mixer_points += ((opt_params[2*j], opt_params[(2*j)+1]),)
    problem_points = ()
    for j in range(n_mixer_points, n_mixer_points+n_problem_points):
        problem_points += ((opt_params[2*j], opt_params[(2*j)+1]),)
    mixer_points = tuple(sorted(mixer_points, key=lambda x: x[0]))
    problem_points = tuple(sorted(problem_points, key=lambda x: x[0]))
    opt_mixer_params = schedule(mixer_points, layers)
    opt_problem_params = schedule(problem_points, layers)

    opt_objective = soln.f

    return opt_mixer_params, opt_problem_params, opt_objective

def bobyqa_minimize_interp2(func, layers, mixer_param_vals_init, \
    problem_param_vals_init, noisy, max_for_global=None):

    n_mixer_vals = len(mixer_param_vals_init)
    n_problem_vals = len(mixer_param_vals_init)

    if max_for_global is None:
        seek_global = False
    else:
        seek_global = True
        bounds = ( np.array([0.0]*(n_mixer_vals+n_problem_vals)), np.array([max_for_global]*(n_mixer_vals+n_problem_vals)) )

    def cost_function(params):
        mixer_points = ()
        for j in range(n_mixer_vals):
            mixer_points += ((j/(n_mixer_vals-1), params[j]),)
        problem_points = ()
        for j in range(n_mixer_vals, n_mixer_vals+n_problem_vals):
            problem_points += (((j-n_mixer_vals)/(n_problem_vals-1), params[j]),)
        mixer_points = tuple(sorted(mixer_points, key=lambda x: x[0]))
        problem_points = tuple(sorted(problem_points, key=lambda x: x[0]))

        mixer_params = schedule(mixer_points, layers)
        problem_params = schedule(problem_points, layers)
        return func(mixer_params, problem_params)

    initial_position = []
    for j in range(n_mixer_vals):
        initial_position += [mixer_param_vals_init[j]]
    for j in range(n_problem_vals):
        initial_position += [problem_param_vals_init[j]]
    initial_position = np.array(initial_position)

    params = initial_position

    if seek_global:
        soln = pybobyqa.solve(cost_function, initial_position, bounds=bounds, objfun_has_noise=noisy, seek_global_minimum=seek_global)
    else:
        soln = pybobyqa.solve(cost_function, initial_position, objfun_has_noise=noisy, seek_global_minimum=seek_global)

    opt_params = np.array(soln.x)
    mixer_points = ()
    for j in range(n_mixer_vals):
        mixer_points += ((j/(n_mixer_vals-1), opt_params[j]),)
    problem_points = ()
    for j in range(n_mixer_vals, n_mixer_vals+n_problem_vals):
        problem_points += (((j-n_mixer_vals)/(n_mixer_vals-1), opt_params[j]),)
    mixer_points = tuple(sorted(mixer_points, key=lambda x: x[0]))
    problem_points = tuple(sorted(problem_points, key=lambda x: x[0]))
    opt_mixer_params = schedule(mixer_points, layers)
    opt_problem_params = schedule(problem_points, layers)

    opt_objective = soln.f
    opt_mixer_vals, opt_problem_vals = opt_params[:n_mixer_vals], opt_params[n_mixer_vals:n_mixer_vals+n_problem_vals]

    extra_output = (opt_mixer_vals, opt_problem_vals)

    return opt_mixer_params, opt_problem_params, opt_objective, extra_output

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

def bobyqa_minimize_fourier(func, layers, mixer_modes_init, problem_modes_init, noisy, max_for_global=None):

    n_mixer_modes = len(mixer_modes_init)
    n_problem_modes = len(problem_modes_init)

    if max_for_global is None:
        seek_global = False
    else:
        seek_global = True
        bounds = ( np.array([0.0]*(n_mixer_modes+n_problem_modes)), np.array([max_for_global]*(n_mixer_modes+n_problem_modes)) )

    def cost_function(params):
        mixer_modes = params[:n_mixer_modes]
        problem_modes = params[n_mixer_modes:n_mixer_modes+n_problem_modes]
        mixer_params, problem_params = [], []
        for j in range(1, layers+1):
            mixer_param = np.sum([mixer_modes[k-1]*np.cos( (k-0.5)*(j-0.5)*(np.pi/layers) ) for k in range(1, n_mixer_modes+1)])
            problem_param = np.sum([problem_modes[k-1]*np.sin( (k-0.5)*(j-0.5)*(np.pi/layers) ) for k in range(1, n_problem_modes+1)])
            mixer_params.append(mixer_param)
            problem_params.append(problem_param)
        mixer_params, problem_params = np.array(mixer_params), np.array(problem_params)
        return func(mixer_params, problem_params)

    initial_position = np.array(list(mixer_modes_init) + list(problem_modes_init))

    maxfun = 100*(n_mixer_modes+n_problem_modes+1)

    if seek_global:
        soln = pybobyqa.solve(cost_function, initial_position, bounds=bounds, objfun_has_noise=noisy, seek_global_minimum=seek_global, maxfun=maxfun)
    else:
        soln = pybobyqa.solve(cost_function, initial_position, objfun_has_noise=noisy, seek_global_minimum=seek_global, maxfun=maxfun)

    opt_params = np.array(soln.x)
    opt_mixer_modes = opt_params[:n_mixer_modes]
    opt_problem_modes = opt_params[n_mixer_modes:n_mixer_modes+n_problem_modes]
    opt_mixer_params, opt_problem_params = [], []
    for j in range(1, layers+1):
        mixer_param = np.sum([opt_mixer_modes[k-1]*np.cos( (k-0.5)*(j-0.5)*(np.pi/layers) ) for k in range(1, n_mixer_modes+1)])
        problem_param = np.sum([opt_problem_modes[k-1]*np.sin( (k-0.5)*(j-0.5)*(np.pi/layers) ) for k in range(1, n_problem_modes+1)])
        opt_mixer_params.append(mixer_param)
        opt_problem_params.append(problem_param)
    opt_mixer_params, opt_problem_params = np.array(opt_mixer_params), np.array(opt_problem_params)

    opt_objective = soln.f

    extra_output = (opt_mixer_modes, opt_problem_modes)

    return opt_mixer_params, opt_problem_params, opt_objective, extra_output

def spsa_minimize_fourier(func, layers, mixer_modes_init, problem_modes_init, runs):

    n_mixer_modes = len(mixer_modes_init)
    n_problem_modes = len(problem_modes_init)

    def cost_function(params):
        mixer_modes = params[:n_mixer_modes]
        problem_modes = params[n_mixer_modes:n_mixer_modes+n_problem_modes]
        mixer_params, problem_params = [], []
        for j in range(1, layers+1):
            mixer_param = np.sum([mixer_modes[k-1]*np.cos( (k-0.5)*(j-0.5)*(np.pi/layers) ) for k in range(1, n_mixer_modes+1)])
            problem_param = np.sum([problem_modes[k-1]*np.sin( (k-0.5)*(j-0.5)*(np.pi/layers) ) for k in range(1, n_problem_modes+1)])
            mixer_params.append(mixer_param)
            problem_params.append(problem_param)
        mixer_params, problem_params = np.array(mixer_params), np.array(problem_params)
        return func(mixer_params, problem_params)

    initial_position = np.array(list(mixer_modes_init) + list(problem_modes_init))

    perturb = 0.1
    lr = 0.1

    final_state = optim_spsa.minimize(cost_function, initial_position, \
        runs=runs, tolerance=1e-8, max_iterations=2000000, alpha=0.602, \
        lr=lr, perturb=perturb, gamma=0.101, blocking=False, \
        allowed_increase=0.5)

    opt_params = np.array(final_state['best_position'])

    opt_mixer_modes = opt_params[:n_mixer_modes]
    opt_problem_modes = opt_params[n_mixer_modes:n_mixer_modes+n_problem_modes]
    opt_mixer_params, opt_problem_params = [], []
    for j in range(1, layers+1):
        mixer_param = np.sum([opt_mixer_modes[k-1]*np.cos( (k-0.5)*(j-0.5)*(np.pi/layers) ) for k in range(1, n_mixer_modes+1)])
        problem_param = np.sum([opt_problem_modes[k-1]*np.sin( (k-0.5)*(j-0.5)*(np.pi/layers) ) for k in range(1, n_problem_modes+1)])
        opt_mixer_params.append(mixer_param)
        opt_problem_params.append(problem_param)
    opt_mixer_params, opt_problem_params = np.array(opt_mixer_params), np.array(opt_problem_params)

    opt_objective = final_state['best_objective_value']

    extra_output = (opt_mixer_modes, opt_problem_modes)

    return opt_mixer_params, opt_problem_params, opt_objective, extra_output



    
