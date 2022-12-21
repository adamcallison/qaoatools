import pybobyqa
import numpy as np
import scipy.interpolate as spi

import optim_spsa
import optim_sa

def spsa_minimize_old(func, mixer_param_init, problem_param_init, runs):
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

def bobyqa_minimize_old(func, mixer_param_init, problem_param_init, noisy, maxfun=None, max_for_global=None):
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
        soln = pybobyqa.solve(cost_function, initial_position, bounds=bounds, objfun_has_noise=noisy, seek_global_minimum=seek_global, maxfun=maxfun)
    else:
        soln = pybobyqa.solve(cost_function, initial_position, objfun_has_noise=noisy, seek_global_minimum=seek_global, maxfun=maxfun)

    opt_params = np.array(soln.x)

    opt_mixer_params = opt_params[:layers]
    opt_problem_params = opt_params[layers:]

    opt_objective = soln.f

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

def bobyqa_minimize_interp2_old(func, layers, mixer_param_vals_init, \
    problem_param_vals_init, noisy, maxfun=None, max_for_global=None):

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
        soln = pybobyqa.solve(cost_function, initial_position, bounds=bounds, objfun_has_noise=noisy, seek_global_minimum=seek_global, maxfun=maxfun)
    else:
        soln = pybobyqa.solve(cost_function, initial_position, objfun_has_noise=noisy, seek_global_minimum=seek_global, maxfun=maxfun)

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

def optimizer_minimize_all(param_type, optimizer, func, layers, mixer_init, problem_init, optimizer_params):
    assert param_type in ('standard', 'interp', 'interp2', 'fourier')
    assert optimizer in ('spsa', 'bobyqa', 'sa')

    if param_type == 'standard':
        mixer_param_init, problem_param_init = mixer_init, problem_init
        assert (len(mixer_param_init), len(problem_param_init)) == (layers, layers)
        initial_position = np.array(list(mixer_param_init) + list(problem_param_init))

        def cost_function(params):
            mixer_params = params[:layers]
            problem_params = params[layers:]
            return func(mixer_params, problem_params)

    if param_type == 'interp':
        mixer_param_points_init, problem_param_points_init = mixer_init, problem_init

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

    if param_type == 'interp2':
        mixer_param_vals_init, problem_param_vals_init = mixer_init, problem_init

        n_mixer_vals = len(mixer_param_vals_init)
        n_problem_vals = len(mixer_param_vals_init)

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

    if param_type == 'fourier':
        mixer_modes_init, problem_modes_init = mixer_init, problem_init

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

    if optimizer == 'bobyqa':
        noisy = optimizer_params.get('noisy', None)
        maxfun = optimizer_params.get('maxfun', None)
        max_for_global = optimizer_params.get('max_for_global', None)
        restart = optimizer_params.get('restart', False)

        if restart and (maxfun is None):
            raise ValueError

        if max_for_global is None:
            seek_global = False
        else:
            seek_global = True
            bounds = ( np.array([0.0]*len(initial_position)), np.array([max_for_global]*len(initial_position)) )

        best_opt_objective, curr_nf = np.inf, 0
        while True:
            maxfun_use = None if maxfun is None else (maxfun - curr_nf)
            if seek_global:
                soln = pybobyqa.solve(cost_function, initial_position, bounds=bounds, objfun_has_noise=noisy, seek_global_minimum=seek_global, maxfun=maxfun_use)
            else:
                soln = pybobyqa.solve(cost_function, initial_position, objfun_has_noise=noisy, seek_global_minimum=seek_global, maxfun=maxfun_use)
            opt_params = np.array(soln.x)
            opt_objective = soln.f
            curr_nf += soln.nf
            if opt_objective < best_opt_objective:
                best_opt_params, best_opt_objective = opt_params, opt_objective
            if (not restart) or (curr_nf >= maxfun):
                break
            else:
                print('')
                print("Restarting!!!    ")
        opt_params, opt_objective = best_opt_params, best_opt_objective

    if optimizer == 'spsa':
        runs = optimizer_params.get('runs', 1)

        perturb = 0.01
        lr = 0.01

        max_iterations = 2000000
        
        final_state = optim_spsa.minimize(cost_function, initial_position, \
            runs=runs, tolerance=1e-8, max_iterations=max_iterations, alpha=0.602, \
            lr=lr, perturb=perturb, gamma=0.101, blocking=False, \
            allowed_increase=0.5)

        opt_params = np.array(final_state['best_position'])
        opt_objective = final_state['best_objective_value']

    if optimizer == 'sa':
        param_max = optimizer_params.get('param_max', np.pi)
        bounds = [(0, param_max) for j in range(2*layers)]
        stepsize = optimizer_params.get('stepsize', 0.05)
        iterations = optimizer_params.get('iterations', 1000)
        runs = optimizer_params.get('runs', 1)
        max_temperature = optimizer_params.get('runs', 10)

        opt_objective, opt_params = optim_sa.run(cost_function, initial_position, bounds, stepsize, iterations, runs, max_temperature)

    if param_type == 'standard':
        opt_mixer_params = opt_params[:layers]
        opt_problem_params = opt_params[layers:]

        extra_output = ()

    if param_type == 'interp':
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

        opt_mixer_points = [(opt_params[2*j], opt_params[(2*j)+1]) for j in range(n_mixer_points)]
        opt_problem_points = [(opt_params[(2*n_mixer_points)+(2*j)], opt_params[(2*n_mixer_points)+((2*j)+1)]) for j in range(n_problem_points)]
        extra_output = (opt_mixer_points, opt_problem_points)

    if param_type == 'interp2':
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

        opt_mixer_vals, opt_problem_vals = opt_params[:n_mixer_vals], opt_params[n_mixer_vals:n_mixer_vals+n_problem_vals]
        extra_output = (opt_mixer_vals, opt_problem_vals)

    if param_type == 'fourier':
        opt_mixer_modes = opt_params[:n_mixer_modes]
        opt_problem_modes = opt_params[n_mixer_modes:n_mixer_modes+n_problem_modes]
        opt_mixer_params, opt_problem_params = [], []
        for j in range(1, layers+1):
            mixer_param = np.sum([opt_mixer_modes[k-1]*np.cos( (k-0.5)*(j-0.5)*(np.pi/layers) ) for k in range(1, n_mixer_modes+1)])
            problem_param = np.sum([opt_problem_modes[k-1]*np.sin( (k-0.5)*(j-0.5)*(np.pi/layers) ) for k in range(1, n_problem_modes+1)])
            opt_mixer_params.append(mixer_param)
            opt_problem_params.append(problem_param)

        opt_mixer_params, opt_problem_params = np.array(opt_mixer_params), np.array(opt_problem_params)
        extra_output = (opt_mixer_modes, opt_problem_modes)

    return opt_mixer_params, opt_problem_params, opt_objective, extra_output

def bobyqa_minimize_all(param_type, func, layers, mixer_init, problem_init, noisy=None, maxfun=None, max_for_global=None, restart=False):
    optimizer = 'bobyqa'
    optimizer_params = {'noisy' : noisy, 'maxfun' : maxfun, 'max_for_global' : max_for_global, 'restart' : restart}
    return optimizer_minimize_all(param_type, optimizer, func, layers, mixer_init, problem_init, optimizer_params)

def spsa_minimize_all(param_type, func, layers, mixer_init, problem_init, runs=1):
    optimizer = 'spsa'
    optimizer_params = {'runs' : runs}
    return optimizer_minimize_all(param_type, optimizer, func, layers, mixer_init, problem_init, optimizer_params)

def sa_minimize_all(param_type, func, layers, mixer_init, problem_init, param_max, stepsize, iterations, runs, max_temperature):
    optimizer = 'sa'
    optimizer_params = {'param_max' : param_max, 'stepsize' : stepsize, 'iterations' : iterations, 'runs' : runs, 'max_temperature' : max_temperature}
    return optimizer_minimize_all(param_type, optimizer, func, layers, mixer_init, problem_init, optimizer_params)

def bobyqa_minimize(func, layers, mixer_param_init, problem_param_init, noisy=None, maxfun=None, max_for_global=None, restart=False):
    return bobyqa_minimize_all('standard', func, layers, mixer_param_init, problem_param_init, noisy, maxfun=maxfun, max_for_global=max_for_global, restart=restart)

def bobyqa_minimize_interp(func, layers, mixer_param_points_init, problem_param_points_init, noisy=None, maxfun=None, max_for_global=None, restart=False):
    return bobyqa_minimize_all('interp', func, layers, mixer_param_points_init, problem_param_points_init, noisy, maxfun=maxfun, max_for_global=max_for_global, \
        restart=restart)

def bobyqa_minimize_interp2(func, layers, mixer_param_vals_init, problem_param_vals_init, noisy=None, maxfun=None, max_for_global=None, restart=False):
    return bobyqa_minimize_all('interp2', func, layers, mixer_param_vals_init, problem_param_vals_init, noisy, maxfun=maxfun, max_for_global=max_for_global, \
        restart=restart)

def bobyqa_minimize_fourier(func, layers, mixer_modes_init, problem_modes_init, noisy=None, maxfun=None, max_for_global=None, restart=False):
    return bobyqa_minimize_all('fourier', func, layers, mixer_modes_init, problem_modes_init, noisy, maxfun=maxfun, max_for_global=max_for_global, \
        restart=restart)

def spsa_minimize(func, layers, mixer_param_init, problem_param_init, runs=1):
    return spsa_minimize_all('standard', func, layers, mixer_param_init, problem_param_init, runs=runs)

def spsa_minimize_interp(func, layers, mixer_param_points_init, problem_param_points_init, runs=1):
    return spsa_minimize_all('interp', func, layers, mixer_param_points_init, problem_param_points_init, runs=runs)

def spsa_minimize_interp2(func, layers, mixer_param_vals_init, problem_param_vals_init, runs=1):
    return spsa_minimize_all('interp2', func, layers, mixer_param_vals_init, problem_param_vals_init, runs=runs)

def spsa_minimize_fourier(func, layers, mixer_modes_init, problem_modes_init, runs=1):
    return spsa_minimize_all('fourier', func, layers, mixer_modes_init, problem_modes_init, runs=runs)

def sa_minimize(func, layers, mixer_init, problem_init, param_max, stepsize, iterations, runs, max_temperature):
    return sa_minimize_all('standard', func, layers, mixer_init, problem_init, param_max, stepsize, iterations, runs, max_temperature)

def sa_minimize_interp(func, layers, mixer_init, problem_init, param_max, stepsize, iterations, runs, max_temperature):
    return sa_minimize_all('interp', func, layers, mixer_init, problem_init, param_max, stepsize, iterations, runs, max_temperature)

def sa_minimize_interp2(func, layers, mixer_init, problem_init, param_max, stepsize, iterations, runs, max_temperature):
    return sa_minimize_all('interp2', func, layers, mixer_init, problem_init, param_max, stepsize, iterations, runs, max_temperature)

def sa_minimize_fourier(func, layers, mixer_init, problem_init, param_max, stepsize, iterations, runs, max_temperature):
    return sa_minimize_all('fourier', func, layers, mixer_init, problem_init, param_max, stepsize, iterations, runs, max_temperature)


