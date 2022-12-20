import numpy as np

def run(cost_function, initial_params, param_bounds, stepsize, iterations, runs, max_temperature):
    best_cost = np.inf
    for run in range(runs):
        cost, params = _run(cost_function, initial_params, param_bounds, stepsize, iterations, max_temperature)
        if cost < best_cost:
            best_cost, best_params = cost, params
    return best_cost, best_params

def _run(cost_function, initial_params, param_bounds, stepsize, iterations, max_temperature):
    params = np.array(initial_params)
    n_params = len(params)
    mins = np.array([param_bounds[j][0] for j in range(n_params)])
    maxs = np.array([param_bounds[j][1] for j in range(n_params)])

    cost = cost_function(params)
    best_cost, best_params = cost, np.array(params)
    for iteration in range(1, iterations):
        step = np.random.default_rng().uniform(size=n_params)
        step = 2.0*(step-0.5)
        step = step / np.sqrt(np.dot(step, step))
        step = step*stepsize
        candidate = np.array(params)
        for j in range(n_params):
            if (candidate[j] + step[j]) < mins[j]:
                candidate[j] = mins[j]
            elif (candidate[j] + step[j]) > maxs[j]:
                candidate[j] = maxs[j]
            else:
                candidate[j] += step[j]
        candidate_cost = cost_function(candidate)
        if candidate_cost < cost:
            best_cost, best_params = candidate_cost, np.array(candidate)

        if candidate_cost <= cost:
            accept = True
        else:
            scale = np.log((iterations+1)/(iteration+1))/np.log(iterations+1)
            temperature = max_temperature*scale
            acceptance_probability = np.exp(-(candidate_cost-cost)/temperature)
            test_probability = np.random.default_rng().uniform()
            accept = test_probability <= acceptance_probability

        if accept:
            cost, params = candidate_cost, np.array(candidate)

    return best_cost, best_params