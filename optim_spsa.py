import numpy as np
from copy import deepcopy

def minimize(expectation_value_function, initial_position, runs=1, \
    tolerance=1e-5, max_iterations=200, alpha=0.602, lr=1.0, perturb=1.0, \
    gamma=0.101, blocking=False, allowed_increase=0.5):

    minimizer_states = []
    for run in range(runs):
        minimizer_state = _minimize(expectation_value_function, \
            initial_position, tolerance=tolerance, \
            max_iterations=max_iterations, alpha=alpha, lr=lr, \
            perturb=perturb, gamma=gamma, blocking=blocking, \
            allowed_increase=allowed_increase)
        minimizer_states.append(minimizer_state)

    obj_vals = [x['best_objective_value'] for x in minimizer_states]
    idx = np.argmin(obj_vals)
    minimizer_state = minimizer_states[idx]
    return minimizer_state
        
def _get_initial_minimizer_state(initial_position, max_iterations, tolerance, \
    expectation_value_function, lr, alpha, perturb, gamma, blocking, \
    allowed_increase):
    minimizer_state = {
        "converged" : False,
        "num_iterations" : 0,
        "num_objective_evaluations" : 0,
        "position" : np.array(initial_position),
        "objective_value" : 0,
        "objective_value_previous_iteration" : np.inf,
        "best_objective_value" : np.inf,
        "best_position" : None,
        "tolerance" : tolerance,
        "lr" : lr,
        "alpha" : alpha,
        "perturb" : perturb,
        "gamma" : gamma,
        "blocking" : blocking,
        "allowed_increase" : allowed_increase,
        "max_iterations" : max_iterations
    }
    return minimizer_state


def _minimize(expectation_value_function, initial_position, tolerance=1e-5, \
    max_iterations=200, alpha=0.602, lr=1.0, perturb=1.0, gamma=0.101, \
    blocking=False, allowed_increase=0.5):
    
    generator = np.random.default_rng()
    initial_position = np.array(initial_position)
    lr_init = lr
    perturb_init = perturb
    
    def _spsa_once(minimizer_state):
        delta_shift = 2*generator.choice([0.0, 1.0], size=minimizer_state['position'].shape[0]) - 1.0
        v_m = expectation_value_function(minimizer_state['position'] - (minimizer_state['perturb']*delta_shift))
        v_p = expectation_value_function(minimizer_state['position'] + (minimizer_state['perturb']*delta_shift))
        
        gradient_estimate = (v_p - v_m) / (2 * minimizer_state['perturb']) * delta_shift
        update = minimizer_state['lr'] * gradient_estimate
        
        minimizer_state['num_objective_evaluations'] += 2
        
        current_obj = expectation_value_function(minimizer_state['position'] - update)
        if minimizer_state['objective_value_previous_iteration'] + \
            minimizer_state['allowed_increase'] >= current_obj or not minimizer_state['blocking']:
            minimizer_state['position'] = minimizer_state['position'] - update
            minimizer_state['objective_value_previous_iteration'] = minimizer_state['objective_value']
            minimizer_state['objective_value'] = current_obj

        best_call_this_time = np.argmin((v_m, v_p, current_obj))
        if (v_m, v_p, current_obj)[best_call_this_time] < minimizer_state['best_objective_value']:
            if best_call_this_time == 0:
                minimizer_state['best_objective_value'] = v_m
                minimizer_state['best_position'] = minimizer_state['position'] - (minimizer_state['perturb']*delta_shift)
            elif best_call_this_time == 1:
                minimizer_state['best_objective_value'] = v_p
                minimizer_state['best_position'] = minimizer_state['position'] + (minimizer_state['perturb']*delta_shift)
            else:
                minimizer_state['best_objective_value'] = current_obj
                minimizer_state['best_position'] = minimizer_state['position'] - update

        return minimizer_state
    
    def _cond(minimizer_state):
        return (minimizer_state['num_iterations'] < minimizer_state['max_iterations']) and (not minimizer_state['converged'])
    
    def _body(minimizer_state):
        new_lr = lr_init / (  (minimizer_state['num_iterations'] + 1 + (0.01*minimizer_state['max_iterations']))**minimizer_state['alpha']  )
        new_perturb = perturb_init / ((minimizer_state['num_iterations']+1)**minimizer_state['gamma'])
        
        minimizer_state['lr'] = new_lr
        minimizer_state['perturb'] = new_perturb

        _spsa_once(minimizer_state)
        minimizer_state['num_iterations'] += 1
        minimizer_state['converged'] = np.abs(minimizer_state['objective_value'] - minimizer_state['objective_value_previous_iteration']) < minimizer_state['tolerance']

        return minimizer_state

    initial_minimizer_state = _get_initial_minimizer_state(initial_position, max_iterations, tolerance, expectation_value_function, lr, alpha, perturb, gamma, blocking, allowed_increase)
    
    initial_minimizer_state['objective_value'] = expectation_value_function(initial_minimizer_state['position'])
    
    minimizer_state = dict(initial_minimizer_state)
    
    while _cond(minimizer_state):
        _body(minimizer_state)
        
    final_minimizer_state = dict(minimizer_state)
        
    return final_minimizer_state