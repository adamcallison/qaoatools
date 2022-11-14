import numpy as np
import itertools

def minimize(func, bounds, nsteps, tols):
    curr_bounds = [tuple(bound) for bound in bounds]
    while True:
        values = []
        for bound in curr_bounds:
            if type(bound) is tuple:
                values.append(np.linspace(bound[0], bound[1], nsteps))
            else:
                values.append([bound])
        values = tuple(values)
        best_obj = np.inf
        tmp = tuple(list(range(len(variable))) for variable in values)
        for idx in itertools.product(*tmp):
            point = tuple(variable[idx[j]] for j, variable in enumerate(values))
            obj = func(point)
            if obj < best_obj:
                best_obj = obj
                best_idx = idx
        new_bounds = []
        end = True
        for j, variable in enumerate(values):
            if len(variable) == 1:
                continue
            eps = np.abs(variable[1] - variable[0])
            if eps < tols[j]:
                new_bounds.append(variable[best_idx[j]])
            else:
                end = False
                idx = best_idx[j]
                if idx == 0:
                    new_bounds.append((variable[0], variable[1]))
                elif idx == len(variable)-1:
                    new_bounds.append((variable[-2], variable[-1]))
                else:
                    new_bounds.append((variable[idx-1], variable[idx+1]))
        if end:
            break
        else:
            curr_bounds = new_bounds
    best_point = np.array([variable[best_idx[j]] for j, variable in \
        enumerate(values)])
  
    return best_obj, best_point
