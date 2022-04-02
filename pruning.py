from copy import deepcopy
import run
import numpy as np
from params import *

def one_shot_pruning(state, prune_params: PruningParams, data_params: DataParams):
    run.prepare_pruner(state, prune_params)
    remaining_params, total_params = state.pruner.stats()
    current_sparsity = remaining_params / total_params
    
    if current_sparsity < prune_params.sparsity:
        print(f"nothing to prune: currently at {current_sparsity}, target is {prune_params.sparsity}")
    
    prune_result = run.prune(state, prune_params, data_params)
    return prune_result


def iterative_pruning(state:State, prune_params: PruningParams, data_params: DataParams, iterations=5):
    run.prepare_pruner(state, prune_params)
    remaining_params, total_params = state.pruner.stats()
    current_sparsity = remaining_params / total_params
    print(f"starting with {current_sparsity * 100}%")

    sparsity_targets = list(np.linspace(current_sparsity, prune_params.sparsity, iterations))
    results = []

    original_state = deepcopy(state.model.state_dict())


    for sparsity in sparsity_targets:
        # train
        result = run.train(state, epochs=1)
        results.append(result)

        # prune
        if sparsity < 1:
            prune_result = run.prune(state, prune_params, data_params)
            results.append(prune_result)

        # rewind
        rewind_weights(state, original_state)

        # display
        remaining_params, total_params = state.pruner.stats()
        current_sparsity = remaining_params / total_params
        print(f"pruned to {current_sparsity * 100}%")

    return results

def rewind_weights(state: State, original_state):
    original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_state.items()))
    model_dict = state.model.state_dict()
    model_dict.update(original_weights)
    state.model.load_state_dict(model_dict)