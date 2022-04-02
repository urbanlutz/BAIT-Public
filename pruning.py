from copy import deepcopy
import run
from params import *

def one_shot_pruning(state, prune_params: PruningParams, data_params: DataParams):
    remaining_params, total_params = prune_params.strategy.stats()
    current_sparsity = remaining_params / total_params
    
    if current_sparsity < prune_params.sparsity:
        print(f"nothing to prune: currently at {current_sparsity}, target is {prune_params.sparsity}")
        return
    
    prune_result = run.prune(state, prune_params, data_params)
    return prune_result


def iterative_pruning(state, args, strategy="mag", target_sparsity=0.1, iterations=5):
    remaining_params, total_params = args.pruner.stats()
    current_sparsity = remaining_params / total_params
    sparsity_targets = list(np.linspace(current_sparsity, target_sparsity, iterations))
    results = []

    original_state = deepcopy(state.model.state_dict())


    for sparsity in sparsity_targets:

        # train
        result = run.train(state, args, epochs=1)
        results.append(result)

        # prune
        if sparsity < 1:
            prune_result = run.prune(state, args, strategy, sparsity)
            results.append(prune_result)

        # rewind
        original_weights = dict(filter(lambda v: (v[0].endswith(('.weight', '.bias'))), original_state.items()))
        model_dict = state.model.state_dict()
        model_dict.update(original_weights)
        state.model.load_state_dict(model_dict)

    return results