import numpy as np
from original_code.synflow.main import get_args as synflow_get_args
import run
from params import *

from copy import deepcopy



def main():

    args = synflow_get_args()
    state = State()
    data_params = DataParams(dataset="cifar10")
    model_params = ModelParams(model_class="lottery", model="resnet20")
    
    run.prepare_data(state, data_params)
    run.prepare_model(state, model_params, data_params)
    
    # do the synflow thing


    pre_result = run.train(state, epochs=1)
    # sparsity = 10**(-float(args.compression))
    prune_params = PruningParams(strategy="synflow", sparsity = 0.5)
    prune_result = run.prune(state, prune_params, data_params)

    # continue with lth
    lth_prune_result = iterative_pruning(state, args, )

    # post_result = run.posttrain(state, args)
    # run.display(pre_result, prune_result, post_result)



def one_shot_pruning(state, prune_params: PruningParams, data_params: DataParams):
    remaining_params, total_params = prune_params.strategy.stats()
    current_sparsity = remaining_params / total_params
    
    if current_sparsity < prune_params.sparsity:
        print(f"nothing to prune: currently at {current_sparsity}, target is {prune_params.sparsity}")
        return
    
    prune_result = run.prune(state, prune_params, data_params)



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

if __name__ == "__main__":
    main()