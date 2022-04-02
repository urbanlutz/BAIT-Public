import numpy as np
from original_code.synflow.main import get_args as synflow_get_args
import run
from params import *
from pruning import *



def main():
    state = State()
    data_params = DataParams(dataset="cifar10")
    run.prepare_data(state, data_params)

    model_params = ModelParams(model_class="lottery", model="resnet20")
    run.prepare_model(state, model_params, data_params)

    
    prune_params = PruningParams(strategy="synflow", sparsity = 0.5)
    prune_result = one_shot_pruning(state, prune_params, data_params)

    pre_result = run.train(state, epochs=10)

    lth_prune_params = PruningParams(strategy="mag", sparsity = 0.1)
    lth_prune_result = iterative_pruning(state, lth_prune_params, data_params, iterations=10, training_epochs=5)


if __name__ == "__main__":
    main()