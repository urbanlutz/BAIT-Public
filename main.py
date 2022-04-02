import numpy as np
from original_code.synflow.main import get_args as synflow_get_args
import run
from params import *
from pruning import *



def main():

    args = synflow_get_args()
    state = State()
    data_params = DataParams(dataset="cifar10")
    model_params = ModelParams(model_class="lottery", model="resnet20")
    
    run.prepare_data(state, data_params)
    run.prepare_model(state, model_params, data_params)
    

   
    prune_params = PruningParams(strategy="synflow", sparsity = 0.5)
    prune_result = run.one_shot_pruning(state, prune_params, data_params)

    pre_result = run.train(state, epochs=1)

    # continue with lth
    lth_prune_result = iterative_pruning(state, args, )

    # post_result = run.posttrain(state, args)
    # run.display(pre_result, prune_result, post_result)





if __name__ == "__main__":
    main()