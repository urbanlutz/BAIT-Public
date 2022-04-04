
import run
from params import *
from pruning import *

from monitor import MONITOR

def main():

    state = State()
    

    data_params = DataParams(dataset="mnist")
    MONITOR.track_param("dataset", "mnist")

    run.prepare_data(state, data_params)

    model_params = ModelParams(model_class="lottery", model="lenet_300_100")

    run.prepare_model(state, model_params, data_params)

    
    # prune_params = PruningParams(strategy="synflow", sparsity = 0.5)
    # prune_result = one_shot_pruning(state, prune_params, data_params)

    lth_prune_params = PruningParams(strategy="mag", sparsity = 0.1)
    lth_prune_result = iterative_pruning(state, lth_prune_params, data_params, iterations=35, training_epochs=100)

    state.save()

if __name__ == "__main__":
    main()