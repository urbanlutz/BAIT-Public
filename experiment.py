from params import *

from monitor import MONITOR
import run
from pruning import *

def lt_baseline():
    print("doing lt baseline")
    state = State()
    

    data_params = DataParams(
        dataset="mnist",
        train_batch_size = 64,
        test_batch_size = 256,
        prune_batch_size = 256,
        workers = 4,
        prune_dataset_ratio = 10,
        )
    run.prepare_data(state, data_params)

    model_params = ModelParams(
        model_class="lottery", 
        model="lenet_300_100",
        lr = 0.001,
        lr_drop_rate= 0.1,
        lr_drops= [],
        weight_decay = 0.0,
        dense_classifier = False,
        pretrained = False,
        optimizer = "adam",
        )
    run.prepare_model(state, model_params, data_params)


    lth_prune_params = PruningParams(
        strategy="mag",
        sparsity = 0.1,
        prune_epochs = 1,
        prune_bias = False,
        prune_batchnorm = False,
        prune_residual = False,
        compression_schedule = "exponential",
        mask_scope = "global",
        reinitialize = False,
        prune_train_mode = False,
        shuffle = False,
        invert = False,
        )
    lth_prune_result = iterative_pruning(state, lth_prune_params, data_params, iterations=35, training_epochs=100)

    state.save()