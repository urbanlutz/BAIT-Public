from original_code.synflow.main import get_args as synflow_get_args
from original_code.synflow.Experiments import singleshot as synflow_runner


class State:
    prune_loader = None
    train_loader = None
    test_loader = None
    model = None
    loss = None
    optimizer = None
    scheduler = None
    input_shape = None
    num_classes = None

if __name__ == "__main__":
    args = synflow_get_args(
        dataset="cifar10", 
        model_class="lottery", 
        model="resnet20",
        pruner="synflow"
        )
    state = State()

    
    # do the synflow thing
    synflow_runner.prepare(state, args)

    pre_result = synflow_runner.pretrain(state, args)
    sparsity = 10**(-float(args.compression))
    prune_result = synflow_runner.prune(state, args, 0.1)

    # continue with lth


    # post_result = synflow_runner.posttrain(state, args)
    # synflow_runner.display(pre_result, prune_result, post_result)

    if args.save:
        synflow_runner.save(state, args)