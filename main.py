from original_code.synflow.main import get_args as synflow_get_args
from original_code.synflow.Experiments import singleshot


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
        model="resnet20"
        )
    state = State()
    singleshot.run(state, args)
    print("test")
