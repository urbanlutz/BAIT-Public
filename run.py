import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from original_code.synflow.Utils import generator
from original_code.synflow.Utils import load
from original_code.synflow.Utils import metrics
from original_code.synflow.train import *
from original_code.synflow.prune import *

from params import *
from environment import ENV

def prepare_data(state: State, params: DataParams):
    print(f'Loading {params.dataset} dataset.')

    state.prune_loader = load.dataloader(params.dataset, params.prune_batch_size, True, params.workers, params.prune_dataset_ratio * params.num_classes)
    state.train_loader = load.dataloader(params.dataset, params.train_batch_size, True, params.workers)
    state.test_loader = load.dataloader(params.dataset, params.test_batch_size, False, params.workers)

def prepare_model(state: State, model_params: ModelParams, data_params: DataParams):
    print(f'Creating {model_params.model_class}-{model_params.model} model.')

    state.model = load.model(model_params.model, model_params.model_class)(data_params.input_shape, 
                                                     data_params.num_classes, 
                                                     model_params.dense_classifier, 
                                                     model_params.pretrained).to(ENV.device)
    state.loss = nn.CrossEntropyLoss()
    opt_class, opt_kwargs = load.optimizer(model_params.optimizer)
    state.optimizer = opt_class(generator.parameters(state.model), lr=model_params.lr, weight_decay=model_params.weight_decay, **opt_kwargs)
    state.scheduler = torch.optim.lr_scheduler.MultiStepLR(state.optimizer, milestones=model_params.lr_drops, gamma=model_params.lr_drop_rate)

def prepare_pruner(state:State, prune_params: PruningParams):
    state.pruner = load.pruner(prune_params.strategy)(generator.masked_parameters(state.model, prune_params.prune_bias, prune_params.prune_batchnorm, prune_params.prune_residual))

def train(state: State, epochs):
    print(f'Train for {epochs} epochs.')
    result = train_eval_loop(state.model, state.loss, state.optimizer, state.scheduler, state.train_loader, state.test_loader, ENV.device, epochs, ENV.verbose)
    return result

def prune(state: State, prune_params: PruningParams, data_params: DataParams):
    print(f'Pruning with {prune_params.strategy} for {prune_params.prune_epochs} epochs.')

            
    prune_loop(state.model, state.loss, state.pruner, state.prune_loader, ENV.device, prune_params.sparsity, 
               prune_params.compression_schedule, prune_params.mask_scope, prune_params.prune_epochs, prune_params.reinitialize, prune_params.prune_train_mode, prune_params.shuffle, prune_params.invert)
    prune_result = metrics.summary(state.model, 
                                state.pruner.scores,
                                metrics.flop(state.model, data_params.input_shape, ENV.device),
                                lambda p: generator.prunable(p, prune_params.prune_batchnorm, prune_params.prune_residual))
    return prune_result


def display(pre_result, prune_result, post_result):
    ## Display Results ##
    frames = [pre_result.head(1), pre_result.tail(1), post_result.head(1), post_result.tail(1)]
    train_result = pd.concat(frames, keys=['Init.', 'Pre-Prune', 'Post-Prune', 'Final'])
    total_params = int((prune_result['sparsity'] * prune_result['size']).sum())
    possible_params = prune_result['size'].sum()
    total_flops = int((prune_result['sparsity'] * prune_result['flops']).sum())
    possible_flops = prune_result['flops'].sum()
    print("Train results:\n", train_result)
    print("Prune results:\n", prune_result)
    print("Parameter Sparsity: {}/{} ({:.4f})".format(total_params, possible_params, total_params / possible_params))
    print("FLOP Sparsity: {}/{} ({:.4f})".format(total_flops, possible_flops, total_flops / possible_flops))

def save(state, args, pre_result, prune_result, post_result):
    ## Save Results and Model ##
    print('Saving results.')
    pre_result.to_pickle("{}/pre-train.pkl".format(args.result_dir))
    post_result.to_pickle("{}/post-train.pkl".format(args.result_dir))
    prune_result.to_pickle("{}/compression.pkl".format(args.result_dir))
    torch.save(state.model.state_dict(),"{}/model.pt".format(args.result_dir))
    torch.save(state.optimizer.state_dict(),"{}/optimizer.pt".format(args.result_dir))
    torch.save(state.scheduler.state_dict(),"{}/scheduler.pt".format(args.result_dir))

