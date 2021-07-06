#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu May 30 16:53:30 2019

@author: guy.hacohen
"""

import argparse
from argparse import Namespace
from main_train_networks import run_expriment


def curriculum_small_mammals(repeats, output_path=""):
    
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.03,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=100,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)

    
def vanilla_small_mammals(repeats, output_path=""):
    
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="vanilla",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.035,
                     lr_decay_rate=1.8,
                     minimal_lr=1e-4,
                     lr_batch_size=600,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


def anti_curriculum_small_mammals(repeats, output_path=""):
    
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="anti",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.025,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-4,
                     lr_batch_size=200,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


def random_small_mammals(repeats, output_path="", data_path=""):
    
    args = Namespace(dataset="cifar100_subset_16",
                     model='stVGG',
                     output_path=output_path,
                     data_path=data_path,
                     verbose=False,
                     optimizer="sgd",
                     curriculum="random",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.025,
                     lr_decay_rate=1.3,
                     minimal_lr=1e-4,
                     lr_batch_size=400,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


def vanilla_cifar10_st_vgg(repeats, output_path="", data_path="", model="cnn"):
    args = Namespace(dataset="cifar10",
                     model=model,
                     output_path=output_path,
                     data_path=data_path,
                     verbose=True,
                     optimizer="sgd",
                     curriculum="vanilla",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.12,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-3,
                     lr_batch_size=700,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    run_expriment(args)

def curriculum_cifar10_st_vgg(repeats, output_path="", data_path="", model="cnn"):
    args = Namespace(dataset="cifar10",
                     model=model,
                     output_path=output_path,
                     data_path=data_path,
                     verbose=True,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.12,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-3,
                     lr_batch_size=700,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)

def vanilla_cifar100_st_vgg(repeats, output_path="", data_path="", model="cnn"):
    
    args = Namespace(dataset="cifar100",
                     model=model,
                     output_path=output_path,
                     data_path=data_path,
                     verbose=True,
                     optimizer="sgd",
                     curriculum="vanilla",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.12,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-3,
                     lr_batch_size=400,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)

def curriculum_cifar100_st_vgg(repeats, output_path="", data_path="", model="cnn"):
    args = Namespace(dataset="cifar100",
                     model=model,
                     output_path=output_path,
                     data_path=data_path,
                     verbose=True,
                     optimizer="sgd",
                     curriculum="curriculum",
                     batch_size=100,
                     num_epochs=140,
                     learning_rate=0.12,
                     lr_decay_rate=1.1,
                     minimal_lr=1e-3,
                     lr_batch_size=400,
                     batch_increase=100,
                     increase_amount=1.9,
                     starting_percent=0.04,
                     order="inception",
                     test_each=50,
                     repeats=repeats,
                     balance=True)
    
    run_expriment(args)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Try to reproduce the results in the paper")

    parser.add_argument("--repeats", default=1, type=int, help="number of times to repeat the experiment.")
    parser.add_argument("--datapath", type=str, default=None, help="the folder where training data can be found.")
    parser.add_argument("--experiment", type=str, default=None, help="which experiment to run")

    args = parser.parse_args()
    print("input argument: {}".format(args))
    output_path = ""
    num_repeats = args.repeats
    data_path = args.datapath
    
#    import datasets.cifar100
#    
#    dataset = datasets.cifar100.Cifar100(False)
    
    
    #case 2 & 3
    if args.experiment == "vanilla_cifar10":
        vanilla_cifar10_st_vgg(num_repeats, output_path=output_path, data_path=data_path, model='cnn')
    elif args.experiment == "curriculum_cifar10":
        curriculum_cifar10_st_vgg(num_repeats, output_path=output_path, data_path=data_path, model='cnn')
    elif args.experiment == "vanilla_cifar100":
        vanilla_cifar100_st_vgg(num_repeats, output_path=output_path, data_path=data_path, model='cnn')
    elif args.experiment == "curriculum_cifar100":
        curriculum_cifar100_st_vgg(num_repeats, output_path=output_path, data_path=data_path, model='cnn')
    
    if args.experiment == "vanilla_cifar10_vgg":
        vanilla_cifar10_st_vgg(num_repeats, output_path=output_path, data_path=data_path, model='vgg')
    elif args.experiment == "curriculum_cifar10_vgg":
        curriculum_cifar10_st_vgg(num_repeats, output_path=output_path, data_path=data_path, model='vgg')
    elif args.experiment == "vanilla_cifar100_vgg":
        vanilla_cifar100_st_vgg(num_repeats, output_path=output_path, data_path=data_path, model='vgg')
    elif args.experiment == "curriculum_cifar100_vgg":
        curriculum_cifar100_st_vgg(num_repeats, output_path=output_path, data_path=data_path, model='vgg')

    if args.experiment == "vanilla_cifar10_svgg":
        vanilla_cifar10_st_vgg(num_repeats, output_path=output_path, data_path=data_path, model='svgg')
    elif args.experiment == "curriculum_cifar10_svgg":
        curriculum_cifar10_st_vgg(num_repeats, output_path=output_path, data_path=data_path, model='svgg')
    elif args.experiment == "vanilla_cifar100_svgg":
        vanilla_cifar100_st_vgg(num_repeats, output_path=output_path, data_path=data_path, model='svgg')
    elif args.experiment == "curriculum_cifar100_svgg":
        curriculum_cifar100_st_vgg(num_repeats, output_path=output_path, data_path=data_path, model='svgg')
    # case 1
#    curriculum_small_mammals(num_repeats, output_path=output_path)
#    vanilla_small_mammals(num_repeats, output_path=output_path)
#    anti_curriculum_small_mammals(num_repeats, output_path=output_path)
#    random_small_mammals(num_repeats, output_path=output_path)
    
    