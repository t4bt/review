from __future__ import print_function
import argparse
import os

import multiprocessing

import numpy as np
import chainer
from chainer.backends import cuda
from chainer import Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

from sklearn.model_selection import train_test_split

import model
import pickle
import prep


def parse_cmd_args():
    parser = argparse.ArgumentParser(description='Chainer Training:')
    parser.add_argument('--batchsize', '-b', type=int, default=500,
                        help='Number of images in each mini-batch')
    parser.add_argument('--learnrate', '-l', type=float, default=1e-3,
                        help='Learning rate for optimizer')
    parser.add_argument('--epoch', '-e', type=int, default=50,
                        help='Number of sweeps over the dataset to train')
    parser.add_argument('--n_gpus', type=int, default=1,
                        help='Number of GPUs (at least 2)')
    parser.add_argument('--out', '-o', default='result/',
                        help='Directory to output the result')
    parser.add_argument('--no_snapshot', action='store_true',
                        help='Suppress storing snapshots.')
    args = parser.parse_args()

    print('# GPUs: {}'.format(args.n_gpus))
    print('# Minibatch-size: {}'.format(args.batchsize))
    print('# epoch: {}'.format(args.epoch))
    print('')

    return args

def my_concat_examples(batch, device=None, padding=None):
    # 引数batchは可変長バッチ
    # batch = [(array([0,1,2,3]),      0),
    #          (array([6,3,7,3,2,4]),  1),
    #          (array([6,9,8,2,5])),   2)]
    if device is None:
        x = np.array([i[0] for i in batch])
        t = np.array([i[1] for i in batch])
        return x, t
    elif device < 0:
        x = np.array([i[0] for i in batch])
        t = np.array([i[1] for i in batch])
        return cuda.to_cpu(x), cuda.to_cpu(t)
    else:
        xp = cuda.cupy
        x = [cuda.to_gpu(xp.array(i[0], dtype=xp.int32), device) for i in batch]
        t = xp.array([i[1] for i in batch], dtype=xp.float32)
        return x, cuda.to_gpu(t, device)

def prepare_extensions(trainer, evaluator, args):
    trainer.extend(evaluator)

    trainer.extend(extensions.ExponentialShift('lr', 0.97), trigger=(10, 'epoch'))
    trainer.extend(extensions.dump_graph('main/loss'))

    if not args.no_snapshot:
        trainer.extend(extensions.snapshot(), trigger=(args.epoch, 'epoch'))

    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(
        ['epoch', 'main/loss', 'validation/main/loss',
         'main/r2', 'validation/main/r2', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(
        ['main/loss', 'validation/main/loss'],
         'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(
        ['main/r2', 'validation/main/r2'],
         'epoch', file_name='r2.png'))
    trainer.extend(extensions.ProgressBar(update_interval=100))

def train_using_gpu(args, model, docs, label, valid_rate=0.1, lr=1e-3, weight_decay=1e-3):
    if args.n_gpus == 1:
        print('Start a training script using single GPU.')
    else:
        multiprocessing.set_start_method('forkserver')
        print('Start a training script using multiple GPUs.')

    threshold = int(len(label)*(1-valid_rate))
    train = datasets.tuple_dataset.TupleDataset(docs[0:threshold], label[0:threshold])
    valid = datasets.tuple_dataset.TupleDataset(docs[threshold:], label[threshold:])

    if args.n_gpus == 1:
        train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    else:
        train_iter = [chainer.iterators.SerialIterator(sub_train, args.batchsize) \
                      for sub_train in chainer.datasets.split_dataset_n_random(train, args.n_gpus)]
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize,
                                                 repeat=False, shuffle=False)

    master_gpu_id = 0
    if args.n_gpus == 1:
        chainer.cuda.get_device_from_id(master_gpu_id).use()
        model.to_gpu()
    else:
        chainer.cuda.get_device_from_id(master_gpu_id).use()

    optimizer = chainer.optimizers.MomentumSGD(lr=lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    #optimizer.add_hook(chainer.optimizer.Lasso(1e-5))

    if args.n_gpus == 1:
        updater = training.StandardUpdater(train_iter, optimizer,
                                           converter=my_concat_examples, device=0)
    else:
        devices_list = {'main': master_gpu_id}
        devices_list.update({'gpu{}'.format(i): i for i in range(1, args.n_gpus)})
        print(devices_list)
        updater = training.updaters.MultiprocessParallelUpdater(train_iter, optimizer,
                                                                converter=my_concat_examples,
                                                                devices=devices_list)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    evaluator = extensions.Evaluator(valid_iter, model,
                                     converter=my_concat_examples,
                                     device=master_gpu_id)

    prepare_extensions(trainer, evaluator, args)

    trainer.run()

    datasize = len(train) * args.epoch
    throughput = datasize / trainer.elapsed_time
    print('Throughput: {} [docs/sec.] ({} / {})'.format(
        throughput, datasize, trainer.elapsed_time))

    model_filepath = os.path.join(args.out, 'trained.model')
    chainer.serializers.save_npz(model_filepath, model)

def train_using_cpu(args, model, docs, label, valid_rate=0.1, lr=1e-3, weight_decay=1e-3):
    print('Start a training script using single CPU.')

    threshold = int(len(label)*(1-valid_rate))
    train = datasets.tuple_dataset.TupleDataset(docs[0:threshold], label[0:threshold])
    valid = datasets.tuple_dataset.TupleDataset(docs[threshold:], label[threshold:])

    train_iter = chainer.iterators.SerialIterator(train, args.batchsize)
    valid_iter = chainer.iterators.SerialIterator(valid, args.batchsize,
                                                 repeat=False, shuffle=False)

    optimizer = chainer.optimizers.SGD(lr=lr)
    optimizer.setup(model)
    optimizer.add_hook(chainer.optimizer.WeightDecay(weight_decay))
    #optimizer.add_hook(chainer.optimizer.Lasso(1e-5))

    updater = training.StandardUpdater(train_iter, optimizer, converter=my_concat_examples)
    trainer = training.Trainer(updater, (args.epoch, 'epoch'), out=args.out)
    evaluator = extensions.Evaluator(valid_iter, model, converter=my_concat_examples)

    prepare_extensions(trainer, evaluator, args)

    trainer.run()

    datasize = len(train) * args.epoch
    throughput = datasize / trainer.elapsed_time
    print('Throughput: {} [docs/sec.] ({} / {})'.format(
        throughput, datasize, trainer.elapsed_time))

    model_filepath = os.path.join(args.out, 'trained.model')
    chainer.serializers.save_npz(model_filepath, model)

if __name__ == '__main__':
    args = parse_cmd_args()

    with open("target.pkl", "rb") as f:
        target = pickle.load(f)
    #docs, n_words = prep.load.sentences()
    docs, n_words, embed_size, initialW = prep.load.wakati2vector(wakati_file="wakati.txt",
                                                                  vector_file="w2v_interior.vec")

    X_train, X_test, y_train, y_test = train_test_split(docs, target, test_size=0.1, random_state=42)

    print("train_size{}".format(len(X_train)))
    print("test_size{}".format(len(X_test)))

    settings = {
        "n_words":n_words,
        "embed_size":embed_size,
        "hidden_size":128,
        "n_layers":1,
        "initialW":initialW
    }
    M = model.MyRegressor(model.BiLSTMwithAttentionRegressor(**settings))
    if args.n_gpus > 0:
        train_using_gpu(args, M, X_train, y_train, lr=args.learnrate, weight_decay=1e-3)
    else:
        train_using_cpu(args, M, X_train, y_train)
