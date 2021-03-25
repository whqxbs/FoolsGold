from __future__ import division
from numpy.linalg import norm
import matplotlib.pyplot as plt
import model_aggregator
import softmax_model_test
import softmax_model_obj
import poisoning_compare

import numpy as np
import utils
import pandas as pd
import pdb
import sys
import heapq
np.set_printoptions(suppress=True)


# Just a simple sandbox for testing out python code, without using Go.
def debug_signal_handler(signal, frame):
    import pdb
    pdb.set_trace()


import signal

signal.signal(signal.SIGINT, debug_signal_handler)


def basic_conv(dataset, num_params, softmax_test, iterations=3000):
    batch_size = 50

    # Global
    softmax_model = softmax_model_obj.SoftMaxModel(dataset, numClasses)

    print("Start training")
    acc_in_iterations = []  # 记录结果
    delta_in_iterations = []
    weights = np.random.rand(num_params) / 100.0

    train_progress = np.zeros(iterations)
    test_progress = np.zeros(iterations)

    for i in range(iterations):
        deltas, _ = softmax_model.privateFun(weights, batch_size=batch_size)
        weights = weights + deltas
        if i % 10 == 0:
            acc_in_iterations.append(
                poisoning_compare.eval(Xtest, ytest, weights, int(from_class), int(to_class),
                                       numClasses, numFeatures, verbose=False))
            if i % 100 == 0:
                print("Train error: %.10f" % softmax_test.train_error(weights))

    print("Done iterations!")
    print("Train error: %d" % softmax_test.train_error(weights))
    print("Test error: %d" % softmax_test.test_error(weights))
    return weights, acc_in_iterations


def no_attack(model_names, numClasses, numParams, softmax_test, iterations=3000):
    # SGD batch size
    batch_size = 50

    # The number of local steps each client takes
    fed_avg_size = 1

    list_of_models = []

    for dataset in model_names[: numClasses]:
        list_of_models.append(softmax_model_obj.SoftMaxModel(dataset, numClasses))

    numClients = len(list_of_models)
    model_aggregator.init(numClients, numParams, numClasses)

    print("\nStart training across " + str(numClients) + " clients with no attack.")

    weights = np.random.rand(numParams) / 100.0
    lr = np.ones(numClients, )
    acc_in_iterations = []
    delta_each_client = []
    train_progress = []
    norm_progress = []
    loss_progress = []

    # The number of previous iterations to use FoolsGold on
    memory_size = 0
    delta_memory = np.zeros((numClients, numParams, memory_size))

    summed_deltas = np.zeros((numClients, numParams))

    for i in range(iterations):

        delta = np.zeros((numClients, numParams))
        losses = np.zeros(numClients)

        ##################################
        # Use significant features filter or not
        ##################################

        # Significant features filter, the top k biggest weights
        # topk = int(numParams / 2)
        # sig_features_idx = np.argpartition(weights, -topk)[-topk:]
        sig_features_idx = np.arange(numParams)

        for k in range(len(list_of_models)):

            delta[k, :], losses[k] = list_of_models[k].privateFun(weights,
                                                                  batch_size=batch_size,
                                                                  num_iterations=fed_avg_size, iter_num=i)

            # normalize delta
            if np.linalg.norm(delta[k, :]) > 1:
                delta[k, :] = delta[k, :] / np.linalg.norm(delta[k, :])

        ##################################
        # Use FoolsGold or something else
        ##################################

        # Use Foolsgold (can optionally clip gradients via Krum)
        # this_delta = model_aggregator.foolsgold(delta, summed_deltas,
        #                                              sig_features_idx, i, weights, clip=1)
        this_delta = np.dot(delta.T, lr)
        # delta_each_client.append(np.hstack((delta[:, 7000], this_delta[7000])))
        # Krum
        # this_delta = model_aggregator.krum(delta, clip=1)

        # Simple Functions
        # this_delta = model_aggregator.average(delta)
        # this_delta = model_aggregator.median(delta)
        # this_delta = model_aggregator.trimmed_mean(delta, 0.2)

        weights = weights + this_delta

        if i % 10 == 0:
            norm_progress.append(np.mean(np.linalg.norm(delta, axis=1)))
            test_error = softmax_test.test_error(weights)
            train_progress.append(test_error)
            acc_in_iterations.append(
                [test_error] + list(poisoning_compare.eval(Xtest, ytest, weights, int(from_class), int(to_class),
                                                           numClasses, numFeatures, verbose=False)))

            # if i % 100 == 0:
            #     print("Validation error: %.5f" % test_error)
    # pd.DataFrame(columns=['client{}'.format(i) for i in range(15)] + ['combined'], data=delta_each_client).to_csv(
    #     'delta.csv')
    test_error = softmax_test.test_error(weights)
    acc_in_iterations.append(
        [test_error] + list(poisoning_compare.eval(Xtest, ytest, weights, int(from_class), int(to_class),
                                                   numClasses, numFeatures, verbose=True)))
    column = ['iteration', 'Test error', 'Accuracy overall', 'Accuracy on other digits',
              'Target Accuracy on source label',
              'Target Accuracy on target label', 'Target Attack Rate']
    acc_in_iterations = np.insert(acc_in_iterations, 0, values=np.arange(0, iterations + 1, 10), axis=1)
    res = pd.DataFrame(columns=column, data=acc_in_iterations)
    res.to_csv('_'.join(argv[:2]) + '_no_attack.csv')
    print("Done iterations!")
    print("Train error: {}".format(softmax_test.train_error(weights)))
    print("Test error: {}".format(softmax_test.test_error(weights)))
    return weights, norm_progress, train_progress, acc_in_iterations


def non_iid(model_names, numClasses, numParams, softmax_test, iterations=3000, solution=None,
            ideal_attack=False):
    # SGD batch size
    batch_size = 50

    # The number of local steps each client takes
    fed_avg_size = 1

    list_of_models = []

    for dataset in model_names:
        list_of_models.append(softmax_model_obj.SoftMaxModel(dataset, numClasses))

    # Include the model that sends the ideal vector on each iteration
    if ideal_attack:
        list_of_models.append(softmax_model_obj.SoftMaxModelEvil(dataPath +
                                                                 "_bad_ideal_4_9", numClasses))

    numClients = len(list_of_models)
    model_aggregator.init(numClients, numParams, numClasses)

    print("\nStart training across " + str(numClients) + " clients with solution "+str(solution)+'.')

    weights = np.random.rand(numParams) / 100.0
    lr = np.ones(numClients, )
    acc_in_iterations = []
    delta_all = []
    train_progress = []
    norm_progress = []
    loss_progress = []

    # The number of previous iterations to use FoolsGold on
    memory_size = 0
    delta_memory = np.zeros((numClients, numParams, memory_size))

    summed_deltas = np.zeros((numClients, numParams))

    for i in range(iterations):

        delta = np.zeros((numClients, numParams))
        losses = np.zeros(numClients)

        ##################################
        # Use significant features filter or not
        ##################################
        
        # Significant features filter, the top k biggest weights
        # topk = int(numParams / 2)
        # sig_features_idx = np.argpartition(weights, -topk)[-topk:]
        sig_features_idx = np.arange(numParams)

        ##################################
        # Use history or not
        ##################################

        if memory_size > 0:

            for k in range(len(list_of_models)):
            
                delta[k, :], losses[k] = list_of_models[k].privateFun(weights,
                   batch_size=batch_size, num_iterations=fed_avg_size)

                # normalize delta
                if np.linalg.norm(delta[k, :]) > 1:
                    delta[k, :] = delta[k, :] / np.linalg.norm(delta[k, :])

                delta_memory[k, :, i % memory_size] = delta[k, :]

            # Track the total vector from each individual client
            summed_deltas = np.sum(delta_memory, axis=2)

        else:

            for k in range(len(list_of_models)):
    
                delta[k, :], losses[k] = list_of_models[k].privateFun(weights, 
                    batch_size=batch_size, num_iterations=fed_avg_size, iter_num=i)

                # normalize delta
                if np.linalg.norm(delta[k, :]) > 1:
                    delta[k, :] = delta[k, :] / np.linalg.norm(delta[k, :])

            # Track the total vector from each individual client
            summed_deltas = summed_deltas + delta
        
        ##################################
        # Use FoolsGold or something else
        ##################################
        if solution:
            if solution == 'fg':
                # Use Foolsgold (can optionally clip gradients via Krum)
                this_delta = model_aggregator.foolsgold(delta, summed_deltas,
        this_delta = model_aggregator.foolsgold(delta, summed_deltas, 
                this_delta = model_aggregator.foolsgold(delta, summed_deltas,
        this_delta = model_aggregator.foolsgold(delta, summed_deltas, 
                this_delta = model_aggregator.foolsgold(delta, summed_deltas,
        this_delta = model_aggregator.foolsgold(delta, summed_deltas, 
                this_delta = model_aggregator.foolsgold(delta, summed_deltas,
        this_delta = model_aggregator.foolsgold(delta, summed_deltas, 
                this_delta = model_aggregator.foolsgold(delta, summed_deltas,
        this_delta = model_aggregator.foolsgold(delta, summed_deltas, 
                this_delta = model_aggregator.foolsgold(delta, summed_deltas,
        this_delta = model_aggregator.foolsgold(delta, summed_deltas, 
                this_delta = model_aggregator.foolsgold(delta, summed_deltas,
                                                        sig_features_idx, i, weights, clip=1)
            if solution == 'ours':
                this_delta, lr = model_aggregator.foolsgold2(delta, summed_deltas,
                                                             sig_features_idx, i, weights, lr, clip=0)
            if solution == 'krum':
                # Krum
                this_delta = model_aggregator.krum(delta, clip=1)
            if solution == 'average':
                this_delta = model_aggregator.average(delta)
            if solution == 'median':
                this_delta = model_aggregator.median(delta)
            if solution == 'trimmed_mean':
                this_delta = model_aggregator.trimmed_mean(delta, 0.2)
        else:
            this_delta = np.dot(delta.T, lr)

        # Simple Functions
        # this_delta = model_aggregator.average(delta)
        # this_delta = model_aggregator.median(delta)
        # this_delta = model_aggregator.trimmed_mean(delta, 0.2)

        weights = weights + this_delta

        if i % 10 == 0:
            delta_index = heapq.nlargest(20, range(len(this_delta)), this_delta.take)
            delta_each_client = []
            for idx in delta_index:
                delta_each_client.append(np.hstack(([i, idx], delta[:, idx], this_delta[idx])))
            delta_all.append(delta_each_client)
            norm_progress.append(np.mean(np.linalg.norm(delta, axis=1)))
            test_error = softmax_test.test_error(weights)
            train_progress.append(test_error)
            # acc_in_iterations.append(
            #     [test_error] + list(poisoning_compare.eval(Xtest, ytest, weights, int(from_class), int(to_class),
            #                                                numClasses, numFeatures, verbose=False)))

            # if i % 100 == 0:
            #     print("Validation error: %.5f" % test_error)
    column = ['iteration', 'deltaInxex'] + ['client{}'.format(i) for i in range(numClients)] + ['combined']
    pd.DataFrame(columns=column,
                 data=np.reshape(delta_all, (-1, len(column)))).to_csv('_'.join(argv) + '_' + str(solution) + '_delta.csv')
    test_error = softmax_test.test_error(weights)
    # acc_in_iterations.append(
    #     [test_error] + list(poisoning_compare.eval(Xtest, ytest, weights, int(from_class), int(to_class),
    #                                                numClasses, numFeatures, verbose=True)))
    # column = ['iteration', 'Test error', 'Accuracy overall', 'Accuracy on other digits',
    #           'Target Accuracy on source label',
    #           'Target Accuracy on target label', 'Target Attack Rate']
    # acc_in_iterations = np.insert(acc_in_iterations, 0, values=np.arange(0, iterations + 1, 10), axis=1)
    # res = pd.DataFrame(columns=column, data=acc_in_iterations)
    # res.to_csv('_'.join(argv) + '_' + str(solution) + '.csv')
    print("Done iterations!")
    print("Train error: {}".format(softmax_test.train_error(weights)))
    print("Test error: {}".format(softmax_test.test_error(weights)))
    return weights, norm_progress, train_progress, acc_in_iterations


def setup_clients(datapath, num_classes, attack_configs):
    models = []

    for i in range(num_classes):
        # If using uniform clients
        # models.append(datapath + "_uniform_" + str(i))
        models.append(datapath + str(i))

    for attack in attack_configs:
        sybils = attack['sybils']
        from_class = attack['from']
        to_class = attack['to']

        if from_class == "b":
            for i in range(sybils):
                models.append(datapath + "_backdoor_" + str(to_class))
        elif from_class == "u":
            for i in range(sybils):
                models.append(datapath + "_untargeted_" + str(to_class))
        else:
            for i in range(sybils):
                models.append(datapath + "_bad_" + str(from_class) + "_" + str(to_class))

    return models


# amazon: 50 classes, 10000 features
# mnist: 10 classes, 784 features
# kdd: 23 classes, 41 features
if __name__ == "__main__":
    argv = sys.argv[1:]

    dataset = argv[0]
    iterations = int(argv[1])

    if dataset == "mnist":
        numClasses = 10
        numFeatures = 784
    elif dataset == "kddcup":
        numClasses = 23
        numFeatures = 41
    elif dataset == "amazon":
        numClasses = 50
        numFeatures = 10000
    else:
        print("Dataset {} not found. Available datasets: mnist kddcup amazon".format(dataset))

    numParams = numClasses * numFeatures
    dataPath = dataset + "/" + dataset

    full_model = softmax_model_obj.SoftMaxModel(dataPath + "_test", numClasses)
    Xtest, ytest = full_model.get_data()
    softmax_test = softmax_model_test.SoftMaxModelTest(dataset, numClasses, numFeatures)

    attack_configs = []
    for attack in argv[2:]:
        attack_delim = attack.split("_")
        attack_configs.append({
            'sybils': int(attack_delim[0]),
            'from': attack_delim[1],
            'to': attack_delim[2]
        })
    from_class = attack_configs[0]['from']
    to_class = attack_configs[0]['to']
    models = setup_clients(dataPath, numClasses, attack_configs)
    print("Clients setup as {}".format(models))

    # FG algorithm
    # weights, norm_prog, train_err_prog, train_each_iter = non_iid(models, numClasses, numParams, softmax_test,
    #                                                               iterations,
    #                                                               ideal_attack=False)
    # no_attack(models, numClasses, numParams, softmax_test, iterations)
    non_iid(models, numClasses, numParams, softmax_test, iterations, ideal_attack=False, solution=None)
    non_iid(models, numClasses, numParams, softmax_test, iterations, ideal_attack=False, solution='ours')
    non_iid(models, numClasses, numParams, softmax_test, iterations, ideal_attack=False, solution='krum')
    non_iid(models, numClasses, numParams, softmax_test, iterations, ideal_attack=False, solution='median')
    non_iid(models, numClasses, numParams, softmax_test, iterations, ideal_attack=False, solution='trimmed_mean')

    # np.savetxt("norm_progress_baseline_20K.csv", norm_prog)
    # np.savetxt("train_progress_baseline_20K.csv", train_err_prog)

    # for attack in attack_configs:
    #
    #     to_class = attack['to']
    #     from_class = attack['from']
    #
    #     if from_class == "b":
    #         backdoor_model = softmax_model_obj.SoftMaxModel(dataPath + "_backdoor_test", numClasses)
    #         Xback, yback = backdoor_model.get_data()
    #         score = poisoning_compare.backdoor_eval(Xback, yback, weights, int(to_class), numClasses, numFeatures)
    #     elif from_class == "u":
    #         # Just send dummy values to the model. Ignore the poisoning results
    #         score = poisoning_compare.eval(Xtest, ytest, weights, int(1), int(7), numClasses, numFeatures)
    #     else:
    #         # score = poisoning_compare.eval(Xtest, ytest, weights, int(from_class), int(to_class), numClasses,
    #         #                                numFeatures)
    #         train_each_iter.append(
    #             poisoning_compare.eval(Xtest, ytest, weights, int(from_class), int(to_class), numClasses,
    #                                    numFeatures))
    #
    # # Sandbox: difference between ideal bad model and global model
    # compare = True
    # if compare:
    #     # bad_weights, train_each_iter_no_defense = basic_conv(dataPath + "_bad_" + from_class + "_" +
    #     #                                                      to_class, numParams, softmax_test, iterations=iterations)
    #     bad_weights, _, _, train_each_iter_no_defense = no_defense(models, numClasses, numParams, softmax_test,
    #                                                                iterations,
    #                                                                ideal_attack=False)
    #     train_each_iter_no_defense.append(poisoning_compare.eval(Xtest, ytest, bad_weights, int(from_class),
    #                                                              int(to_class), numClasses, numFeatures))
    #
    #     diff = np.reshape(bad_weights - weights, (numClasses, numFeatures))
    #     abs_diff = np.reshape(np.abs(bad_weights - weights), (numClasses,
    #                                                           numFeatures))

    # acc_each_iter1 = np.array(train_each_iter)[:, 0]
    # atk_rate1 = np.array(train_each_iter)[:, -1]
    # acc_each_iter2 = np.array(train_each_iter_no_defense)[:, 0]
    # atk_rate2 = np.array(train_each_iter_no_defense)[:, -1]
    # plt.plot(np.arange(1, iterations + 1, 10), acc_each_iter1)
    # plt.plot(np.arange(1, iterations + 1, 10), acc_each_iter2)
    # plt.plot(np.arange(1, iterations + 1, 10), atk_rate1)
    # plt.plot(np.arange(1, iterations + 1, 10), atk_rate2)
    # plt.show()
