#pragma once

#include <iostream>

struct AlphazeroParams {
    std::string run_name;
    int n_iterations;
    int n_selfplay_iterations;
    int num_parallel_games;
    int num_searches;
    float learning_rate;
    float weight_decay;
    int batch_size;
    float temperature;
    float dirichlet_epsilon;
    float dirichlet_alpha;
    float c_puct;
    int resnet_num_blocks;
    int resnet_hidden;
    int n_epochs;
    float max_grad_norm;
    float target_kl;
    float coef_policy;
    float coef_value;
    int num_threads_mcts;
    int num_threads_games;
    int num_multiprocessing;
};