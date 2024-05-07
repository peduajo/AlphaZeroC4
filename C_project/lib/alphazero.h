#pragma once

#include <iostream>
#include <torch/script.h>
#include <torch/torch.h>
#include "game.h"
#include "params.h"
#include "mcts.h"
#include <algorithm>
#include <random>
#include <vector>

class AlphaZero{
public:
    Game game;
    AlphazeroParams args;
    torch::Device device;
    std::string base_path_data = "../data/";
    int iteration_idx;
    int process_idx;
    ThreadPool pool_games;
    std::shared_ptr<BufferEvalManager> buffer_eval_manager;
public:
    AlphaZero(const Game& game, const AlphazeroParams& args, int iteration_idx, int process_idx, const torch::Device& device) 
        : game(game),
          args(args),
          iteration_idx(iteration_idx),
          process_idx(process_idx),
          device(device),
          pool_games(args.num_threads_games){

        buffer_eval_manager = std::make_shared<BufferEvalManager>(game, device, args.batch_size);
    }

    void pipeline_self_play();
private:
    void _save_memory(std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> memory);
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> _self_play(bool debug);
    void _generate_self_play_data(std::string model_path);
};