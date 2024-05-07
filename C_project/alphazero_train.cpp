#include <torch/torch.h>
#include "lib/game.h"
#include <iostream>
#include "lib/params.h"
#include "lib/alphazero.h"
#include <chrono>
#include <torch_tensorrt/logging.h>

int main(int argc, const char* argv[]) {
    AlphazeroParams alphazero_params;
    alphazero_params.run_name = "alphazero_connect4_resnet";
    alphazero_params.n_selfplay_iterations = 100;
    alphazero_params.num_searches = 800;
    alphazero_params.batch_size = 64;
    alphazero_params.temperature = 1.0;
    alphazero_params.dirichlet_epsilon = 0.25;
    alphazero_params.dirichlet_alpha = 1.0;
    alphazero_params.c_puct = 4.0;
    alphazero_params.num_threads_mcts = 16;
    alphazero_params.num_threads_games = 20;

    if (argc != 3) {
        std::cerr << "usage: alphazero_train <iteration_idx> <process_idx>\n";
        return -1;
    }
    
    Game game;

    int iteration_idx = std::stoi(argv[1]);
    int process_idx = std::stoi(argv[2]);

    if (torch::cuda::is_available()) {
        std::cout << "CUDA is available! Training on GPU." << std::endl;
    } else {
        std::cout << "CUDA is not available. Training on CPU." << std::endl;
    }
    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;

    torch_tensorrt::logging::set_reportable_log_level(torch_tensorrt::logging::Level::kWARNING);

    AlphaZero alphazero(game, alphazero_params, iteration_idx, process_idx, device);
    alphazero.pipeline_self_play();

    return 0;
}