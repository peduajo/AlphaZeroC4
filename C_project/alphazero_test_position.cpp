#include <torch/torch.h>
#include "lib/game.h"
#include <iostream>
#include "lib/params.h"
#include <chrono>
#include "lib/mcts.h"
#include <torch/script.h>

int main() {
    AlphazeroParams alphazero_params;
    alphazero_params.run_name = "alphazero_connect4_resnet";
    alphazero_params.n_iterations = 500;
    alphazero_params.n_selfplay_iterations = 100;
    alphazero_params.num_searches = 800;
    alphazero_params.batch_size = 128;
    alphazero_params.temperature = 0.05;
    alphazero_params.dirichlet_epsilon = 0.25;
    alphazero_params.dirichlet_alpha = 1.0;
    alphazero_params.c_puct = 2.0;
    alphazero_params.num_threads_mcts = 8;


    Game game;

    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << device << std::endl;

    std::shared_ptr<BufferEvalManager> buffer_eval_manager1 = std::make_shared<BufferEvalManager>(game, device, 8);
    buffer_eval_manager1->set_model("../models/model_27.pt");
    std::thread controller_thread1([buffer_eval_manager1] { buffer_eval_manager1->buffer_eval_controller(); });
    MCTS mcts1(game, alphazero_params, device, buffer_eval_manager1);

    torch::Tensor state_raw = torch::tensor({{ 1, 0, 0, 0, 0, 0, 0},
                                             {-1, 0, 0, 0, 0, 0, 0},
                                             { 1, 0, 0, 0, 0, 0, 0},
                                             {-1, 1, 0, 0, 0, 0,-1},
                                             { 1, 1, 0, 0, 0, 0,-1},
                                             { 1,-1, 1,-1,-1, 1,-1}});

    int current_player = 0;
    
    //auto state = game.change_perspective(state_raw, 1);
    
    auto decoded_state = game.get_encoded_state(state_raw, current_player);
    std::cout << decoded_state << std::endl;
    mcts1.reset_game(state_raw, false, current_player);
    torch::Tensor action_probs = mcts1.search(false);

    std::cout << action_probs << std::endl;

    torch::Tensor temp_action_probs = torch::pow(action_probs, 1.0/alphazero_params.temperature);
    temp_action_probs /= torch::sum(temp_action_probs);

    std::cout << temp_action_probs << std::endl;

    buffer_eval_manager1->terminate_thread_buffer_eval = true;
    if (controller_thread1.joinable()) {
        controller_thread1.join(); // Espera a que el hilo controlador termine
    }
    mcts1.clean_tree();
    return 0;
}