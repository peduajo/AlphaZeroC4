#include <torch/torch.h>
#include "lib/game.h"
#include <iostream>
#include "lib/params.h"
#include <chrono>
#include "lib/mcts.h"
#include <torch/script.h>
#include "lib/mcts_random.h"

int main(int argc, const char* argv[]) {
    if (argc != 4) {
        std::cerr << "usage: alphazero_test <mode> <model 1> <model 2>\n";
        return -1;
    }

    std::string mode = argv[1];
    std::string model1_path = argv[2];
    std::string model2_path = argv[3];

    AlphazeroParams alphazero_params;
    alphazero_params.run_name = "alphazero_connect4_resnet";
    alphazero_params.n_iterations = 500;
    alphazero_params.n_selfplay_iterations = 100;
    alphazero_params.num_searches = 1600;
    alphazero_params.batch_size = 128;
    alphazero_params.temperature = 0.05;
    alphazero_params.dirichlet_epsilon = 0.25;
    alphazero_params.dirichlet_alpha = 1.0;
    alphazero_params.c_puct = 4.0;
    alphazero_params.num_threads_mcts = 8;


    Game game;

    torch::Tensor is = game.get_initial_state();
    auto tiempo_inicio = std::chrono::high_resolution_clock::now();
    auto v = game.get_valid_moves(is);

    auto tiempo_fin = std::chrono::high_resolution_clock::now();
    auto duracion = std::chrono::duration_cast<std::chrono::microseconds>(tiempo_fin - tiempo_inicio);

    // Convierte la duración a segundos
    double segundos = static_cast<double>(duracion.count()) / 1'000'000.0;

    std::cout << "Tiempo gastado en tomar jugadas válidas: " << segundos << std::endl;

    torch::Device device = torch::cuda::is_available() ? torch::kCUDA : torch::kCPU;
    std::cout << device << std::endl;

    if (mode == "self_play"){
        std::shared_ptr<BufferEvalManager> buffer_eval_manager1 = std::make_shared<BufferEvalManager>(game, device, 8);
        buffer_eval_manager1->set_model(model1_path);
        std::thread controller_thread1([buffer_eval_manager1] { buffer_eval_manager1->buffer_eval_controller(); });
        MCTS mcts1(game, alphazero_params, device, buffer_eval_manager1);

        std::shared_ptr<BufferEvalManager> buffer_eval_manager2 = std::make_shared<BufferEvalManager>(game, device, 8);
        buffer_eval_manager2->set_model(model2_path);
        std::thread controller_thread2([buffer_eval_manager2] { buffer_eval_manager2->buffer_eval_controller(); });
        MCTS mcts2(game, alphazero_params, device, buffer_eval_manager2);

        torch::Tensor neutral_state;
        torch::Tensor state;
        torch::Tensor action_probs;
        int action;

        int counts_wins_X = 0;
        int counts_wins_O = 0;
        int counts_draws = 0;

        int matches = 20;
        int total_steps = 0;

        for (int i = 0; i < matches; ++i){
            int step = 0;
            int current_player = 1;
            while (true){
                if (current_player == 1){
                    if (step == 0){
                        torch::Tensor initial_state = game.get_initial_state();
                        state = initial_state;
                        mcts1.reset_game(initial_state, false, current_player);
                    }
                    action_probs = mcts1.search(false);
                    torch::Tensor temp_action_probs = torch::pow(action_probs, 1.0/alphazero_params.temperature);
                    temp_action_probs /= torch::sum(temp_action_probs);
                    torch::Tensor indices = torch::multinomial(temp_action_probs, 1, false);
                    action = indices[0].item<int>();
                    mcts1.update_root(action);
                    if(step > 0){
                        mcts2.update_root(action);
                    }
                    neutral_state = mcts1.root->state;
                }else{
                    mcts2.reset_game(neutral_state, false, current_player);
                    action_probs = mcts2.search(false);
                    torch::Tensor temp_action_probs = torch::pow(action_probs, 1.0/alphazero_params.temperature);
                    temp_action_probs /= torch::sum(temp_action_probs);
                    torch::Tensor indices = torch::multinomial(temp_action_probs, 1, false);
                    action = indices[0].item<int>();
                    mcts2.update_root(action);
                    mcts1.update_root(action);
                    neutral_state = mcts2.root->state;
                }
                state = game.get_next_state(state, action, current_player);

                auto terminated = game.get_value_and_terminated(state, action);
                float value = terminated.first;
                bool is_terminal = terminated.second;
                if (is_terminal){
                    std::cout << neutral_state << std::endl;
                    if (value == 1.0 and current_player == 1){
                        std::cout << "gana X" << std::endl;
                        counts_wins_X++;
                    }else if(value == 1.0 and current_player == -1){
                        std::cout << "gana O" << std::endl;
                        counts_wins_O++;
                    }else{
                        std::cout << "empate" << std::endl;
                        counts_draws++;
                    }
                    mcts1.clean_tree();
                    mcts2.clean_tree();
                    break;
                }
                current_player = game.get_opponent(current_player);
                step++;
            }
            total_steps += step;
        }
        
        float avg_steps = total_steps/matches;
        std::cout << "Victorias X: " << counts_wins_X << std::endl;
        std::cout << "Victorias O: " << counts_wins_O << std::endl;
        std::cout << "Empates: " << counts_draws << std::endl;
        std::cout << "Promedio jugadas: " << avg_steps << std::endl;

        buffer_eval_manager1->terminate_thread_buffer_eval = true;
        if (controller_thread1.joinable()) {
            controller_thread1.join();
        }
        buffer_eval_manager2->terminate_thread_buffer_eval = true;
        if (controller_thread2.joinable()) {
            controller_thread2.join();
        }
    }else if(mode == "against_god"){

        std::shared_ptr<BufferEvalManager> buffer_eval_manager1 = std::make_shared<BufferEvalManager>(game, device, 8);
        buffer_eval_manager1->set_model(model1_path);
        std::thread controller_thread1([buffer_eval_manager1] { buffer_eval_manager1->buffer_eval_controller(); });
        MCTS mcts(game, alphazero_params, device, buffer_eval_manager1);

        torch::Tensor neutral_state;
        torch::Tensor state;
        torch::Tensor action_probs;
        int action;

        int step = 0;
        int current_player = 1;
        while (true){
            if (current_player == 1){
                if (step == 0){
                    torch::Tensor initial_state = game.get_initial_state();
                    state = initial_state;
                }
                if (step == 0){
                    torch::Tensor initial_state = game.get_initial_state();
                    state = initial_state;
                    mcts.reset_game(initial_state, false, current_player);
                }

                action_probs = mcts.search(false);
                torch::Tensor temp_action_probs = torch::pow(action_probs, 1.0/alphazero_params.temperature);
                temp_action_probs /= torch::sum(temp_action_probs);
                torch::Tensor indices = torch::multinomial(temp_action_probs, 1, false);
                action = indices[0].item<int>();
                mcts.update_root(action);
            }else{
                std::cout << "Por favor, introduce un numero entero: ";
                std::cin >> action;
                mcts.update_root(action);
            }
            state = game.get_next_state(state, action, current_player);
            neutral_state = game.change_perspective(state, -1);

            auto terminated = game.get_value_and_terminated(state, action);
            float value = terminated.first;
            bool is_terminal = terminated.second;
            std::cout << neutral_state << std::endl;
            if (is_terminal){
                std::cout << neutral_state << std::endl;
                if (value == 1.0 and current_player == 1){
                    std::cout << "gana X" << std::endl;
                }else if(value == 1.0 and current_player == -1){
                    std::cout << "gana O" << std::endl;
                }else{
                    std::cout << "empate" << std::endl;
                }
                mcts.clean_tree();
                break;
            }
            current_player = game.get_opponent(current_player);
            step++;
        }
        buffer_eval_manager1->terminate_thread_buffer_eval = true;
        if (controller_thread1.joinable()) {
            controller_thread1.join();
        }
    }else{

        std::shared_ptr<BufferEvalManager> buffer_eval_manager1 = std::make_shared<BufferEvalManager>(game, device, 8);
        buffer_eval_manager1->set_model(model1_path);
        std::thread controller_thread1([buffer_eval_manager1] { buffer_eval_manager1->buffer_eval_controller(); });
        MCTS mcts(game, alphazero_params, device, buffer_eval_manager1);
        MCTSRand mctsr(game, 10000);

        torch::Tensor neutral_state;
        torch::Tensor state;
        torch::Tensor action_probs;
        int action;

        int step = 0;
        int current_player = 1;
        while (true){
            if (step == 0){
                torch::Tensor initial_state = game.get_initial_state();
                state = initial_state;
                mctsr.reset_game(initial_state, current_player);
            }else if(step == 1){
                mcts.reset_game(neutral_state, false, current_player);
            }
            if (current_player == 1){
                action_probs = mctsr.search();
                torch::Tensor indices = torch::multinomial(action_probs, 1, false);
                action = indices[0].item<int>();
                mctsr.update_root(action);
                if(step > 0){
                    mcts.update_root(action);
                }
                
            }else{
                action_probs = mcts.search(false);
                torch::Tensor temp_action_probs = torch::pow(action_probs, 1.0/alphazero_params.temperature);
                temp_action_probs /= torch::sum(temp_action_probs);
                torch::Tensor indices = torch::multinomial(temp_action_probs, 1, false);
                action = indices[0].item<int>();
                mcts.update_root(action);
                mctsr.update_root(action);
            }
            state = game.get_next_state(state, action, current_player);
            neutral_state = game.change_perspective(state, -1);

            auto terminated = game.get_value_and_terminated(state, action);
            float value = terminated.first;
            bool is_terminal = terminated.second;
            std::cout << neutral_state << std::endl;
            if (is_terminal){
                std::cout << neutral_state << std::endl;
                if (value == 1.0 and current_player == 1){
                    std::cout << "gana X" << std::endl;
                }else if(value == 1.0 and current_player == -1){
                    std::cout << "gana O" << std::endl;
                }else{
                    std::cout << "empate" << std::endl;
                }
                mcts.clean_tree();
                mctsr.clean_tree();
                break;
            }
            current_player = game.get_opponent(current_player);
            step++;
        }
        buffer_eval_manager1->terminate_thread_buffer_eval = true;
        if (controller_thread1.joinable()) {
            controller_thread1.join();
        }
    }

    return 0;
}