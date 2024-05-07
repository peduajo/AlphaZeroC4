#include "alphazero.h"
#include <torch/torch.h>
#include <torch/script.h>
#include <vector>
#include <tuple>
#include <algorithm>
#include <random>
#include <future>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include "cuda.h"
#include <cmath>
#include <sys/stat.h>
#include <filesystem>

#include <iostream>
#include <memory>
#include <unistd.h>
#include <sys/wait.h> 

namespace fs = std::filesystem;


std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> AlphaZero::_self_play(bool debug){
    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int>> memory;
    MCTS mcts(game, args, device, buffer_eval_manager);
    int player = 1;
    torch::Tensor state = game.get_initial_state();
    mcts.reset_game(state, false, 1);

    int hist_outcome;
    torch::Tensor hist_neutral_state;
    torch::Tensor hist_action_probs;
    torch::Tensor hist_q_values;
    int hist_player;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, int> tuple_games;
    std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor> tuple;
    float temperature = args.temperature;

    float q_value;

    while (true){
        torch::Tensor neutral_state = mcts.root->state;
        torch::Tensor action_probs = mcts.search(debug);
        q_value = static_cast<float>(mcts.root->value_sum)/mcts.root->visit_count;
        auto root_encoded_state = game.get_encoded_state(mcts.root->state.unsqueeze(0), player);
        tuple_games = std::make_tuple(root_encoded_state, action_probs, torch::tensor({q_value}), player);
        memory.push_back(tuple_games);

        if (memory.size() > 11){
            temperature = 0.05;
        }

        torch::Tensor temp_action_probs = torch::pow(action_probs, 1.0/temperature);
        temp_action_probs /= torch::sum(temp_action_probs);
            
        torch::Tensor indices = torch::multinomial(temp_action_probs, 1, false);
        int action = indices[0].item<int>();
        mcts.update_root(action);

        state = game.get_next_state(state, action, player);

        auto terminated = game.get_value_and_terminated(state, action);
        float value = terminated.first;
        bool is_terminal = terminated.second;

        if (debug){
            std::cout << neutral_state << std::endl;
            std::cout << "Tensor de probabilidades" << action_probs << std::endl;
            std::cout << "Tensor de probabilidades temperature" << temp_action_probs << std::endl;
            std::cout << "Acción tomada: " << action << std::endl;
            std::cout << "Terminal: " << std::to_string(is_terminal) << std::endl;
            std::cout << "----------------------------------" << std::endl;
        }

        if (is_terminal){
            std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> return_memory;

            for (const auto& m : memory) {
                std::tie(hist_neutral_state, hist_action_probs, hist_q_values, hist_player) = m;
                if (hist_player == player){
                    hist_outcome = value;
                }else{
                    hist_outcome = game.get_opponent_value(value);
                }
                auto hist_outcome_tensor = torch::tensor({hist_outcome});
                tuple = std::make_tuple(hist_neutral_state, hist_action_probs, hist_q_values, hist_outcome_tensor);
                return_memory.push_back(tuple);
            }
            mcts.clean_tree();
            //std::cout << mcts.created_nodes << std::endl;
            return return_memory;
        }

        player = game.get_opponent(player);
    }
}


void AlphaZero::_save_memory(std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> memory){

    auto dir_path = base_path_data + std::to_string(iteration_idx);
    auto dir_path_states = dir_path + "/states/";
    auto dir_path_policy = dir_path + "/policy/";
    auto dir_path_q_values = dir_path + "/q_values/";
    auto dir_path_values = dir_path + "/values/";

    std::vector<torch::Tensor> state_vec;
    std::vector<torch::Tensor> policy_vec;
    std::vector<torch::Tensor> q_val_vec;
    std::vector<torch::Tensor> value_vec;

    torch::Tensor st;
    torch::Tensor p;
    torch::Tensor q;
    torch::Tensor v;

    for (const auto& m : memory) {
        std::tie(st, p, q, v) = m;
        state_vec.push_back(st);
        policy_vec.push_back(p);
        q_val_vec.push_back(q);
        value_vec.push_back(v);
    }

    auto dir_path_states_file = dir_path_states + std::to_string(process_idx) + ".zip";
    auto states_tensor = torch::stack(state_vec, 0);
    auto bytes_s = torch::jit::pickle_save(states_tensor);
    std::ofstream fout_s(dir_path_states_file, std::ios::out | std::ios::binary);
    fout_s.write(bytes_s.data(), bytes_s.size());
    fout_s.close();


    auto dir_path_policy_file = dir_path_policy + std::to_string(process_idx) + ".zip";
    auto policy_tensor = torch::stack(policy_vec, 0);
    auto bytes_p = torch::jit::pickle_save(policy_tensor);
    std::ofstream fout_p(dir_path_policy_file, std::ios::out | std::ios::binary);
    fout_p.write(bytes_p.data(), bytes_p.size());
    fout_p.close();

    auto dir_path_q_value_file = dir_path_q_values + std::to_string(process_idx) + ".zip";
    auto q_value_tensor = torch::stack(q_val_vec, 0);
    auto bytes_v = torch::jit::pickle_save(q_value_tensor);
    std::ofstream fout_v(dir_path_q_value_file, std::ios::out | std::ios::binary);
    fout_v.write(bytes_v.data(), bytes_v.size());
    fout_v.close();

    auto dir_path_value_file = dir_path_values + std::to_string(process_idx) + ".zip";
    auto value_tensor = torch::stack(value_vec, 0);
    auto bytes_q = torch::jit::pickle_save(value_tensor);
    std::ofstream fout_q(dir_path_value_file, std::ios::out | std::ios::binary);
    fout_q.write(bytes_q.data(), bytes_q.size());
    fout_q.close();
}

void AlphaZero::_generate_self_play_data(std::string model_path){
    auto init_timestamp = std::chrono::high_resolution_clock::now();

    try {
        buffer_eval_manager->set_model(model_path);
    }catch (const c10::Error& e) {
        std::cerr << "Error loading model: " << e.what() << std::endl;
        std::exit(EXIT_FAILURE);
    }

    std::thread controller_thread([this] { this->buffer_eval_manager->buffer_eval_controller(); });

    std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>> memory;

    std::vector<std::future<std::vector<std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor>>>> futures;
    for (int i = 0; i < args.n_selfplay_iterations; ++i) {
        futures.emplace_back(
            pool_games.enqueue([this] { 
                return this->_self_play(false);
            })
        );         
    }

    for (auto& f : futures) {
        auto match_memory = f.get();
        memory.insert(memory.end(), match_memory.begin(), match_memory.end());
    }

    auto end_timestamp = std::chrono::high_resolution_clock::now();
    auto duration_selfplay = std::chrono::duration_cast<std::chrono::microseconds>(end_timestamp - init_timestamp);

    // Convierte la duración a segundos
    double secs = static_cast<double>(duration_selfplay.count()) / 1'000'000.0;
    double mins = secs / 60.0;

    float steps_avg = memory.size()/args.n_selfplay_iterations;

    std::cout << "Timestamp: " << mins << " | Games played: " << args.n_selfplay_iterations << "| Mean steps per game: " << steps_avg << std::endl;

    this->_save_memory(memory);

    buffer_eval_manager->terminate_thread_buffer_eval = true;
    if (controller_thread.joinable()) {
        controller_thread.join(); 
    }
}


void AlphaZero::pipeline_self_play(){

    std::string model_path = "../models/model_" + std::to_string(iteration_idx) + ".ts";

    auto dir_path = base_path_data + std::to_string(iteration_idx);

    if (process_idx == 0){
        struct stat info;
        if (stat(dir_path.c_str(), &info) != -1) {
            try {
                fs::remove_all(dir_path);
            } catch (const fs::filesystem_error& e) {
                throw std::runtime_error("Error deleting directory");
            }
        }

        auto dir_path_states = dir_path + "/states";
        auto dir_path_policy = dir_path + "/policy";
        auto dir_path_q_values = dir_path + "/q_values";
        auto dir_path_values = dir_path + "/values";
        auto dir_path_clean = dir_path + "/clean";

        const char* cDirPath = dir_path.c_str();
        const char* cDirPathStates = dir_path_states.c_str();
        const char* cDirPathPolicy = dir_path_policy.c_str();
        const char* cDirPathValues = dir_path_values.c_str();
        const char* cDirPathQValues = dir_path_q_values.c_str();
        const char* cDirPathClean = dir_path_clean.c_str();

        if (mkdir(cDirPath, 0755) == -1) {
            throw std::runtime_error("Error al crear el directorio padre");
        }
        if (mkdir(cDirPathStates, 0755) == -1) {
            throw std::runtime_error("Error al crear el directorio padre");
        }
        if (mkdir(cDirPathPolicy, 0755) == -1) {
            throw std::runtime_error("Error al crear el directorio padre");
        }
        if (mkdir(cDirPathQValues, 0755) == -1) {
            throw std::runtime_error("Error al crear el directorio padre");
        }
        if (mkdir(cDirPathValues, 0755) == -1) {
            throw std::runtime_error("Error al crear el directorio padre");
        }
        if (mkdir(cDirPathClean, 0755) == -1) {
            throw std::runtime_error("Error al crear el directorio padre");
        }
    }
    
    this->_generate_self_play_data(model_path);
}
