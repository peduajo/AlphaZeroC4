#include <iostream>
#include <torch/torch.h>
#include <torch/script.h>
#include <utility>
#include "mcts.h"
#include <memory>
#include <cmath>
#include <chrono>
#include <cuda_runtime.h>
#include <boost/archive/binary_oarchive.hpp>
#include <future>
#include "node.h"
#include <fstream>

constexpr auto max_wait_time_inference = std::chrono::milliseconds(25); // Ejemplo: 100 ms

std::string tensorToString(const torch::Tensor& tensor) {
    std::stringstream ss;
    ss << tensor;
    return ss.str();
}

void MCTS::clean_tree(){
    //std::cout << initial_state_node->state << std::endl;
    if (initial_state_node != nullptr){
        delete initial_state_node;
        //std::cout << initial_state_node->contador_eliminaciones << std::endl;
        initial_state_node = nullptr;
    }
}

void MCTS::reset_game(torch::Tensor state, bool debug, int current_player){
    initial_state_node = new Node(game, args, state, 1, debug, current_player);
    created_nodes = 1;
    root = initial_state_node;
}

void MCTS::update_root(int action){
    root = root->children[action];
    //std::cout << root->state << std::endl;
}

torch::Tensor MCTS::search(bool debug){
    torch::Tensor obs_state;
    torch::Tensor policy;
    torch::Tensor valid_moves;
    float value;
    bool is_terminal;
    std::pair<bool, float> terminated;

    obs_state = game.get_encoded_state(root->state.unsqueeze(0), root->current_player);

    std::promise< std::pair<torch::Tensor, float> > promise;
    auto future = promise.get_future();

    {
        std::lock_guard<std::mutex> guard(buffer_eval_manager->buffer_eval_mutex);
        buffer_eval_manager->buffer_eval_inputs.push_back(std::move(obs_state));
        buffer_eval_manager->buffer_eval_promises.push_back(std::move(promise));
        if (buffer_eval_manager->buffer_eval_inputs.size() >= buffer_eval_manager->batch_size) {
            buffer_eval_manager->buffer_eval_cond.notify_one();
        }
    }

    auto prediction = future.get();
    policy = prediction.first;

    auto alpha_dirichlet_tensor = torch::ones(game.action_size, torch::dtype(torch::kFloat32)) * args.dirichlet_alpha;

    auto dirichlet_noise = torch::_sample_dirichlet(alpha_dirichlet_tensor);

    auto base_policy = (1 - args.dirichlet_epsilon) * policy;
    auto noise_policy = args.dirichlet_epsilon * dirichlet_noise;

    policy = base_policy + noise_policy;

    valid_moves = game.get_valid_moves(root->state);

    policy *= valid_moves;
    policy /= torch::sum(policy);

    if (!root->is_expanded){
        root->expand(policy);
        created_nodes += root->children.size();
    }else{
        for (int action = 0; action < policy.size(0); ++action) {
            float prob = policy[action].item<float>();
            if (prob > 0.0){
                root->children[action]->prior = prob;
            }
        }
    }

    std::vector<std::future<void>> futures;

    for (int i = 0; i < args.num_searches; ++i) {
        futures.emplace_back(
            pool.enqueue([this] { 
                this->make_thread_simulations();
            })
        );
    }

    for (auto & future : futures) {
        future.get();
    }

    torch::Tensor action_probs = torch::zeros({game.action_size});
    for (const auto& par: root->children){
        auto child = par.second;
        action_probs.index_put_({child->action_taken}, child->visit_count);
    }

    action_probs /= torch::sum(action_probs);

    return action_probs; 
}

std::tuple<Node*, float, bool> MCTS::select_and_check_terminal(Node* node){
    while (node->is_expanded){
        node = node->select();
    }

    auto terminated = game.get_value_and_terminated(node->state, node->action_taken);

    float value = terminated.first;
    bool is_terminal = terminated.second;

    value = game.get_opponent_value(value);

    return {node, value, is_terminal};
}


void MCTS::make_thread_simulations(){
    torch::Tensor obs_state;
    torch::Tensor policy;
    torch::Tensor valid_moves;

    Node* node = root;

    auto selection_data = this->select_and_check_terminal(node);
    node = std::get<0>(selection_data);
    auto value = std::get<1>(selection_data);
    auto is_terminal = std::get<2>(selection_data);

    bool thread_expanded = false;

    while (!is_terminal && !thread_expanded){
    //if (!is_terminal){
        obs_state = game.get_encoded_state(node->state.unsqueeze(0), node->current_player);

        std::promise< std::pair<torch::Tensor, float> > promise;
        auto future = promise.get_future();

        {
            std::lock_guard<std::mutex> guard(buffer_eval_manager->buffer_eval_mutex);
            buffer_eval_manager->buffer_eval_inputs.push_back(std::move(obs_state));
            buffer_eval_manager->buffer_eval_promises.push_back(std::move(promise));
            if (buffer_eval_manager->buffer_eval_inputs.size() >= buffer_eval_manager->batch_size) {
                buffer_eval_manager->buffer_eval_cond.notify_one();
            }
        }

        auto prediction = future.get();
        policy = prediction.first;
        value = prediction.second;

        valid_moves = game.get_valid_moves(node->state);

        policy *= valid_moves;
        policy /= torch::sum(policy);

        thread_expanded = node->expand(policy);
        if (!thread_expanded){
            auto selection_data = this->select_and_check_terminal(node);
            node = std::get<0>(selection_data);
            value = std::get<1>(selection_data);
            is_terminal = std::get<2>(selection_data);
        }
    }

    node->backpropagate(value);
}


void BufferEvalManager::set_model(std::string model_path){
    std::cout << "Cargando modelo del path: " << model_path << std::endl;
    model = torch::jit::load(model_path);
    std::cout << "Modelo cargado!" << std::endl;
    model.to(device);
    model.eval();

    torch::Tensor state_raw = torch::tensor({{ 0, 0, 0, 0, 0, 0, 0},
                                             { 0, 0, 0, 0, 0, 0, 0},
                                             { 0, 0, 0, 0, 0, 0, 0},
                                             { 0, 0, 0, 1, 0, 0, 0},
                                             { 0, 0, 0, 1, 0, 0, 0},
                                             { 0,-1,-1, 1,-1, 0, 0}});

    torch::Tensor obs = game.get_encoded_state(state_raw, 1).to(device).unsqueeze(0);

    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(obs);
    std::cout << inputs << std::endl;
    auto inference = model.forward(inputs).toTuple();
    auto inference_elements = inference->elements();

    torch::Tensor action_probs = inference_elements[0].toTensor();
    action_probs = torch::softmax(action_probs, 1).to(torch::kCPU);

    std::cout << action_probs << std::endl;
}

std::pair<torch::Tensor, torch::Tensor> BufferEvalManager::batch_inference(std::vector<torch::jit::IValue> inputs){
    torch::NoGradGuard no_grad;

    auto inference = model.forward(inputs).toTuple();
    auto inference_elements = inference->elements();
    auto new_policies = inference_elements[0].toTensor();
    new_policies = torch::softmax(new_policies, 1).to(torch::kCPU);
    auto new_values = inference_elements[1].toTensor().to(torch::kCPU);

    return std::make_pair(new_policies, new_values);
}

void BufferEvalManager::buffer_eval_controller(){
    std::unique_lock<std::mutex> lock(buffer_eval_mutex);
    std::vector<torch::jit::IValue> inputs;
    while(true){
        while (!terminate_thread_buffer_eval && buffer_eval_inputs.size() < batch_size) {
            if (buffer_eval_cond.wait_for(lock, max_wait_time_inference, [this]{ return buffer_eval_inputs.size() >= batch_size || terminate_thread_buffer_eval; })) {
                break;
            } else {
                if (!buffer_eval_inputs.empty()) {
                    break;
                }
            }
        }
        if (terminate_thread_buffer_eval && buffer_eval_inputs.empty()) {
            break;
        }
        auto batch_encoded_states = torch::concat(buffer_eval_inputs, 0).to(device);
        inputs.push_back(batch_encoded_states);

        auto predictions = this->batch_inference(inputs);
        auto policies = predictions.first;
        auto values = predictions.second;
        for (int i = 0; i < buffer_eval_inputs.size(); ++i) {
            auto policy = policies.index({i, "..."});
            auto value = values[i].item<float>();
            auto pair_pred = std::make_pair(policy, value);
            buffer_eval_promises[i].set_value(pair_pred);
        }

        buffer_eval_inputs.clear();
        buffer_eval_promises.clear();
        inputs.clear();
    }
}