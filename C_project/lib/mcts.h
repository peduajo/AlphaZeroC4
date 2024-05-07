#pragma once

#include <iostream>
#include <torch/torch.h>
#include "game.h"
#include "params.h"
#include <utility>
#include <vector>
#include "node.h"
#include "ThreadPool.h"

class Node;



class SafeMapTransitionsCache {
public:
    void update(const std::string& key, torch::Tensor& value) {
        std::lock_guard<std::mutex> guard(map_mutex);
        transitions_cache[key] = value;
        keys_list_order.push_front(key);
        list_size += 1;
        if (list_size>max_size){
            transitions_cache.erase(keys_list_order.back());
            keys_list_order.pop_back();
            list_size -= 1;
        }
    }

    std::pair<torch::Tensor, bool> get(const std::string& key) {
        std::lock_guard<std::mutex> guard(map_mutex);
        auto it = transitions_cache.find(key);

        torch::Tensor value;
        if (it != transitions_cache.end()) {
            value = it->second;
            return std::make_pair(value, true);
        }
        return std::make_pair(value, false);
    }

    int get_size(){
        return transitions_cache.size();
    }

    void clear(){
        transitions_cache.clear();
    }

private:
    std::list<std::string> keys_list_order;
    std::unordered_map<std::string, torch::Tensor> transitions_cache;
    std::mutex map_mutex;
    int max_size = 15'000'000;
    int list_size = 0;
};

class SafeMapInferenceCache {
public:
    void update(const std::string& key, std::pair<torch::Tensor, float>& value) {
        std::lock_guard<std::mutex> guard(map_mutex);
        inference_cache[key] = value;

        keys_list_order.push_front(key);
        list_size += 1;

        if (list_size>max_size){
            inference_cache.erase(keys_list_order.back());
            keys_list_order.pop_back();
            list_size -= 1;
        }
    }

    std::pair<std::pair<torch::Tensor, float>, bool> get(const std::string& key) {
        std::lock_guard<std::mutex> guard(map_mutex);
        auto it = inference_cache.find(key);

        std::pair<torch::Tensor, float> value;
        if (it != inference_cache.end()) {
            value = it->second;
            return std::make_pair(value, true);
        }
        return std::make_pair(value, false);
    }

    int get_size(){
        return inference_cache.size();
    }

    void clear(){
        inference_cache.clear();
    }

private:
    std::list<std::string> keys_list_order;
    std::unordered_map<std::string, std::pair<torch::Tensor, float>> inference_cache;
    std::mutex map_mutex;
    int max_size = 15'000'000;
    int list_size = 0;
};

class SafeMapTerminatedCache {
public:
    void update(const std::string& key, const std::pair<bool, float>& value) {
        std::lock_guard<std::mutex> guard(map_mutex);
        terminated_cache[key] = value;

        keys_list_order.push_front(key);
        list_size += 1;
        if (list_size>max_size){
            terminated_cache.erase(keys_list_order.back());
            keys_list_order.pop_back();
            list_size -= 1;
        }
    }

    std::pair<std::pair<bool, float>, bool> get(const std::string& key) {
        std::lock_guard<std::mutex> guard(map_mutex);
        auto it = terminated_cache.find(key);

        std::pair<bool, float> value;
        if (it != terminated_cache.end()) {
            value = it->second;
            return std::make_pair(value, true);
        }
        return std::make_pair(value, false);
    }

    void clear(){
        terminated_cache.clear();
    }

    int get_size(){
        return terminated_cache.size();
    }

private:
    std::unordered_map<std::string, std::pair<bool, float>> terminated_cache;
    std::mutex map_mutex;
    int max_size = 20'000'000;
    std::list<std::string> keys_list_order;
    int list_size = 0;
};


class BufferEvalManager {
public:
    Game game;
    torch::jit::script::Module model;
    torch::Device device;
    std::mutex buffer_eval_mutex;
    std::condition_variable buffer_eval_cond;
    std::vector<torch::Tensor> buffer_eval_inputs;
    std::vector< std::promise< std::pair<torch::Tensor, float> > > buffer_eval_promises;
    bool terminate_thread_buffer_eval;
    int batch_size;
    torch::ScalarType dtype;
public:
    BufferEvalManager(const Game& game,const torch::Device& device, int batch_size) 
        : game(game), device(device), batch_size(batch_size), terminate_thread_buffer_eval(false){
    }

    void buffer_eval_controller();
    std::pair<torch::Tensor, torch::Tensor> batch_inference(std::vector<torch::jit::IValue> inputs);
    void set_model(std::string model_path);
};


class MCTS {
public:
    Game game;
    torch::Device device;
    AlphazeroParams args;
    Node* root;
    Node* initial_state_node;
    int num_searches_per_thread;
    int created_nodes;
    SafeMapTerminatedCache terminated_cache_manager;
    SafeMapTransitionsCache transitions_cache_manager;
    SafeMapInferenceCache inference_cache_manager;
    ThreadPool pool;
    std::shared_ptr<BufferEvalManager> buffer_eval_manager;
public:
    MCTS(const Game& game, const AlphazeroParams& args,const torch::Device& device, std::shared_ptr<BufferEvalManager> buffer_eval_manager) 
        : game(game), args(args), initial_state_node(nullptr), pool(args.num_threads_mcts), device(device), buffer_eval_manager(buffer_eval_manager){
        num_searches_per_thread = args.num_searches / args.num_threads_mcts;
        created_nodes = 0;
    }

    torch::Tensor search(bool debug);
    void reset_game(torch::Tensor state, bool degug, int current_player);
    void update_root(int action);
    void clean_tree();
    void make_thread_simulations();
    std::tuple<Node*, float, bool> select_and_check_terminal(Node* node);
};
