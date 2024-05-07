#pragma once

#include <iostream>
#include <torch/torch.h>
#include "game.h"
#include "params.h"
#include <utility>
#include <vector>

class NodeR {
public: 
    Game game;
    torch::Tensor state;
    NodeR* parent;
    int action_taken;
    float prior;
    int visit_count;
    int virtual_loss;
    float value_sum;
    bool is_expanded;
    int current_player;
    std::unordered_map<int, NodeR*> children;
public:
    NodeR(const Game& game, const torch::Tensor& state, int visit_count, int current_player)
        : game(game),
          state(state),
          parent(nullptr),
          action_taken(-1),
          visit_count(visit_count),
          value_sum(0),
          current_player(current_player),
          is_expanded(false){}
    
    NodeR(const Game& game, const torch::Tensor& state, NodeR* parent, int  action_taken, int current_player)
        : game(game),
          state(state),
          parent(parent),
          action_taken(action_taken),
          visit_count(0),
          value_sum(0),
          current_player(current_player),
          is_expanded(false){}
    
    ~NodeR(){
        for (const auto& par: children){
            auto child = par.second;
            delete child;
            child = nullptr;
        }
    }

    //inline bool is_fully_expanded();
    NodeR* select(const torch::Tensor& valid_moves);
    void expand(const torch::Tensor& valid_moves);
    void backpropagate(const float& value);
};

class MCTSRand {
public:
    Game game;
    int num_searches;
    NodeR* root;
    NodeR* initial_state_node;
public:
    MCTSRand(const Game& game, int num_searches) 
        : game(game), num_searches(num_searches){
    }

    torch::Tensor search();
    void reset_game(torch::Tensor state, int current_player);
    void update_root(int action);
    void clean_tree();
};
