#pragma once

#include <iostream>
#include <torch/torch.h>
#include "game.h"
#include "params.h"
#include <utility>
#include <vector>
#include "mcts.h"

class MCTS;

class Node {
public: 
    Game game;
    AlphazeroParams args;
    torch::Tensor state;
    Node* parent;
    int action_taken;
    float prior;
    int visit_count;
    int virtual_loss;
    float value_sum;
    bool is_expanded;
    int current_player;
    int expanded_times = 0;
    std::unordered_map<int, Node*> children;
    bool debug_node;
    std::mutex node_mutex;
    static int contador_eliminaciones;
public:
    Node(const Game& game, const AlphazeroParams& args, const torch::Tensor& state, int visit_count, bool debug_node, int current_player)
        : game(game),
          args(args),
          state(state),
          parent(nullptr),
          action_taken(-1),
          prior(0.0),
          visit_count(visit_count),
          value_sum(0),
          current_player(current_player),
          virtual_loss(0),
          is_expanded(false),
          debug_node(debug_node){}
    
    Node(const Game& game, const AlphazeroParams& args, const torch::Tensor& state, Node* parent, int  action_taken, float prior, int current_player)
        : game(game),
          args(args),
          state(state),
          parent(parent),
          action_taken(action_taken),
          prior(prior),
          visit_count(0),
          value_sum(0),
          current_player(current_player),
          virtual_loss(0),
          is_expanded(false),
          debug_node(false){}
    
    ~Node(){
        for (const auto& par: children){
            auto child = par.second;
            delete child;
            child = nullptr;
            ++contador_eliminaciones;
        }
    }

    Node* select();
    float get_ucb(Node* child);
    bool expand(const torch::Tensor& policy);
    void backpropagate(const float& value);
};