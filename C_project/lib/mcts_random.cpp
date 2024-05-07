#include <iostream>
#include <utility>
#include "mcts_random.h"
#include <memory>
#include <cmath>
#include <chrono>
#include <fstream>


void NodeR::backpropagate(const float& value){
    value_sum += value;
    visit_count += 1; 

    float opponent_value = game.get_opponent_value(value);

    if (parent != nullptr){
        parent->backpropagate(opponent_value);
    }
}

void NodeR::expand(const torch::Tensor& valid_moves){
    for (int action = 0; action < valid_moves.size(0); ++action) {
        int valid = valid_moves[action].item<float>();

        if (valid > 0.0){            
            auto child_state = game.get_next_state(state, action, 1);
            child_state = game.change_perspective(child_state, -1);
            int opp_player = game.get_opponent(current_player);

            NodeR* child = new NodeR(game, child_state, this, action, opp_player);
            //this->children.push_back(child);
            this->children[action] = child;
        }    
    }
    is_expanded = true;
}


NodeR* NodeR::select(const torch::Tensor& valid_moves){
    torch::Tensor indices = torch::multinomial(valid_moves, 1, false);
    int action = indices[0].item<int>();
    auto random_child = children[action];
    return random_child;
}


void MCTSRand::clean_tree(){
    if (initial_state_node != nullptr){
        delete initial_state_node;
        initial_state_node = nullptr;
    }
}

void MCTSRand::reset_game(torch::Tensor state, int current_player){
    initial_state_node = new NodeR(game, state, 1, current_player);
    root = initial_state_node;
}

void MCTSRand::update_root(int action){
    root = root->children[action];
}

torch::Tensor MCTSRand::search(){
    torch::Tensor obs_state;
    torch::Tensor policy;
    torch::Tensor valid_moves;
    float value;
    std::pair<bool, float> terminated;

    valid_moves = game.get_valid_moves(root->state).to(torch::kFloat32);

    if (!root->is_expanded){
        root->expand(valid_moves);
    }
    for (int i = 0; i < num_searches; ++i){
        NodeR* node = root;
        bool is_terminal = false;
        auto valid_moves_s = valid_moves.clone();
        //std::cout << "Num search: " << i << std::endl;
        while(!is_terminal){
            //std::cout << "Selecting" << std::endl;
            node = node->select(valid_moves_s);
            valid_moves_s = game.get_valid_moves(node->state).to(torch::kFloat32);
            terminated = game.get_value_and_terminated(node->state, node->action_taken);

            value = terminated.first;
            is_terminal = terminated.second;

            if(!is_terminal && !node->is_expanded){
                //std::cout << "Expanding" << std::endl;
                node->expand(valid_moves_s);
            }
        }
        //std::cout << "Backpropagating value: " << value << std::endl;
        node->backpropagate(value);
    }

    torch::Tensor action_probs = torch::zeros({game.action_size});
    torch::Tensor action_probs_final = torch::zeros({game.action_size});

    for (const auto& par: root->children){
        auto child = par.second;
        auto q_value = static_cast<float>(child->value_sum)/child->visit_count;
        action_probs.index_put_({child->action_taken}, q_value);
    }

    std::cout << "Probs: " << action_probs << std::endl;
    auto best_action = torch::argmax(action_probs).item<int>();
    std::cout << "action_probs size: " << action_probs.sizes() << std::endl;
    action_probs_final.index_put_({best_action}, 1);
    std::cout << "Probs: " << action_probs_final << std::endl;

    return action_probs_final; 
}