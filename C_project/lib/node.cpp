#include "mcts.h"
#include <memory>
#include <cmath>
#include <iostream>
#include <torch/torch.h>
#include "node.h"

int Node::contador_eliminaciones = 0;

bool Node::expand(const torch::Tensor& policy){
    std::lock_guard<std::mutex> lock(node_mutex);
    if (!is_expanded){
        for (int action = 0; action < policy.size(0); ++action) {
            float prob = policy[action].item<float>();

            if (prob > 0.0){            
                auto child_state = game.get_next_state(state, action, 1);
                child_state = game.change_perspective(child_state, -1);
                int opp_player = game.get_opponent(current_player);

                Node* child = new Node(game, args, child_state, this, action, prob, opp_player);
                this->children[action] = child;
            }    
        }
        is_expanded = true;
        expanded_times += 1;
        return true;
    }else{
        return false;
    }
}

Node* Node::select(){
    Node* best_child = nullptr;
    float best_ucb = -99999.0;

    //for (auto& child : children) {
    for (const auto& par: children){
        auto child = par.second;
        float ucb = this->get_ucb(child);
        if (ucb > best_ucb){
            best_child = child;
            best_ucb = ucb;
        }
    }
    best_child->virtual_loss += 1;
    return best_child;
}

float Node::get_ucb(Node* child){
    float q_value = 0.0;
    if (child->visit_count > 0){
        float value_sum_adjusted = child->value_sum + child->virtual_loss;
        float q_value_prob = ((value_sum_adjusted / child->visit_count) + 1) / 2;
        q_value = 1 - q_value_prob;
    }
    float ucb = q_value + args.c_puct * (std::sqrt(visit_count) / (child->visit_count + 1)) * child->prior;
    return ucb;
}

void Node::backpropagate(const float& value){
    value_sum += value;
    visit_count += 1; 
    virtual_loss -= 1;

    float opponent_value = game.get_opponent_value(value);

    if (parent != nullptr){
        parent->backpropagate(opponent_value);
    }
}