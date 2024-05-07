#include <iostream>
#include <torch/torch.h>
#include <utility>
#include "game.h"

torch::Tensor Game::get_initial_state(){ //OK
    return torch::zeros({row_count, column_count});
}

torch::Tensor Game::get_next_state(const torch::Tensor& state, int action, int player){ //OK
    torch::Tensor new_state = state.clone();

    int row = torch::nonzero(new_state.index({"...", action}) == 0).max().item<int>();
    new_state.index_put_({row, action}, player);

    return new_state;
}

torch::Tensor Game::get_valid_moves(const torch::Tensor& state){ //OK
    //tomamos la primera fila y se comparan los ceros
    torch::Tensor valid_moves = (state.index({0,"..."}) == 0).to(torch::kUInt8);
    return valid_moves;
}

std::pair<int, bool> Game::get_value_and_terminated(const torch::Tensor& state, int action){ //OK
    if (_check_win(state, action)){
        return std::pair<int, bool>(1, true);
    }
    if (torch::sum(get_valid_moves(state)).item<int>() == 0){
        return std::pair<int, bool>(0, true);
    }
    return std::pair<int, bool>(0, false);
}

int Game::_count(int offset_row, int offset_column, int row, int action, int player, const torch::Tensor& state){
    for(int i = 1; i < in_a_row; ++i){
        int r = row + offset_row * i;
        int c = action + offset_column * i;
        if (r < 0 || r >= row_count || c < 0 || c >= column_count){
            return i - 1;
        }else{
            int check_player = state.index({r, c}).item<int>();
            if (check_player != player){
                return i - 1;
            }
        }
    }
    return in_a_row - 1;
}

bool Game::_check_win(const torch::Tensor& state, int action){
    //en este caso el valor -1 significa que no está inicializada la acción ya que C++ no tiene None
    if (action == -1){
        return false;
    }

    int row = torch::nonzero(state.index({"...", action})).min().item<int>();
    int column = action;
    int player = state.index({row, column}).item<int>();

    bool cond1 = this->_count(1, 0, row, action, player, state) >= in_a_row - 1;
    bool cond2 = (this->_count(0, 1, row, action, player, state) + this->_count(0, -1, row, action, player, state)) >= in_a_row - 1;
    bool cond3 = (this->_count(1, 1, row, action, player, state) + this->_count(-1, -1, row, action, player, state)) >= in_a_row - 1;
    bool cond4 = (this->_count(1, -1, row, action, player, state) + this->_count(-1, 1, row, action, player, state)) >= in_a_row - 1;
    
    return (cond1 || cond2 || cond3 || cond4);
}

int Game::get_opponent(int player){ //OK
    return -player;
}

float Game::get_opponent_value(float value){ //OK
    return -value;
}

torch::Tensor Game::change_perspective(const torch::Tensor& state, int player){ //OK
    return state * player;
}

torch::Tensor Game::get_encoded_state(const torch::Tensor& state, int current_player){ //Cambiando a forma para transformer
    
    torch::Tensor mask_opp_pieces = state == -1;
    torch::Tensor mask_out_pieces = state == 1;
    torch::Tensor mask_player = torch::zeros(mask_out_pieces.sizes());

    if (current_player == 1){
        mask_player += 1;
    }

    torch::Tensor encoded_state = torch::stack({mask_opp_pieces, mask_out_pieces, mask_player}).to(torch::kFloat32);

    if(encoded_state.dim() == 4){
        encoded_state = encoded_state.permute({1, 0, 2, 3});
    }

    return encoded_state;
}