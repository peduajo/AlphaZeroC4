#pragma once

#include <iostream>
#include <torch/torch.h>
#include <utility>

class Game {
public:
    int row_count;
    int column_count;
    int action_size;
    int n_cells;
    int in_a_row;
    std::unordered_map<int, std::vector<int>> token_dict;

public:
    Game() {
        row_count = 6;
        column_count = 7;
        action_size = column_count;
        n_cells = row_count * column_count;
        in_a_row = 4;
        for (int i = 0; i < 42; ++i) {
            token_dict[0].push_back(i);
            token_dict[1].push_back(42 + i);
            token_dict[2].push_back(84 + i);
        }
    }
    torch::Tensor get_initial_state();
    torch::Tensor get_next_state(const torch::Tensor& state, int action, int player);
    torch::Tensor get_valid_moves(const torch::Tensor& state);
    std::pair<int, bool> get_value_and_terminated(const torch::Tensor& state, int action);
    int get_opponent(int player);
    float get_opponent_value(float value);
    torch::Tensor change_perspective(const torch::Tensor& state, int player);
    torch::Tensor get_encoded_state(const torch::Tensor& state, int current_player);

private:
    bool _check_win(const torch::Tensor& state, int action);
    int _count(int offset_row, int offset_column, int row, int action, int player, const torch::Tensor& state);
};