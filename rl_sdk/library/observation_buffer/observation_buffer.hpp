#pragma once

#include <torch/torch.h>
#include <vector>

// ref: https://github.com/fan-ziqi/rl_sar
class ObservationBuffer {
public:
    ObservationBuffer(int num_envs, int num_obs, int include_history_steps);
    ObservationBuffer();

    void reset(std::vector<int> reset_idxs, torch::Tensor new_obs);
    void insert(torch::Tensor new_obs);
    torch::Tensor get_obs_vec(std::vector<int> obs_ids);

private:
    int num_envs;
    int num_obs;
    int include_history_steps;
    int num_obs_total;
    torch::Tensor obs_buf;
};
