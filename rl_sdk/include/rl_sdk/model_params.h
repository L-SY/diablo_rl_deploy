//
// Created by lsy on 25-4-23.
//

// model_params.hpp
#pragma once
#include <string>
#include <vector>
#include <torch/torch.h>

class ModelParams {
public:
  virtual ~ModelParams() = default;

  std::string model_name;
  std::string framework;
  bool use_history;
  double dt;
  int decimation;
  int num_observations;
  std::vector<std::string> observations;
  std::vector<int> observations_history;
  double action_scale_pos;
  double action_scale_vel;
  int num_of_dofs;
  double lin_vel_scale;
  double ang_vel_scale;
  double dof_pos_scale;
  double dof_vel_scale;
  double clip_obs;

  torch::Tensor clip_actions_upper;
  torch::Tensor clip_actions_lower;
  torch::Tensor torque_limits;
  torch::Tensor commands_scale;
  torch::Tensor default_dof_pos;

protected:
  void init_tensor_view(torch::Tensor& tensor, std::vector<double> values) {
    tensor = torch::tensor(values).view({1, -1});
  }
};
