//
// Created by lsy on 25-4-23.
//

#pragma once
#include "rl_sdk/model_params.h"

class DiabloParams : public ModelParams {
public:
  DiabloParams() {
    framework = "isaacgym";
    use_history = true;
    dt = 0.005;
    decimation = 2;
    num_observations = 25;
    observations = { "ang_vel", "gravity_vec", "commands", "dof_pos", "dof_vel", "actions" };
    observations_history = { 0, 1, 2, 3, 4 };
    clip_obs = 100.0;
    action_scale_pos = 0.5;
    action_scale_vel = 10.0;
    num_of_dofs = 6;
    lin_vel_scale = 2.0;
    ang_vel_scale = 0.25;
    dof_pos_scale = 1.0;
    dof_vel_scale = 0.05;

    init_tensor_view(clip_actions_upper, {100, 100, 100, 100, 100, 100});
    init_tensor_view(clip_actions_lower, {-100, -100, -100, -100, -100, -100});
    init_tensor_view(commands_scale, {2.0, 0.25, 5.0});
    init_tensor_view(torque_limits, {26, 26, 26, 26, 26, 26});
    init_tensor_view(default_dof_pos, {0.0, 0.0, 0.0, 0.0});
  }
};

