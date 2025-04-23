#include "rl_sdk/rl_sdk.h"
#include "iostream"
#include "observation_buffer.hpp"

torch::Tensor rl_sdk::ComputeObservation()
{
  std::vector<torch::Tensor> obs_list;

  // "ang_vel", "gravity_vec", "commands", "dof_pos", "dof_vel", "actions"
  obs_list.push_back(obs.ang_vel * params->ang_vel_scale);
  obs_list.push_back(QuatRotateInverse(obs.base_quat, obs.gravity_vec, params->framework));
  obs_list.push_back(obs.commands * params->commands_scale);
  obs_list.push_back((obs.dof_pos - params->default_dof_pos) * params->dof_pos_scale);
  obs_list.push_back(obs.dof_vel * params->dof_vel_scale);
  obs_list.push_back(obs.actions);

  torch::Tensor obs = torch::cat(obs_list, 1);
  torch::Tensor clamped_obs = torch::clamp(obs, -params->clip_obs, params->clip_obs);
  return clamped_obs;
}

void rl_sdk::InitRL()
{
  obs = Observations{
    .lin_vel     = torch::zeros({1, 3}),
    .ang_vel     = torch::zeros({1, 3}),
    .gravity_vec = torch::tensor({{0.0, 0.0, -1.0}}),
    .commands    = torch::zeros({1, 3}),
    .base_quat   = torch::tensor({{0.0, 0.0, 0.0, 1.0}}),
    .dof_pos     = torch::zeros({1, 4}),
    .dof_vel     = torch::zeros({1, params->num_of_dofs}),
    .actions     = torch::zeros({1, params->num_of_dofs})
  };
  history_obs_buf = ObservationBuffer(1, params->num_observations, params->observations_history.size());

  output_command = torch::zeros({ 1, params->num_of_dofs });
  output_dof_pos = params->default_dof_pos;

  control = Control{.vel_x = 0.0, .vel_yaw = 0.0, .pos_z = 0.0};
}

torch::Tensor rl_sdk::ComputeCommand(torch::Tensor actions)
{
  torch::Tensor actions_scaled = actions * params->action_scale_pos;
  double scale_factor = params->action_scale_vel / params->action_scale_pos;

  actions_scaled[0][2] *= scale_factor;
  actions_scaled[0][5] *= scale_factor;
  torch::Tensor command = actions_scaled;

  return command;
}

torch::Tensor rl_sdk::QuatRotateInverse(torch::Tensor q, torch::Tensor v, const std::string& framework)
{
  torch::Tensor q_w, q_vec;
  if (framework == "isaacsim") {
    q_w = q.select(1, 0);
    q_vec = q.slice(1, 1, 4);
  } else if (framework == "isaacgym") {
    q_w = q.select(1, 3);
    q_vec = q.slice(1, 0, 3);
  } else {
    throw std::invalid_argument("Unknown framework: " + framework);
  }

  const auto B = q.size(0);
  const auto w2 = q_w.pow(2).unsqueeze(-1);

  torch::Tensor a = v * (2.0 * w2 - 1.0);
  torch::Tensor b = 2.0 * q_w.unsqueeze(-1) * torch::cross(q_vec, v, -1);
  torch::Tensor c = 2.0 * q_vec * torch::bmm(q_vec.view({B, 1, 3}),v.view({B, 3, 1})).squeeze(-1);

  return a - b + c;
}


torch::Tensor rl_sdk::Forward()
{
  torch::autograd::GradMode::set_enabled(false);
  torch::Tensor clamped_obs = ComputeObservation();
  torch::Tensor actions;
  torch::Tensor latent;

  if(params->use_history)
  {
    history_obs_buf.insert(clamped_obs);
    history_obs = history_obs_buf.get_obs_vec(this->params->observations_history);

    std::vector<torch::jit::IValue> inputs;
    inputs.emplace_back(clamped_obs);
    inputs.emplace_back(history_obs);

    auto output = model.forward(inputs).toTuple();

    actions = output->elements()[0].toTensor();
    latent = output->elements()[1].toTensor();
  }
  else
    actions = model.forward({ clamped_obs }).toTensor();

  return actions;
}

void rl_sdk::SetObservation()
{
  obs.ang_vel = torch::tensor(robot_state.imu.gyroscope).unsqueeze(0);
  obs.commands = torch::tensor({ { control.vel_x, control.vel_yaw, control.pos_z } });
  obs.base_quat = torch::tensor(robot_state.imu.quaternion).unsqueeze(0);
  obs.dof_pos = torch::tensor(robot_state.motor_state.q).narrow(0, 0, params->num_of_dofs - 2).unsqueeze(0);
  obs.dof_vel = torch::tensor(robot_state.motor_state.dq).narrow(0, 0, params->num_of_dofs).unsqueeze(0);
  //  obs.actions = torch::tensor(robot_state.actions).narrow(0, 0, params->num_of_dofs).unsqueeze(0);
}