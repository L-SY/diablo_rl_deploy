//
// Created by lsy on 24-5-23.
//

#include <string>
#include <pluginlib/class_list_macros.h>
#include "wheeled_bipedal_rl_controller/WheeledBipedalRLController.h"

namespace rl_controller
{
bool WheeledBipedalRLController::init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& root_nh,
                                              ros::NodeHandle& controller_nh)
{
  // Hardware interface
  imuSensorHandle_ = robot_hw->get<hardware_interface::ImuSensorInterface>()->getHandle("base_imu");
  robotStateHandle_ = robot_hw->get<hardware_interface::RobotStateInterface>()->getHandle("robot_state");

  // gazebo_service
  controller_nh.param<std::string>("gazebo_model_name", gazebo_model_name_, "");
  gazebo_set_model_state_client_ = controller_nh.serviceClient<gazebo_msgs::SetModelState>("/gazebo/set_model_state");
  gazebo_pause_physics_client_ = controller_nh.serviceClient<std_srvs::Empty>("/gazebo/pause_physics");
  gazebo_unpause_physics_client_ = controller_nh.serviceClient<std_srvs::Empty>("/gazebo/unpause_physics");

  cmdSub_ = controller_nh.subscribe("command", 1, &WheeledBipedalRLController::commandCB, this);

  // rl_interface
  robotStatePub_ = controller_nh.advertise<rl_msgs::RobotState>("/rl/robot_state", 1);
  rlCommandSub_ = controller_nh.subscribe("/rl/command", 1, &WheeledBipedalRLController::rlCommandCB, this);

  controller_nh.param("default_length", default_length_, 0.18);

  // prostrate
  ros::NodeHandle prostrate_nh(controller_nh, "prostrate");
  prostrate_nh.param("hip", prostrateHip_, 0.);
  prostrate_nh.param("knee", prostrateKnee_, 0.);

  // gravity_ff
  controller_nh.param("add_gravity_ff", addGravityFF_, false);
  controller_nh.param("action_inertia", actionInertia_, 0.1);

  if (addGravityFF_)
  {
    ros::NodeHandle vmc_nh(controller_nh, "vmc");
    double left_l1, left_l2, right_l1, right_l2;
    vmc_nh.param("hip_bias", hipBias_, 0.);
    vmc_nh.param("knee_bias", kneeBias_, 0.);
    vmc_nh.param("gravity_feedforward", gravityFeedforward_, 50.0);

    vmc_nh.param("left_vmc/l1", left_l1, 0.14);
    vmc_nh.param("left_vmc/l2", left_l2, 0.14);
    vmc_nh.param("right_vmc/l1", right_l1, 0.14);
    vmc_nh.param("right_vmc/l2", right_l2, 0.14);

    ros::NodeHandle vmc_left_nh(controller_nh, std::string("vmc_left"));
    ros::NodeHandle vmc_right_nh(controller_nh, std::string("vmc_left"));
    leftSerialVMCPtr_ = std::make_shared<vmc::SerialVMC>(left_l1,left_l2, vmc_left_nh);
    rightSerialVMCPtr_ = std::make_shared<vmc::SerialVMC>(right_l1,right_l2, vmc_right_nh);
  }

  auto* effortJointInterface = robot_hw->get<hardware_interface::EffortJointInterface>();
  jointHandles_.push_back(effortJointInterface->getHandle("left_fake_hip_joint"));
  jointHandles_.push_back(effortJointInterface->getHandle("left_hip_joint"));
  jointHandles_.push_back(effortJointInterface->getHandle("left_wheel_joint"));
  jointHandles_.push_back(effortJointInterface->getHandle("right_fake_hip_joint"));
  jointHandles_.push_back(effortJointInterface->getHandle("right_hip_joint"));
  jointHandles_.push_back(effortJointInterface->getHandle("right_wheel_joint"));

  // Low-level-controller
  Pids_.resize(jointHandles_.size());
  for (size_t i = 0; i < jointHandles_.size(); ++i)
  {
    ros::NodeHandle joint_nh(controller_nh, std::string("gains/") + jointHandles_[i].getName());
    Pids_[i].reset();
    if (!Pids_[i].init(joint_nh))
    {
      ROS_WARN_STREAM("Failed to initialize PID gains from ROS parameter server.");
      return false;
    }
  }
  ROS_INFO_STREAM("Load lower-level controller right");

  // dynamic integral compensation controller
  ros::NodeHandle compensation_nh(controller_nh, std::string("gains/") + "dynamic_integral_compensation");
  dynamicIntegralCompensation_.reset();
  dynamicIntegralCompensation_.init(compensation_nh);

  initStateMsg();

  actions_.resize(6);
  lastAction_.resize(6);

  geometry_msgs::Twist initTwist{};
  cmdRtBuffer_.initRT(initTwist);
  return true;
}

void WheeledBipedalRLController::starting(const ros::Time& /*unused*/)
{
  ROS_INFO_STREAM("WheeledBipedalRLController Starting!");
  controllerState_ = NORMAL;
}

void WheeledBipedalRLController::update(const ros::Time& time, const ros::Duration& period)
{
  if (useVMC_ || addGravityFF_)
  {
      leftSerialVMCPtr_->update(
          M_PI + jointHandles_[0].getPosition() - hipBias_ ,jointHandles_[0].getVelocity(), jointHandles_[0].getEffort(),
          jointHandles_[1].getPosition() - M_PI - kneeBias_, jointHandles_[1].getVelocity(), jointHandles_[1].getEffort());
      rightSerialVMCPtr_->update(
          M_PI + jointHandles_[3].getPosition() - hipBias_,jointHandles_[3].getVelocity(), jointHandles_[3].getEffort(),
          jointHandles_[4].getPosition() - M_PI - kneeBias_, jointHandles_[4].getVelocity(), jointHandles_[4].getEffort());
  }
  rl(time,period);
//  prostrate(time,period);
  pubRLState();
}

void WheeledBipedalRLController::prostrate (const ros::Time& time, const ros::Duration& period)
{
  auto rt_buffer = cmdRtBuffer_.readFromRT();

  jointHandles_[0].setCommand(Pids_[0].computeCommand(prostrateHip_ - jointHandles_[0].getPosition(),period));
  jointHandles_[1].setCommand(Pids_[1].computeCommand(prostrateKnee_ - jointHandles_[1].getPosition(),period));
  jointHandles_[3].setCommand(Pids_[3].computeCommand(prostrateHip_ - jointHandles_[3].getPosition(),period));
  jointHandles_[4].setCommand(Pids_[4].computeCommand(prostrateKnee_ - jointHandles_[4].getPosition(),period));
  // wheel
  double Vx = rt_buffer->linear.x;
  double Vyaw = rt_buffer->angular.z;
  double Vleft = Vx / 2 - Vyaw;
  double Vright = Vx / 2 + Vyaw;
  jointHandles_[2].setCommand(Pids_[2].computeCommand(Vleft-jointHandles_[2].getVelocity(),period));
  jointHandles_[5].setCommand(Pids_[5].computeCommand(Vright-jointHandles_[5].getVelocity(),period));
}

void WheeledBipedalRLController::rl(const ros::Time& time, const ros::Duration& period)
{
//TODO ：add set control.x, control.y, control.yaw.
  setCommand(time, period);
}

void WheeledBipedalRLController::commandCB(const geometry_msgs::Twist& msg)
{
  cmdRtBuffer_.writeFromNonRT(msg);
}

void WheeledBipedalRLController::rlCommandCB(const std_msgs::Float64MultiArray&  msg)
{
  rlCmdRtBuffer_.writeFromNonRT(msg);
}

void WheeledBipedalRLController::pubRLState()
{
  robotStateMsg_.stamp = ros::Time::now();
  robotStateMsg_.imu_states.header.frame_id = imuSensorHandle_.getName();
  robotStateMsg_.imu_states.orientation.x = imuSensorHandle_.getOrientation()[0];
  robotStateMsg_.imu_states.orientation.y = imuSensorHandle_.getOrientation()[1];
  robotStateMsg_.imu_states.orientation.z = imuSensorHandle_.getOrientation()[2];
  robotStateMsg_.imu_states.orientation.w = imuSensorHandle_.getOrientation()[3];

  geometry_msgs::Quaternion base;
  double roll, yaw;
  base.x = imuSensorHandle_.getOrientation()[0];
  base.y = imuSensorHandle_.getOrientation()[1];
  base.z = imuSensorHandle_.getOrientation()[2];
  base.w = imuSensorHandle_.getOrientation()[3];
  robot_common::quatToRPY(base,roll,basePitch_,yaw);

  robotStateMsg_.rpy[0] = roll;
  robotStateMsg_.rpy[1] = basePitch_;
  robotStateMsg_.rpy[2] = yaw;

  robotStateMsg_.imu_states.angular_velocity.x = imuSensorHandle_.getAngularVelocity()[0];
  robotStateMsg_.imu_states.angular_velocity.y = imuSensorHandle_.getAngularVelocity()[1];
  robotStateMsg_.imu_states.angular_velocity.z = imuSensorHandle_.getAngularVelocity()[2];

  robotStateMsg_.imu_states.linear_acceleration.x = imuSensorHandle_.getLinearAcceleration()[0];
  robotStateMsg_.imu_states.linear_acceleration.y = imuSensorHandle_.getLinearAcceleration()[1];
  robotStateMsg_.imu_states.linear_acceleration.z = imuSensorHandle_.getLinearAcceleration()[2];
  for(size_t i = 0; i < jointHandles_.size(); ++i)
  {
    robotStateMsg_.joint_states.name[i] = jointHandles_[i].getName();
    robotStateMsg_.joint_states.position[i] = jointHandles_[i].getPosition();
    robotStateMsg_.joint_states.velocity[i] = jointHandles_[i].getVelocity();
    robotStateMsg_.joint_states.effort[i] = jointHandles_[i].getEffort();
  }
  auto cmdBuff = cmdRtBuffer_.readFromRT();
  double linear_x = cmdBuff->linear.x;
  double angular_z = cmdBuff->angular.z;
  double linear_z = cmdBuff->linear.z;

  robotStateMsg_.commands[0] = linear_x;
  robotStateMsg_.commands[1] = angular_z;
  robotStateMsg_.commands[2] = default_length_ + linear_z;

  if (linear_x == 0.0 && angular_z == 0.0 && linear_z == 0.0) {
    robotStateMsg_.commands[0] = 0.0;
    robotStateMsg_.commands[1] = 0.0;
    robotStateMsg_.commands[2] = default_length_;
  }

  robotStateMsg_.left.l = leftSerialVMCPtr_->r_;
  robotStateMsg_.left.l_dot = leftSerialVMCPtr_->dr_;
  robotStateMsg_.left.theta = leftSerialVMCPtr_->theta_;
  robotStateMsg_.left.theta_dot = leftSerialVMCPtr_->dtheta_;
  robotStateMsg_.right.l = rightSerialVMCPtr_->r_;
  robotStateMsg_.right.l_dot = rightSerialVMCPtr_->dr_;
  robotStateMsg_.right.theta = rightSerialVMCPtr_->theta_;
  robotStateMsg_.right.theta_dot = rightSerialVMCPtr_->dtheta_;

  for (int i = 0; i < 6; ++i)
  {
    robotStateMsg_.actions[i] = actions_[i];
  }
  robotStatePub_.publish(robotStateMsg_);
}

void WheeledBipedalRLController::setCommand(const ros::Time& time, const ros::Duration& period)
{
  auto rt_buffer = rlCmdRtBuffer_.readFromRT();
  auto& data = rt_buffer->data;
  if (data.empty())
  {
      jointHandles_[0].setCommand(Pids_[0].computeCommand(0 - jointHandles_[0].getPosition(), period));
      jointHandles_[1].setCommand(Pids_[1].computeCommand(0 - jointHandles_[1].getPosition(), period));
      jointHandles_[3].setCommand(Pids_[3].computeCommand(0 - jointHandles_[3].getPosition(), period));
      jointHandles_[4].setCommand(Pids_[4].computeCommand(0 - jointHandles_[4].getPosition(), period));

      jointHandles_[2].setCommand(0);
      jointHandles_[5].setCommand(0);
  }
  else
  {
    for (int i = 0; i < static_cast<int>(data.size()); ++i)
    {
      actions_[i] = data[i];
    }

    std::vector<double> leftJointCmd = {0., 0.};
    std::vector<double> rightJointCmd = {0., 0.};
    if (addGravityFF_)
    {
      leftJointCmd = leftSerialVMCPtr_->getDesJointEff(leftSerialVMCPtr_->phi1_,leftSerialVMCPtr_->phi2_, gravityFeedforward_, 0.);
      rightJointCmd = rightSerialVMCPtr_->getDesJointEff(rightSerialVMCPtr_->phi1_,rightSerialVMCPtr_->phi2_,gravityFeedforward_, 0.);
    }

    double leftHipCmd = Pids_[0].computeCommand(((1-actionInertia_)*actions_[0] + actionInertia_*lastAction_[0])-jointHandles_[0].getPosition(),period) + leftJointCmd[0];
    double leftKneeCmd = Pids_[1].computeCommand(((1-actionInertia_)*actions_[1] + actionInertia_*lastAction_[1])-jointHandles_[1].getPosition(),period) + leftJointCmd[1];
    double rightHipCmd = Pids_[3].computeCommand(((1-actionInertia_)*actions_[3] + actionInertia_*lastAction_[3])-jointHandles_[3].getPosition(),period) + rightJointCmd[0];
    double rightKneeCmd = Pids_[4].computeCommand(((1-actionInertia_)*actions_[4] + actionInertia_*lastAction_[4])-jointHandles_[4].getPosition(),period) + rightJointCmd[1];

    jointHandles_[0].setCommand(leftHipCmd);
    jointHandles_[1].setCommand(leftKneeCmd);
    jointHandles_[3].setCommand(rightHipCmd);
    jointHandles_[4].setCommand(rightKneeCmd);
    jointHandles_[2].setCommand(Pids_[2].computeCommand(actions_[2]-jointHandles_[2].getVelocity(),period));
    jointHandles_[5].setCommand(Pids_[5].computeCommand(actions_[5]-jointHandles_[5].getVelocity(),period));
  }
  lastAction_ = data;
}

void WheeledBipedalRLController::initStateMsg()
{
  robotStateMsg_.stamp = ros::Time::now();
  robotStateMsg_.imu_states = sensor_msgs::Imu();

  size_t num_joints = jointHandles_.size();
  auto& joints = robotStateMsg_.joint_states;
  joints.name.resize(num_joints);
  joints.position.assign(num_joints, 0.0);
  joints.velocity.assign(num_joints, 0.0);
  joints.effort.assign(num_joints, 0.0);

  robotStateMsg_.commands = std::vector<double>{0.0, 0.0, default_length_};

  robotStateMsg_.rpy = std::vector<double>(3, 0.0);
  robotStateMsg_.actions = std::vector<double>(6, 0.0);
}


}  // namespace rl_controller

PLUGINLIB_EXPORT_CLASS(rl_controller::WheeledBipedalRLController, controller_interface::ControllerBase)