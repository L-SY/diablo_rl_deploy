//
// Created by lsy on 24-5-23.
//

#pragma once

#include <effort_controllers/joint_effort_controller.h>
#include <effort_controllers/joint_position_controller.h>
#include <effort_controllers/joint_velocity_controller.h>
#include <controller_interface/multi_interface_controller.h>
#include <hardware_interface/joint_command_interface.h>
#include <realtime_tools/realtime_publisher.h>
#include <hardware_interface/imu_sensor_interface.h>

// robot_common
#include "robot_common/interface/hardware_interface/robot_state_interface.h"
#include "robot_common/utilities/tf_rt_broadcaster.h"
#include "robot_common/utilities/ori_tool.h"
#include "robot_common/utilities/lp_filter.h"
#include "geometry_msgs/Twist.h"

// Gazebo
#include <gazebo_msgs/SetModelState.h>
#include "std_srvs/Empty.h"

// robot_state
#include "sensor_msgs/JointState.h"
#include "sensor_msgs/Imu.h"
#include "rl_msgs/RobotState.h"
#include "std_msgs/Float64MultiArray.h"

// vmc
#include "vmc/SerialVMC.h"



namespace rl_controller
{
class WheeledBipedalRLController
  : public controller_interface::MultiInterfaceController<hardware_interface::RobotStateInterface,
                                                          hardware_interface::EffortJointInterface,
                                                          hardware_interface::ImuSensorInterface>
{
  enum ControllerState
  {
    NORMAL,
    RL
  };

public:
  WheeledBipedalRLController() = default;
  bool init(hardware_interface::RobotHW* robot_hw, ros::NodeHandle& root_nh, ros::NodeHandle& controller_nh) override;
  void starting(const ros::Time& time) override;
  void update(const ros::Time& time, const ros::Duration& period) override;
  void setCommand(const ros::Time& time, const ros::Duration& period);
  void pubRLState();
  void initStateMsg();
private:
  void prostrate(const ros::Time& time, const ros::Duration& period);
  void rl(const ros::Time& time, const ros::Duration& period);
  void commandCB(const geometry_msgs::Twist& msg);
  void rlCommandCB(const std_msgs::Float64MultiArray& msg);

  int controllerState_ = NORMAL;
  ros::Time startTime_;
  bool stateChanged_ = false;
  ros::Subscriber cmdSub_;

  // Interface
  hardware_interface::RobotStateHandle robotStateHandle_;
  hardware_interface::ImuSensorHandle imuSensorHandle_;
  std::vector<hardware_interface::JointHandle> jointHandles_;

  //  Gazebo Service
  std::string gazebo_model_name_;
  ros::ServiceClient gazebo_set_model_state_client_;
  ros::ServiceClient gazebo_pause_physics_client_;
  ros::ServiceClient gazebo_unpause_physics_client_;

  realtime_tools::RealtimeBuffer<geometry_msgs::Twist> cmdRtBuffer_{};

  // Low level controller
  std::vector<control_toolbox::Pid> Pids_;
  control_toolbox::Pid dynamicIntegralCompensation_;
  double default_length_;

  // rl_interface
  rl_msgs::RobotState robotStateMsg_;
  ros::Publisher robotStatePub_;
  ros::Subscriber rlCommandSub_;
  realtime_tools::RealtimeBuffer<std_msgs::Float64MultiArray> rlCmdRtBuffer_{};

  // vmc
  bool useVMC_, addGravityFF_;
  double gravityFeedforward_, hipBias_, kneeBias_, actionInertia_;
  std::shared_ptr<vmc::SerialVMC> leftSerialVMCPtr_;
  std::shared_ptr<vmc::SerialVMC> rightSerialVMCPtr_;

  // prostrate
  double prostrateHip_;
  double prostrateKnee_;

  double basePitch_;
  std::vector<double> actions_;
  std::vector<double> lastAction_;
};

}  // namespace rl_controller