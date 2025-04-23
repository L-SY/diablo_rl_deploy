cmake_minimum_required(VERSION 3.10)
project(wheeled_bipedal_rl_controller)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Add DCMAKE_CURRENT_SOURCE_DIR
add_definitions(-DCMAKE_CURRENT_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

add_definitions(-Wall -Werror)

set(CATKIN_PACKAGE_DEPENDENCIES
        roscpp

        effort_controllers
        pluginlib
        controller_interface
        realtime_tools
        robot_common
        rl_sdk
        gazebo_ros
        rl_msgs
)

## Find catkin macros and libraries
find_package(catkin REQUIRED COMPONENTS
        ${CATKIN_PACKAGE_DEPENDENCIES}
)

find_package(Torch REQUIRED)
find_package(Eigen3 REQUIRED)

###################################
## catkin specific configuration ##
###################################
catkin_package(
        INCLUDE_DIRS
        include
        ${EIGEN3_INCLUDE_DIR}
        CATKIN_DEPENDS
        ${CATKIN_PACKAGE_DEPENDENCIES}
        LIBRARIES
        ${PROJECT_NAME}
        DEPENDS
)

include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${TORCH_INCLUDE_DIRS}

)

add_library(vmc
        src/vmc/SerialVMC.cpp
)

target_link_libraries(vmc
        ${catkin_LIBRARIES}
        ${TORCH_LIBRARIES}
)

add_library(${PROJECT_NAME}
        src/WheeledBipedalRLController.cpp
)

target_link_libraries(${PROJECT_NAME}
        vmc
        ${catkin_LIBRARIES}
        ${TORCH_LIBRARIES}
)

## Declare cpp executables
add_executable(obs2action
        test/obs2action.cpp
)

target_link_libraries(obs2action
        ${catkin_LIBRARIES}
        ${TORCH_LIBRARIES}
)

#############
## Install ##
#############

## Mark executables and/or libraries for installation
install(
        TARGETS ${PROJECT_NAME}
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Mark cpp header files for installation
install(
        DIRECTORY include/${PROJECT_NAME}/
        DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
        FILES_MATCHING PATTERN "*.h"
)
install(
        FILES wheeled_bipedal_rl_controller_plugins.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

## Mark other files for installation
#install(
#        DIRECTORY test
#        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
#)

#############
## Testing ##
#############

#if (${CATKIN_ENABLE_TESTING})
#  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread")
#  ## Add gtest based cpp test target and link libraries
#  catkin_add_gtest(${PROJECT_NAME}-test
#      test/test_ros_package_template.cpp
#      test/AlgorithmTest.cpp)
#  target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME}_core)
#endif ()
