cmake_minimum_required(VERSION 3.0.2)
project(rqt_sine_publisher)

find_package(catkin REQUIRED COMPONENTS
        rospy
        rqt_gui
        rqt_gui_py
        std_msgs
)

catkin_python_setup()

catkin_package(
        CATKIN_DEPENDS rospy rqt_gui rqt_gui_py std_msgs
)

install(FILES plugin.xml
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(DIRECTORY resource
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
)

install(
        DIRECTORY src/${PROJECT_NAME}
        DESTINATION ${CATKIN_PACKAGE_PYTHON_DESTINATION}
)