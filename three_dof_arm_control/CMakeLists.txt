cmake_minimum_required(VERSION 2.8.3)
project(three_dof_arm_control)

find_package(catkin REQUIRED
  roscpp
  rospy
  std_msgs
  control_msgs
)

catkin_package()

install(DIRECTORY config
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})

install(DIRECTORY launch
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION})
