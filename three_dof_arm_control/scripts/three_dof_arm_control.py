#!/usr/bin/env python

import rospy
import roslib
from control_msgs.msg import JointControllerState
from std_msgs.msg import Float64
from geometry_msgs.msg import Pose, Point
import math
import numpy as np
import time

joint1 = 0.0
joint2 = 0.0
joint3 = 0.0

MAX_STEP = 0.2

pub_joint1 = rospy.Publisher('/three_dof_arm/joint1_position_controller/command', Float64, queue_size=10)
pub_joint2 = rospy.Publisher('/three_dof_arm/joint2_position_controller/command', Float64, queue_size=10)
pub_joint3 = rospy.Publisher('/three_dof_arm/joint3_position_controller/command', Float64, queue_size=10)

def joint1_position_callback(data):
    global joint1
    joint1 = data.process_value

def joint2_position_callback(data):
    global joint2
    joint2 = data.process_value

def joint3_position_callback(data):
    global joint3
    joint3 = data.process_value

def forward_kinematics():
    global joint1
    global joint2
    global joint3

    x = 1.0*math.sin(joint2)*math.cos(joint1)*math.cos(joint3) + 1.0*math.sin(joint2)*math.cos(joint1) + 1.0*math.sin(joint3)*math.cos(joint1)*math.cos(joint2)
    y = 1.0*math.sin(joint1)*math.sin(joint2)*math.cos(joint3) + 1.0*math.sin(joint1)*math.sin(joint2) + 1.0*math.sin(joint1)*math.sin(joint3)*math.cos(joint2)
    z = -1.0*math.sin(joint2)*math.sin(joint3) + 1.0*math.cos(joint2)*math.cos(joint3) + 1.0*math.cos(joint2) + 2.0

    return x, y, z

def calc_jacobian():
    global joint1
    global joint2
    global joint3

    a00 = -1.0*math.sin(joint1)*math.sin(joint2)*math.cos(joint3) - 1.0*math.sin(joint1)*math.sin(joint2) - 1.0*math.sin(joint1)*math.sin(joint3)*math.cos(joint2)
    a01 = -1.0*math.sin(joint2)*math.sin(joint3)*math.cos(joint1) + 1.0*math.cos(joint1)*math.cos(joint2)*math.cos(joint3) + 1.0*math.cos(joint1)*math.cos(joint2)
    a02 = -1.0*math.sin(joint2)*math.sin(joint3)*math.cos(joint1) + 1.0*math.cos(joint1)*math.cos(joint2)*math.cos(joint3)

    a10 = 1.0*math.sin(joint2)*math.cos(joint1)*math.cos(joint3) + 1.0*math.sin(joint2)*math.cos(joint1) + 1.0*math.sin(joint3)*math.cos(joint1)*math.cos(joint2)
    a11 = -1.0*math.sin(joint1)*math.sin(joint2)*math.sin(joint3) + 1.0*math.sin(joint1)*math.cos(joint2)*math.cos(joint3) + 1.0*math.sin(joint1)*math.cos(joint2)
    a12 = -1.0*math.sin(joint1)*math.sin(joint2)*math.sin(joint3) + 1.0*math.sin(joint1)*math.cos(joint2)*math.cos(joint3)

    a20 = 0
    a21 = -1.0*math.sin(joint2)*math.cos(joint3) - 1.0*math.sin(joint2) - 1.0*math.sin(joint3)*math.cos(joint2)
    a22 = -1.0*math.sin(joint2)*math.cos(joint3) - 1.0*math.sin(joint3)*math.cos(joint2)

    j = np.array([      [a00, a01, a02], \
                        [a10, a11, a12], \
                        [a20, a21, a22]
                ], dtype="float")
    return j


def inverse_kinematics_callback(data):
    global joint1
    global joint2
    global joint3
    global pub_joint1
    global pub_joint2
    global pub_joint3

    position = Point()

    while (True):
        position.x, position.y, position.z = forward_kinematics()
        delta_end = np.array([data.position.x-position.x, data.position.y-position.y, data.position.z-position.z])
        distance = math.sqrt(delta_end[0]**2 + delta_end[1]**2 + delta_end[2]**2)

        print(position.x, position.y, position.z, distance)

        if (distance < 0.1):
            break

        # print(distance)

        while (distance > MAX_STEP):
            delta_end = delta_end/2

            distance = math.sqrt(delta_end[0]**2 + delta_end[1]**2 + delta_end[2]**2)

        j = calc_jacobian()

        j_inv = np.linalg.pinv(j)

        delta_angles = j_inv.dot(delta_end)

        pub_joint1.publish(joint1+delta_angles[0])
        pub_joint2.publish(joint2+delta_angles[1])
        pub_joint3.publish(joint3+delta_angles[2])

        time.sleep(0.1)

if __name__ == "__main__":
    global joint1
    global joint2
    global joint3

    rospy.init_node("three_dof_arm_control", anonymous=False)

    rospy.Subscriber("/three_dof_arm/joint1_position_controller/state", JointControllerState, joint1_position_callback)
    rospy.Subscriber("/three_dof_arm/joint2_position_controller/state", JointControllerState, joint2_position_callback)
    rospy.Subscriber("/three_dof_arm/joint3_position_controller/state", JointControllerState, joint3_position_callback)

    rospy.Subscriber("/three_dof_arm/inverse_kinematics", Pose, inverse_kinematics_callback)

    rospy.spin()
