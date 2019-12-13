#!/usr/bin/env python
# -*- coding: UTF-8 -*-

import rospy
from uwds3_simulation.underworlds_simulation import UnderworldsSimulation

class Uwds3SimulationNode(object):
    def __init__(self):
        rospy.init_node("uwds3_simulation")
        rospy.loginfo("[simulation] Starting Underworlds simulation...")
        self.simulator = UnderworldsSimulation()
        rospy.loginfo("[simulation] Underworlds simulation ready !")

    def run(self):
        while not rospy.is_shutdown():
            rospy.spin()

if __name__ == '__main__':
    simulation = Uwds3SimulationNode().run()
