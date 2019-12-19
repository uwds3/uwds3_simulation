import rospy
import numpy as np
#from uwds3_msgs.msg import FeatureStamped
import pybullet as p
from sensor_msgs.msg import JointState
import tf2_ros
from tf2_ros import Buffer, TransformListener, TransformBroadcaster


class InternalSimulator(object):
    def __init__(self, uwds_simulation):

        self.robot_loaded = False
        self.simulator = uwds_simulation
        self.client_simulator_id = self.simulator.client_simulator_id
        self.global_frame_id = self.simulator.global_frame_id
        self.base_frame_id = self.simulator.base_frame_id
        self.robot_urdf_file_path = self.simulator.robot_urdf_file_path

        #self.joint_states_feature_publisher = rospy.Publisher("joint_states_feature", FeatureStamped, queue_size=1)
        rospy.loginfo("[simulation] Subscribing to /joint_states topic...")
        self.joint_state_subscriber = rospy.Subscriber("/joint_states", JointState, self.joint_states_callback, queue_size=1)

    def joint_states_callback(self, joint_states_msg):
        success, t, q = self.simulator.get_transform_from_tf2(self.global_frame_id, self.base_frame_id)
        if success is True:
            if self.robot_loaded is False:
                try:
                    self.robot_loaded = True
                    self.simulator.load_urdf("myself", self.robot_urdf_file_path, t, q)
                except Exception as e:
                    rospy.logwarn("[simulation] Exception occured: {}".format(e))
            try:
                self.simulator.update_entity("myself", t, q)
            except Exception as e:
                rospy.logwarn("[simulation] Exception occured: {}".format(e))
        if self.robot_loaded is True:
            joint_indices = []
            target_positions = []
            base_link_sim_id = self.simulator.entity_id_map["myself"]
            for joint_state_index, joint_name in enumerate(joint_states_msg.name):
                joint_sim_index = self.simulator.joint_id_map[base_link_sim_id][joint_name]
                info = p.getJointInfo(base_link_sim_id, joint_sim_index, physicsClientId=self.client_simulator_id)
                joint_name_sim = info[1]
                current_position = p.getJointState(base_link_sim_id, joint_sim_index)[0]
                assert(joint_name == joint_name_sim)
                joint_position = joint_states_msg.position[joint_state_index]
                if abs(joint_position-current_position) > self.simulator.position_tolerance:
                    joint_indices.append(joint_sim_index)
                    target_positions.append(joint_position)
            p.setJointMotorControlArray(base_link_sim_id,
                                        joint_indices,
                                        controlMode=p.POSITION_CONTROL,
                                        targetPositions=target_positions,
                                        physicsClientId=self.client_simulator_id)
