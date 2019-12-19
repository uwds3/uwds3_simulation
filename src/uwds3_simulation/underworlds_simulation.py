import rospy
import pybullet as p
import numpy as np
from proprioception.internal_simulator import InternalSimulator
from monitoring.perspective_monitor import PerspectiveMonitor
import tf2_ros
import message_filters
from sensor_msgs.msg import Image
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from uwds3_msgs.msg import SceneChangesStamped
import yaml


class PrimitiveShape(object):
    BOX = 0
    SPHERE = 1
    CYLINDER = 2
    CAPSULE = 3


class UnderworldsSimulation(object):
    def __init__(self):
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer)
        self.tf_broadcaster = TransformBroadcaster()

        self.base_frame_id = rospy.get_param("~base_frame_id", "base_footprint")
        self.global_frame_id = rospy.get_param("~global_frame_id", "odom")

        self.env_urdf_file_path = rospy.get_param("~env_urdf_file_path", "")
        self.cad_models_additional_search_path = rospy.get_param("~cad_models_additional_search_path", "")

        self.static_entities_config_filename = rospy.get_param("~static_entities_config_filename", "")

        self.entity_id_map = {}
        self.reverse_entity_id_map = {}

        self.joint_id_map = {}
        self.reverse_joint_id_map = {}

        self.constraint_id_map = {}

        self.use_gui = rospy.get_param("~use_gui", True)
        if self.use_gui is True:
            self.client_simulator_id = p.connect(p.GUI)
        else:
            self.client_simulator_id = p.connect(p.DIRECT)

        if self.cad_models_additional_search_path != "":
            p.setAdditionalSearchPath(self.cad_models_additional_search_path)

        if self.env_urdf_file_path != "":
            self.load_urdf(self.global_frame_id, self.env_urdf_file_path, [0,0,0], [0,0,0,1])

        if self.static_entities_config_filename != "":
            with open(self.static_entities_config_filename, 'r') as stream:
                static_entities = yaml.safe_load(stream)
                for entity in static_entities:
                    start_position = [entity["position"]["x"],
                                      entity["position"]["y"],
                                      entity["position"]["z"]]
                    start_orientation = [entity["orientation"]["x"],
                                         entity["orientation"]["y"],
                                         entity["orientation"]["z"],
                                         entity["orientation"]["w"]]
                    self.load_urdf(entity["label"], entity["file"], start_position, start_orientation)

        p.setGravity(0, 0, -10)
        p.setRealTimeSimulation(0)

        self.rgb_image_topic = rospy.get_param("~rgb_image_topic", "/camera/rgb/image_raw")
        self.rgb_camera_info_topic = rospy.get_param("~rgb_camera_info_topic", "/camera/rgb/camera_info")

        self.depth_image_topic = rospy.get_param("~depth_image_topic", "/camera/depth/image_raw")
        self.depth_camera_info_topic = rospy.get_param("~depth_camera_info_topic", "/camera/depth/camera_info")

        self.tracks_topic = rospy.get_param("~tracks_topic", "tracks")

        self.position_tolerance = rospy.get_param("~position_tolerance", 0.001)
        self.orientation_tolerance = rospy.get_param("~orientation_tolerance", 0.001)

        self.robot_urdf_file_path = rospy.get_param("~robot_urdf_file_path", "r2d2.urdf")

        self.internal_simulator = InternalSimulator(self)

        self.perspective_monitor = PerspectiveMonitor(self)

        self.use_depth = rospy.get_param("~use_depth", False)

        if self.use_depth is True:
            rospy.loginfo("[simulation] Subscribing to '/{}' topic...".format(self.tracks_topic))
            self.tracks_sub = message_filters.Subscriber(self.tracks_topic, SceneChangesStamped)

            rospy.loginfo("[simulation] Subscribing to '/{}' topic...".format(self.rgb_image_topic))
            self.rgb_image_sub = message_filters.Subscriber(self.rgb_image_topic, Image)

            rospy.loginfo("[simulation] Subscribing to '/{}' topic...".format(self.depth_image_topic))
            self.depth_image_sub = message_filters.Subscriber(self.depth_image_topic, Image)

            self.sync = message_filters.ApproximateTimeSynchronizer([self.tracks_sub, self.rgb_image_sub, self.depth_image_sub], 10, 0.1, allow_headerless=True)
            self.sync.registerCallback(self.observation_callback)
        else:
            rospy.loginfo("[simulation] Subscribing to '/{}' topic...".format(self.tracks_topic))
            self.tracks_sub = message_filters.Subscriber(self.tracks_topic, SceneChangesStamped)

            rospy.loginfo("[simulation] Subscribing to '/{}' topic...".format(self.rgb_image_topic))
            self.rgb_image_sub = message_filters.Subscriber(self.rgb_image_topic, Image)

            self.sync = message_filters.ApproximateTimeSynchronizer([self.tracks_sub, self.rgb_image_sub], 10, 0.1, allow_headerless=True)
            self.sync.registerCallback(self.observation_callback)

    def load_urdf(self, id, filename, t, q, remove_friction=False):
        try:
            base_link_sim_id = p.loadURDF(filename, t, q, flags=p.URDF_ENABLE_SLEEPING or p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
        except Exception as e:
            rospy.logwarn("[simulation] Error loading '{}': {}".format(filename, e))
            return
        # If file successfully loaded
        if base_link_sim_id >= 0:
            self.entity_id_map[id] = base_link_sim_id
            # Create a joint map to ease exploration
            self.reverse_entity_id_map[base_link_sim_id] = id
            self.joint_id_map[base_link_sim_id] = {}
            self.reverse_joint_id_map[base_link_sim_id] = {}
            for i in range(0, p.getNumJoints(base_link_sim_id, physicsClientId=self.client_simulator_id)):
                info = p.getJointInfo(base_link_sim_id, i, physicsClientId=self.client_simulator_id)
                self.joint_id_map[base_link_sim_id][info[1]] = info[0]
                self.reverse_joint_id_map[base_link_sim_id][info[0]] = info[1]
            rospy.loginfo("[simulation] '{}' File successfully loaded".format(filename))
        else:
            raise ValueError("Invalid URDF file provided: '{}' ".format(filename))

    def get_transform_from_tf2(self, source_frame, target_frame, time=None):
        try:
            if time is not None:
                trans = self.tf_buffer.lookup_transform(source_frame, target_frame, time)
            else:
                trans = self.tf_buffer.lookup_transform(source_frame, target_frame, rospy.Time(0))
            x = trans.transform.translation.x
            y = trans.transform.translation.y
            z = trans.transform.translation.z

            rx = trans.transform.rotation.x
            ry = trans.transform.rotation.y
            rz = trans.transform.rotation.z
            rw = trans.transform.rotation.w

            return True, [x, y, z], [rx, ry, rz, rw]
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException) as e:
            rospy.logwarn("[simulation] Exception occured: {}".format(e))
            return False, [0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]

    def update_constraint(self, id, t, q):
        if id not in self.entity_id_map:
            raise ValueError("Entity <{}> is not loaded into the simulator".format(id))
        if id in self.constraint_id[id]:
            p.changeConstraint(self.constraint_id[id], t, jointChildFrameOrientation=q, maxForce=50)
        else:
            self.constraint_id_map[id] = p.createConstraint(self.entity_id_map[id], -1, -1, -1, p.JOINT_FIXED, [0, 0, 0], [0, 0, 0], t, childFrameOrientation=q)

    def remove_constraint(self, id):
        if id not in self.constraint_id_map:
            raise ValueError("Constraint for entity <{}> not created".format(id))
        p.removeConstraint(self.constraint_id[id])

    def update_entity(self, id, t, q):
        if id not in self.entity_id_map:
            raise ValueError("Entity <{}> is not loaded into the simulator".format(id))
        base_link_sim_id = self.entity_id_map[id]
        t_current, q_current = p.getBasePositionAndOrientation(base_link_sim_id)
        update_position = not np.allclose(np.array(t_current), t, atol=self.position_tolerance)
        update_orientation = not np.allclose(np.array(q_current), q, atol=self.position_tolerance)
        if update_position is True or update_orientation is True:
            p.resetBasePositionAndOrientation(base_link_sim_id, t, q, physicsClientId=self.client_simulator_id)

    def observation_callback(self, tracks_msg, bgr_image_msg, depth_image_msg=None):
        p.stepSimulation()
        for track in tracks_msg.changes.nodes:
            if track.has_camera is True and track.is_located is True:
                position = track.pose_stamped.pose.pose.position
                orientation = track.pose_stamped.pose.pose.orientation
                t = [position.x, position.y, position.z]
                q = [orientation.x, orientation.y, orientation.z, orientation.w]
                self.perspective_monitor.estimate(t, q, track.camera)
