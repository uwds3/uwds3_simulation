import rospy
import pybullet as p
import numpy as np
from proprioception.joint_states_listener import JointStatesListener
from estimation.perspective_estimator import PerspectiveEstimator
import tf2_ros
import message_filters
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from tf2_ros import Buffer, TransformListener, TransformBroadcaster
from uwds3_msgs.msg import SceneChangesStamped, PrimitiveShape
from visualization_msgs.msg import MarkerArray, Marker
from tf.transformations import translation_matrix, quaternion_matrix, quaternion_from_matrix, translation_from_matrix, quaternion_from_euler
import yaml


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

        self.bridge = CvBridge()

        self.entity_id_map = {}
        self.reverse_entity_id_map = {}

        self.joint_id_map = {}
        self.reverse_joint_id_map = {}

        self.constraint_id_map = {}

        self.markers_id_map = {}

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
                                         entity["orientation"]["z"]]
                    start_orientation = quaternion_from_euler(entity["orientation"]["x"],
                                                              entity["orientation"]["y"],
                                                              entity["orientation"]["z"],
                                                              'rxyz')
                    self.load_urdf(entity["id"], entity["file"], start_position, start_orientation)

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

        self.joint_states_listener = JointStatesListener(self)

        self.perspective_estimator = PerspectiveEstimator(self)

        self.use_depth = rospy.get_param("~use_depth", False)

        self.publish_markers = rospy.get_param("publish_markers", True)

        self.publish_perspectives = rospy.get_param("publish_perspectives", True)
        if self.publish_perspectives is True:
            self.perspective_publisher = rospy.Publisher("perspective_viz", Image, queue_size=1)

        if self.publish_markers is True:
            self.marker_publisher = rospy.Publisher("internal_simulation_viz", MarkerArray, queue_size=1)

        self.track_sub = rospy.Subscriber(self.tracks_topic, SceneChangesStamped, self.observation_callback)

        if self.publish_markers is True:
            self.simulation_timer = rospy.Timer(rospy.Duration(1.0/24.0), self.visualization_callback)

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

    def add_shape(self, track):
        if track.has_shape is True:
            if track.shape.type == PrimitiveShape.CYLINDER:
                position = []
                orientation = []
                visual_shape_id = p.createVisualShape(p.GEOM_CYLINDER,
                                                      radius=track.shape.dimensions[0],
                                                      length=track.shape.dimensions[1])
                collision_shape_id = p.createCollisionShape(p.GEOM_CYLINDER,
                                                            radius=track.shape.dimensions[0],
                                                            length=track.shape.dimensions[1])
                entity_id = p.createMultiBody(baseCollisionShapeIndex=collision_shape_id,
                                              baseVisualShapeIndex=visual_shape_id,
                                              basePosition=position,
                                              baseOrientation=orientation,
                                              flags=p.URDF_ENABLE_SLEEPING or p.URDF_ENABLE_CACHED_GRAPHICS_SHAPES)
                if entity_id >= 0:
                    self.entity_id_map[track.id] = entity_id
                    self.reverse_entity_id_map[entity_id] = track.id
            elif track.shape.type == PrimitiveShape.MESH:
                pass
            else:
                raise NotImplementedError("Only cylinder shapes for unknown objects supported at the moment.")

    def remove_shape(self, track):
        pass

    def update_entity(self, id, t, q):
        if id not in self.entity_id_map:
            raise ValueError("Entity <{}> is not loaded into the simulator".format(id))
        base_link_sim_id = self.entity_id_map[id]
        t_current, q_current = p.getBasePositionAndOrientation(base_link_sim_id)
        update_position = not np.allclose(np.array(t_current), t, atol=self.position_tolerance)
        update_orientation = not np.allclose(np.array(q_current), q, atol=self.position_tolerance)
        if update_position is True or update_orientation is True:
            p.resetBasePositionAndOrientation(base_link_sim_id, t, q, physicsClientId=self.client_simulator_id)

    def publish_marker_array(self):
        marker_array = MarkerArray()
        marker_array.markers = []
        now = rospy.Time()
        for sim_id in range(0, p.getNumBodies()):
            visual_shapes = p.getVisualShapeData(sim_id)

            for shape in visual_shapes:
                marker = Marker()
                entity_id = shape[0]
                link_id = shape[1]
                type = shape[2]
                dimensions = shape[3]
                mesh_file_path = shape[4]
                position = shape[5]
                orientation = shape[6]
                rgba_color = shape[7]

                if link_id != -1:
                    link_state = p.getLinkState(sim_id, link_id)
                    t_link = link_state[0]
                    q_link = link_state[1]
                    t_inertial = link_state[2]
                    q_inertial = link_state[3]

                    tf_world_link = np.dot(translation_matrix(t_link), quaternion_matrix(q_link))
                    tf_inertial_link = np.dot(translation_matrix(t_inertial), quaternion_matrix(q_inertial))
                    world_transform = np.dot(tf_world_link, np.linalg.inv(tf_inertial_link))
                else:
                    t_link, q_link = p.getBasePositionAndOrientation(sim_id)
                    world_transform = np.dot(translation_matrix(t_link), quaternion_matrix(q_link))

                marker.header.frame_id = self.global_frame_id
                marker.header.stamp = now
                marker.id = entity_id + (link_id + 1) << 24 # same id convention than bullet
                marker.action = Marker.MODIFY
                marker.ns = self.reverse_entity_id_map[sim_id]

                trans_shape = np.dot(translation_matrix(position), quaternion_matrix(orientation))
                shape_transform = np.dot(world_transform, trans_shape)
                position = translation_from_matrix(shape_transform)
                orientation = quaternion_from_matrix(shape_transform)

                marker.pose.position.x = position[0]
                marker.pose.position.y = position[1]
                marker.pose.position.z = position[2]

                marker.pose.orientation.x = orientation[0]
                marker.pose.orientation.y = orientation[1]
                marker.pose.orientation.z = orientation[2]
                marker.pose.orientation.w = orientation[3]

                if len(rgba_color) > 0:
                    marker.color.r = rgba_color[0]
                    marker.color.g = rgba_color[1]
                    marker.color.b = rgba_color[2]
                    marker.color.a = rgba_color[3]

                if type == p.GEOM_SPHERE:
                    marker.type = Marker.SPHERE
                    marker.scale.x = dimensions[0]*2.0
                    marker.scale.y = dimensions[0]*2.0
                    marker.scale.z = dimensions[0]*2.0
                elif type == p.GEOM_BOX:
                    marker.type = Marker.CUBE
                    marker.scale.x = dimensions[0]
                    marker.scale.y = dimensions[1]
                    marker.scale.z = dimensions[2]
                elif type == p.GEOM_CAPSULE:
                    marker.type = Marker.SPHERE
                elif type == p.GEOM_CYLINDER:
                    marker.type = Marker.CYLINDER
                    marker.scale.x = dimensions[1]*2.0
                    marker.scale.y = dimensions[1]*2.0
                    marker.scale.z = dimensions[0]
                elif type == p.GEOM_PLANE:
                    marker.type = Marker.CUBE
                    marker.scale.x = dimensions[0]
                    marker.scale.y = dimensions[1]
                    marker.scale.z = 0.0001
                elif type == p.GEOM_MESH:
                    marker.type = Marker.MESH_RESOURCE
                    marker.mesh_resource = "file://"+mesh_file_path
                    marker.scale.x = dimensions[0]
                    marker.scale.y = dimensions[1]
                    marker.scale.z = dimensions[2]
                    marker.mesh_use_embedded_materials = True
                else:
                    raise NotImplementedError

                marker.lifetime = rospy.Duration(1.0)

                marker_array.markers.append(marker)
        self.marker_publisher.publish(marker_array)

    def observation_callback(self, tracks_msg):
        p.stepSimulation()
        for track in tracks_msg.changes.nodes:
            if track.has_camera is True and track.is_located is True:
                position = track.pose_stamped.pose.pose.position
                orientation = track.pose_stamped.pose.pose.orientation
                t = [position.x, position.y, position.z]
                q = [orientation.x, orientation.y, orientation.z, orientation.w]
                rgb_image, depth_image, detections, viz_frame = self.perspective_estimator.estimate(t, q, track.camera)
                viz_img_msg = self.bridge.cv2_to_imgmsg(viz_frame)
                self.perspective_publisher.publish(viz_img_msg)

    def visualization_callback(self, event):
        self.publish_marker_array()
