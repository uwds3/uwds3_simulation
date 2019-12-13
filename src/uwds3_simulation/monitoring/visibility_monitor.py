import rospy
import cv2
import numpy as np
import pybullet as p
from sensor_msgs.msg import Image, CameraInfo
from tf.transformations import euler_from_quaternion, inverse_matrix, compose_matrix, quaternion_matrix, translation_matrix, translation_from_matrix
#from uwds3_human_description.human_visual_model import HumanVisualModel
from cv_bridge import CvBridge

class HumanVisualModel(object):
    FOV = 60.0 # human field of view
    WIDTH = 90 # image width resolution for rendering
    HEIGHT = 68  # image height resolution for rendering
    CLIPNEAR = 0.001 # clipnear
    CLIPFAR = 1e+3 # clipfar
    ASPECT = 1.333 # aspect ratio for rendering
    SACCADE_THRESHOLD = 0.01 # angular variation in rad/s
    SACCADE_ESPILON = 0.005 # error in angular variation
    FOCUS_DISTANCE_FIXATION = 0.1 # focus distance when performing a fixation
    FOCUS_DISTANCE_SACCADE = 0.5 # focus distance when performing a saccade

    def get_camera_info(self):
        camera_info = CameraInfo()
        width = HumanVisualModel.WIDTH
        height = HumanVisualModel.HEIGHT
        camera_info.width = width
        camera_info.height = height
        focal_length = height
        center = (height/2, width/2)
        camera_matrix = np.array([[focal_length, 0, center[0]],
                                 [0, focal_length, center[1]],
                                 [0, 0, 1]], dtype="double")
        P_matrix = np.array([[focal_length, 0, center[0], 0],
                            [0, focal_length, center[1], 0],
                            [0, 0, 1, 0]], dtype="double")

        dist_coeffs = np.zeros((4, 1))
        camera_info.distortion_model = "blob"
        camera_info.D = list(dist_coeffs)
        camera_info.K = list(camera_matrix.flatten())
        camera_info.P = list(P_matrix.flatten())
        return camera_info

class VisibilityMonitor(object):
    def __init__(self, uwds_simulation):
        self.simulator = uwds_simulation
        self.filter_modes = ["MASK", "IMGFILTER", "ZFILTER", "ALLFILTER"]
        self.bridge = CvBridge()
        self.perspective_publisher = rospy.Publisher("perspective_viz", Image, queue_size=1)

    def estimate(self, t, q, camera_info, focus_distance=1.0, mode="MASK"):
        detections = np.array([])

        rot = quaternion_matrix(q)
        trans = translation_matrix(t)
        target = translation_matrix([0.0, 0.0, 1000.0])
        target_position = translation_from_matrix(np.dot(np.dot(trans, rot), target))
        view_matrix = p.computeViewMatrix(t, target_position, [0, 0, 1])
        projection_matrix = p.computeProjectionMatrixFOV(HumanVisualModel.FOV,
                                                         HumanVisualModel.ASPECT,
                                                         HumanVisualModel.CLIPNEAR,
                                                         HumanVisualModel.CLIPFAR)
        camera_image = p.getCameraImage(HumanVisualModel.WIDTH,
                                        HumanVisualModel.HEIGHT,
                                        viewMatrix=view_matrix,
                                        renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                        projectionMatrix=projection_matrix)

        depth_image = camera_image[3]
        mask_image = camera_image[4]
        rgb_image = camera_image[2]

        # points = []
        # center_u = int(HumanVisualModel.WIDTH/2.)
        # center_v = int(HumanVisualModel.HEIGHT/2.)
        # center_point = np.array(pinhole_camera_model.projectPixelTo3dRay((center_u, center_v))) * depth_map[center_v, center_u]
        # viz_frame = np.zeros((HumanVisualModel.HEIGHT,HumanVisualModel.WIDTH,3), np.uint8)
        # for v in range(HumanVisualModel.HEIGHT):
        #     #v_norm = v / float(height)
        #     for u in range(HumanVisualModel.WIDTH):
        #         if u == center_u and v == center_v:
        #             continue
        #         object_id = camera_image[4][v, u]
        #         pt3d = np.array(pinhole_camera_model.projectPixelTo3dRay((u, v))) * depth_map[v, u]
        #         points.append(pt3d)
        #         d = math.sqrt(np.sum((center_point - pt3d)**2)) # Euler distance
        #         d_clipped = min(d, focus_distance)
        #         d_norm = d_clipped / focus_distance
        #         intensity = 1.0 - d_norm
        #         (r, g, b) = colorsys.hsv_to_rgb(min(1.0-intensity, 0.8333), 1.0, 1.0)
        #         viz_frame[v, u] = (b*255, g*255, r*255)
        # print(viz_frame.shape)
        # print(center_point)
        # print(len(points))

        viz_img_msg = self.bridge.cv2_to_imgmsg(np.array(depth_image).reshape(HumanVisualModel.HEIGHT,HumanVisualModel.WIDTH))
        self.perspective_publisher.publish(viz_img_msg)
        return False, detections
