import rospy
import cv2
import numpy as np
import pybullet as p
import dlib
from sensor_msgs.msg import Image, CameraInfo
from tf.transformations import euler_from_quaternion, inverse_matrix, compose_matrix, quaternion_matrix, translation_matrix, translation_from_matrix
from cv_bridge import CvBridge

class HumanVisualModel(object):
    FOV = 90.0 # human field of view
    WIDTH = 480 # image width resolution for rendering
    HEIGHT = 360  # image height resolution for rendering
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
        perspective_timer = cv2.getTickCount()
        detections = []
        rot = quaternion_matrix(q)
        trans = translation_matrix(t)
        target = translation_matrix([0.0, 0.0, 1000.0])
        target_position = translation_from_matrix(np.dot(np.dot(trans, rot), target))
        view_matrix = p.computeViewMatrix(t, target_position, [0, 0, 1])
        projection_matrix = p.computeProjectionMatrixFOV(HumanVisualModel.FOV,
                                                         HumanVisualModel.ASPECT,
                                                         HumanVisualModel.CLIPNEAR,
                                                         HumanVisualModel.CLIPFAR)
        rendering_timer = cv2.getTickCount()
        camera_image = p.getCameraImage(HumanVisualModel.WIDTH,
                                        HumanVisualModel.HEIGHT,
                                        viewMatrix=view_matrix,
                                        renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                        projectionMatrix=projection_matrix)

        rendering_fps = cv2.getTickFrequency() / (cv2.getTickCount() - rendering_timer)
        depth_image = camera_image[3]
        mask_image = camera_image[4]
        detection_timer = cv2.getTickCount()

        unique, counts = np.unique(np.array(mask_image).flatten(), return_counts=True)
        for sim_id, count in zip(unique, counts):
            if sim_id > 0:
                cv_mask = np.array(mask_image.copy())
                cv_mask[cv_mask != sim_id] = 0
                cv_mask[cv_mask == sim_id] = 255
                detection = cv2.boundingRect(cv_mask.astype(np.uint8))
                detections.append(detection)
        detection_fps = cv2.getTickFrequency() / (cv2.getTickCount() - detection_timer)
        perspective_fps = cv2.getTickFrequency() / (cv2.getTickCount() - perspective_timer)

        cv_image_array = np.array(depth_image, np.float32).reshape(HumanVisualModel.HEIGHT,HumanVisualModel.WIDTH)
        cv_image_norm = cv2.normalize(cv_image_array, cv_image_array, 0, 1, cv2.NORM_MINMAX)*255
        viz_frame = cv2.cvtColor(cv_image_norm.astype('uint8'), cv2.COLOR_GRAY2BGR)

        rendering_fps_str = "Rendering fps: {:0.1f}hz".format(rendering_fps)
        detection_fps_str = "Detection fps: {:0.1f}hz".format(detection_fps)
        perspective_fps_str = "Perspective taking fps: {:0.1f}hz".format(perspective_fps)
        cv2.putText(viz_frame, "Nb tracks: {}".format(len(detections)), (5, 25),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(viz_frame, rendering_fps_str, (5, 45),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(viz_frame, detection_fps_str, (5, 65),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        cv2.putText(viz_frame, perspective_fps_str, (5, 85),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 1)
        for detection in detections:
            x, y, w, h = detection
            viz_frame = cv2.rectangle(viz_frame, (x,y), (x+w,y+h), (0,255,0), 2)
        viz_img_msg = self.bridge.cv2_to_imgmsg(viz_frame)
        self.perspective_publisher.publish(viz_img_msg)
        return False, detections
