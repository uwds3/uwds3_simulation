import rospy
import cv2
import numpy as np
import pybullet as p
from pyuwds3.types.detection import Detection
from .static_saliency_estimator import StaticSaliencyEstimator
from sensor_msgs.msg import CameraInfo
from tf.transformations import quaternion_matrix, translation_matrix, translation_from_matrix


class HumanVisualModel(object):
    FOV = 90.0 # human field of view
    WIDTH = 480 # image width resolution for rendering
    HEIGHT = 360  # image height resolution for rendering
    CLIPNEAR = 0.01 # clipnear
    CLIPFAR = 25 # clipfar
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


class PerspectiveEstimator(object):
    def __init__(self, uwds_simulation, use_saliency=True):
        self.simulator = uwds_simulation
        self.use_saliency = use_saliency
        if self.use_saliency is not False:
            self.saliency_estimator = StaticSaliencyEstimator()

    def estimate(self, t, q, camera_info, target_position=None, occlusion_treshold=0.01, output_viz=True):
        perspective_timer = cv2.getTickCount()
        visible_detections = []
        rot = quaternion_matrix(q)
        trans = translation_matrix(t)
        if target_position is None:
            target = translation_matrix([0.0, 0.0, 1000.0])
            target_position = translation_from_matrix(np.dot(np.dot(trans, rot), target))
        view_matrix = p.computeViewMatrix(t, target_position, [0, 0, 1])

        width = HumanVisualModel.WIDTH
        height = HumanVisualModel.HEIGHT

        projection_matrix = p.computeProjectionMatrixFOV(HumanVisualModel.FOV,
                                                         HumanVisualModel.ASPECT,
                                                         HumanVisualModel.CLIPNEAR,
                                                         HumanVisualModel.CLIPFAR)

        rendered_width = int(width/3.0)
        rendered_height = int(height/3.0)

        width_ratio = width/rendered_width
        height_ratio = height/rendered_height

        camera_image = p.getCameraImage(rendered_width,
                                        rendered_height,
                                        viewMatrix=view_matrix,
                                        renderer=p.ER_BULLET_HARDWARE_OPENGL,
                                        projectionMatrix=projection_matrix)

        rgb_image = cv2.resize(np.array(camera_image[2]), (width, height))
        depth_image = np.array(camera_image[3], np.float32).reshape((rendered_height, rendered_width))

        far = HumanVisualModel.CLIPFAR
        near = HumanVisualModel.CLIPNEAR
        real_depth_image = far * near / (far - (far - near) * depth_image)

        if self.use_saliency is not False:
            saliency_map, saliency_heatmap = self.saliency_estimator.estimate(camera_image[2], real_depth_image)
            saliency_heatmap_resized = cv2.resize(saliency_heatmap, (width, height))

        mask_image = camera_image[4]
        unique, counts = np.unique(np.array(mask_image).flatten(), return_counts=True)

        for sim_id, count in zip(unique, counts):
            if sim_id > 0:
                cv_mask = np.array(mask_image.copy())
                cv_mask[cv_mask != sim_id] = 0
                cv_mask[cv_mask == sim_id] = 255
                xmin, ymin, w, h = cv2.boundingRect(cv_mask.astype(np.uint8))

                visible_area = w*h+1
                screen_area = rendered_width*rendered_height+1
                if screen_area-visible_area == 0:
                    confidence = 1.0
                else:
                    confidence = visible_area/float(screen_area-visible_area)
                #TODO compute occlusion score as a ratio between visible 2d bbox and projected 2d bbox areas
                depth = real_depth_image[int(ymin+h/2.0)][int(xmin+w/2.0)]

                xmin = int(xmin*width_ratio)
                ymin = int(ymin*height_ratio)
                w = int(w*width_ratio)
                h = int(h*height_ratio)

                id = self.simulator.reverse_entity_id_map[sim_id]
                if confidence > occlusion_treshold:
                    det = Detection(int(xmin), int(ymin), int(xmin+w), int(ymin+h), id, confidence, depth=depth)
                    visible_detections.append(det)

        # for subject_detection in visible_detections:
        #     for object_detection in visible_detections:
        #         if subject_detection != object_detection:
        #             pass #TODO create inference batch for egocentric relation detection

        perspective_fps = cv2.getTickFrequency() / (cv2.getTickCount() - perspective_timer)

        if output_viz is True:
            viz_frame = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
            if self.use_saliency is not False:
                viz_frame = cv2.addWeighted(saliency_heatmap_resized, 0.4, viz_frame, 0.7, 0)
            cv2.rectangle(viz_frame, (0, 0), (250, 40), (200, 200, 200), -1)
            perspective_fps_str = "Perspective taking fps: {:0.1f}hz".format(perspective_fps)
            cv2.putText(viz_frame, "Nb detections: {}".format(len(visible_detections)), (5, 15),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            cv2.putText(viz_frame, perspective_fps_str, (5, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            for detection in visible_detections:
                detection.draw(viz_frame, (0, 200, 0))

            return rgb_image, real_depth_image, visible_detections, viz_frame
        else:
            return rgb_image, real_depth_image, visible_detections
