import pybullet as p

class ObjectState(object):
    UNKNOWN = 0
    PLAUSIBLE = 1
    INCONSISTENT = 2


class ObjectActionState(object):
    UNKNOWN = 0
    PLACED = 1
    HELD = 2
    RELEASED = 3


class ActionEvent(object):
    PLACE = 0
    PICK = 1
    RELEASE = 2


class ConsistentPoseEstimator(object):
    def __init__(self, uwds_simulation):
        self.simulator = uwds_simulation
        self.objects_state = {}
        self.objects_action_state = {}

    def estimate(self, track):
        if track.id not in self.objects_state:
            self.objects_action_state[track.id] = ObjectActionState.UNKNOWN
            self.objects_state[track.id] = ObjectState.UNKNOWN
            self.simulator.add_shape(track)
        else:
            # evaluate consistency => is it lying on something ?
            sim_id = self.simulator.entity_id_map[track.id]
            aabb = p.getAABB(sim_id)
            aabb_min = aabb[0]
            aabb_max = aabb[1]
            if aabb_min[3] > 0.035:
                aabb_max[3] = aabb_min[3]
                aabb_min[3] -= 0.035
                if aabb_min[3] < 0:
                    aabb_min[3] = 0
                overlaping_objects = p.getOverlappingObjects(aabb_min, aabb_max)
                if len(overlaping_objects) == 0:
                    self.objects_state[track.id] = ObjectState.INCONSISTENT
                else:
                    self.objects_state[track.id] = ObjectState.PLAUSIBLE
            else:
                self.objects_state[track.id] = ObjectState.PLAUSIBLE
