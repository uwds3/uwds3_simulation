
class ObjectState(object):
    PLAUSIBLE = 0
    INCONSISTENT = 1


class PhysicsMonitor(object):
    def __init__(self, uwds_simulation):
        self.simulator = uwds_simulation

    def update(self, tracks):
        for track in tracks:
            pass #TODO
