import yt.wrapper as yt

import coordinator

class JobClient:
    def __init__(self, coordinator: coordinator.Coordinator):
        self.coordinator = coordinator