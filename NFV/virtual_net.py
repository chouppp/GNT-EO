from dataclasses import dataclass


@dataclass
class VirtualNetworkEvent:
    """
    A class representing an event in the virtual network request simulator.

    Attributes:
        v_net_id (int): The ID of the virtual network.
        time (float): The time of the event.
        type (int): The type of the event (1 for arrival, 0 for leave).
        id (int): The ID of the event.
    """
    id: int
    type: int
    v_net_id: int
    time: float

    def __post_init__(self):
        if self.type not in [0, 1]:
            raise ValueError("Event type must be 0 (leave) or 1 (arrival)")
        if self.v_net_id < 0:
            raise ValueError("Virtual network ID must be non-negative")
        if self.time < 0:
            raise ValueError("Event time must be non-negative")

    def __repr__(self):
        return f"VirtualNetworkEvent(v_net_id={self.v_net_id}, time={self.time}, type={self.type}, id={self.id})"

    def __str__(self):
        return self.__repr__()

    def __getitem__(self, key):
        return getattr(self, key)

    def __setitem__(self, key, value):
        setattr(self, key, value)

