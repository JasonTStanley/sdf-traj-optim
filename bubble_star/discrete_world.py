import numpy as np
from attrs import define

from astar import AStar
from sdf import SDF


@define
class BubbleStarConfig:
    resolution: float


class Node:
    x: int
    y: int
    from_bubble_rad: float

    def __init__(self, xy: tuple, from_bubble_rad: float):
        self.x = xy[0]
        self.y = xy[1]
        self.from_bubble_rad = from_bubble_rad

    def __hash__(self):
        return hash((self.x, self.y))

    def __eq__(self, other):
        return self.x == other.x and self.y == other.y


def circle_edge_cells(cx, cy, r):
    cells = set()

    x = r
    y = 0
    d = 1 - r

    def add_symmetry(cx, cy, x, y):
        cells.update(
            {
                Node((cx + x, cy + y), r),
                Node((cx - x, cy + y), r),
                Node((cx + x, cy - y), r),
                Node((cx - x, cy - y), r),
                Node((cx + y, cy + x), r),
                Node((cx - y, cy + x), r),
                Node((cx + y, cy - x), r),
                Node((cx - y, cy - x), r),
            }
        )

    while x >= y:
        add_symmetry(cx, cy, x, y)
        y += 1
        if d < 0:
            d += 2 * y + 1
        else:
            x -= 1
            d += 2 * (y - x) + 1

    return list(cells)


class BubbleStar(AStar):
    sdf: SDF
    config: BubbleStarConfig

    def node_to_pos(self, node: Node):
        return self.config.resolution * np.array([node.x, node.y])

    def is_goal_reached(self, current: Node, goal: Node) -> bool:
        return current == goal

    def sdf_at_node(self, node: Node):
        pos = self.config.resolution * np.array([node.x, node.y])
        rad = self.sdf.query(pos)
        return rad // self.config.resolution

    def heuristic_cost_estimate(self, n1: Node, n2: Node):
        """computes the 'direct' distance between two (x,y) tuples"""
        return np.sqrt((n1.x - n2.x) ** 2 + (n1.y - n2.y) ** 2)

    def distance_between(self, n1, n2):
        """this method always returns 1, as two 'neighbors' are always adjacent"""
        return n2.from_bubble_rad

    def neighbors(self, node):
        # get the neighbors to a cell.
        # only need to consider the neighbors at the edge of the bubble
        # we should be able to also clear the neighbors if the bubble we sample includes them.
        return circle_edge_cells(node.x, node.y, self.sdf_at_node(node))


def runBubbleStar(
    sdf: SDF, start: np.ndarray, goal: np.ndarray, config: BubbleStarConfig
):
    # discretize space
    bounds = sdf.bounds
    x_cells = int((bounds.x_max - bounds.x_min) // config.resolution)
    # xs = np.linspace(bounds.x_min, bounds.x_max, num=x_cells)
    y_cells = int((bounds.y_max - bounds.y_min) // config.resolution)
    # ys = np.linspace(bounds.y_min, bounds.y_max, num=y_cells)

    def to_idx(pos):
        return pos[0] // config.resolution, pos[1] // config.resolution

    final_node = Node(to_idx(goal))
    start_node = Node(to_idx(start))
    cost_map = np.inf * np.ones((x_cells, y_cells))
    closed = []
    open = []
    final_in_closed = False
    while not final_in_closed:

        pass

    # start by sampling at start to initialize the algorithm
    b1 = sdf.query(start)
    bend = sdf.query(goal)
    pass
