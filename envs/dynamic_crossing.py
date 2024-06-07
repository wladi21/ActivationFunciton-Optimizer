import numpy as np

from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.grid import Grid
from minigrid.core.world_object import Door, Goal, Lava, Wall, Ball, Key
from minigrid.minigrid_env import MiniGridEnv
from minigrid.core.mission import MissionSpace
from minigrid.core.world_object import Door, Goal, Lava, Wall, Ball, Key

import itertools as itt

from typing import Any

import os


class DynamicCrossingEnv(MiniGridEnv):
    """Dynamic version of crossing env with configurable distractors and topology changes

    TODO: Explain usage
    """

    def __init__(
        self,
        size=11,
        num_crossings=1,
        obstacle_type=Wall,
        max_steps=None,
        num_dists=3,
        change_type="Walls",
        seed=1,
        **kwargs,
    ):
        self.num_crossings = num_crossings
        self.obstacle_type = obstacle_type
        self.num_dists = num_dists
        self.change_type = change_type
        self.seed = seed

        assert change_type in ["Walls", "Distractions", "Crossings", "Size"]

        mission_space = MissionSpace(mission_func=self._gen_mission)

        if max_steps is None:
            max_steps = 4 * size**2

        super().__init__(
            mission_space=mission_space,
            grid_size=size,
            see_through_walls=False,  # Set this to True for maximum speed
            max_steps=max_steps,
            **kwargs,
        )

    @staticmethod
    def _gen_mission():
        return "find the opening and get to the green goal square"

    def _gen_grid(self, width, height):
        assert width % 2 == 1 and height % 2 == 1  # odd size

        # Create an empty grid
        self.grid = Grid(width, height)

        # Generate the surrounding walls
        self.grid.wall_rect(0, 0, width, height)

        # Place the agent in the top-left corner
        self.agent_pos = np.array((1, 1))
        self.agent_dir = 0

        # Place a goal square in the bottom-right corner
        self.put_obj(Goal(), width - 2, height - 2)

        # Place obstacles (lava or walls)
        v, h = object(), object()  # singleton `vertical` and `horizontal` objects

        # Lava rivers or walls specified by direction and position in grid
        rivers = [(v, i) for i in range(2, height - 2, 2)]
        rivers += [(h, j) for j in range(2, width - 2, 2)]
        self.np_random.shuffle(rivers)
        rivers = rivers[: self.num_crossings]  # sample random rivers
        rivers_v = sorted(pos for direction, pos in rivers if direction is v)
        rivers_h = sorted(pos for direction, pos in rivers if direction is h)
        obstacle_pos = itt.chain(
            itt.product(range(1, width - 1), rivers_h),
            itt.product(rivers_v, range(1, height - 1)),
        )
        for i, j in obstacle_pos:
            self.put_obj(self.obstacle_type(), i, j)

        # Sample path to goal
        path = [h] * len(rivers_v) + [v] * len(rivers_h)
        self.np_random.shuffle(path)

        # Create openings
        limits_v = [0] + rivers_v + [height - 1]
        limits_h = [0] + rivers_h + [width - 1]
        room_i, room_j = 0, 0
        for direction in path:
            if direction is h:
                i = limits_v[room_i + 1]
                j = self.np_random.choice(
                    range(limits_h[room_j] + 1, limits_h[room_j + 1])
                )
                room_i += 1
            elif direction is v:
                i = self.np_random.choice(
                    range(limits_v[room_i] + 1, limits_v[room_i + 1])
                )
                j = limits_h[room_j + 1]
                room_j += 1
            else:
                assert False
            self.grid.set(i, j, None)

        # Add distractors if needed -- TODO: Scale to other types of distractors
        for x in range(self.num_dists):
            flag = True
            while flag:
                pos_x = np.random.randint(1, width - 1)
                pos_y = np.random.randint(1, height - 1)

                if self.grid.get(pos_x, pos_y) is None and (
                    pos_x != self.agent_pos[0] and pos_y != self.agent_pos[1]
                ):
                    print(pos_x, pos_y)
                    print(self.agent_pos)
                    import pdb
                    pdb.set_trace()
                    self.add_ball(pos_x, pos_y)
                    flag = False

        self.mission = (
            "avoid the lava and get to the green goal square"
            if self.obstacle_type == Lava
            else "find the opening and get to the green goal square"
        )

    def add_ball(self, x, y, color="grey"):
        self.put_obj(Ball(color), x, y)

    def add_key(self, x, y, color="grey"):
        self.put_obj(Key(color), x, y)
