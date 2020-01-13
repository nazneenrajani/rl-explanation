import numpy as np
from abc import ABC
import gym_minigrid

MINIGRID_OBJECT_TO_IDX = gym_minigrid.wrappers.OBJECT_TO_IDX
MINIGRID_COLOR_TO_IDX = gym_minigrid.wrappers.COLOR_TO_IDX
MINIGRID_ACTION_TO_IDX = {'left': 0,
                          'right': 1,
                          'forward': 2,
                          'pickup': 3,
                          'drop': 4,
                          'toggle': 5,
                          'done': 6}

# up, right, down, left
DIRECTIONAL_ADJUSTMENT = np.array([[[-1], [0]],
                                   [[0], [-1]],
                                   [[1], [0]],
                                   [[0], [1]]])


class Predicate(ABC):

    def __init__(self):
        pass

    def name(self):
        return type(self).__name__

    @staticmethod
    def predict(s, a, sp):
        pass

    def describe_on(self):
        pass

    def describe_off(self):
        pass


class MiniGridHitLava(Predicate):

    def __init__(self):
        super().__init__()

    # Check if the agent hit the lava
    @staticmethod
    def hit_lava(s, a):
        # Check if the lava is in front of the agent and the agent moved forward
        if s[(s.shape[0] - 1) // 2, -2, 0] == 9 and a == 2:
            return True
        return False

    @staticmethod
    def predict(s, a, sp):
        return MiniGridHitLava.hit_lava(s, a)

    @staticmethod
    def describe_on():
        return "hit lava"

    @staticmethod
    def describe_off():
        return "didn't hit lava"


class MiniGridLavaInFront(Predicate):

    def __init__(self):
        super().__init__()

    # Check if the agent hit the lava
    @staticmethod
    def lava_in_front(s):
        # Check if the lava is in front of the agent
        if s[(s.shape[0] - 1) // 2, -2, 0] == 9:
            return True
        return False

    @staticmethod
    def predict(s, a, sp):
        return MiniGridLavaInFront.lava_in_front(s)

    @staticmethod
    def describe_on():
        return "in front of lava"

    @staticmethod
    def describe_off():
        return "not in front of lava"


class FullyObservableMiniGridNextToWall(Predicate):

    # up, down, left, right
    DIRECTIONAL_ADJUSTMENT = np.array([[[-1], [0]], [[1], [0]], [[0], [1]], [[0], [-1]]])

    def __init__(self):
        super().__init__()

    # Check if the agent is walking next to the wall
    @staticmethod
    def next_to_wall(s):
        # Squeeze out extra dimensions
        s = s.cpu().squeeze()
        # Make sure we've got the channels in front
        assert(s.shape[0] == 3)
        # Check if the wall is next to the agent
        locs = np.array(np.where(s[0] == 10)) + FullyObservableMiniGridNextToWall.DIRECTIONAL_ADJUSTMENT
        if np.any((s[0][locs[:, 0], locs[:, 1]] == 2).cpu().numpy()):
            return True
        return False

    @staticmethod
    def predict(s, a, sp):
        return FullyObservableMiniGridNextToWall.next_to_wall(s)

    @staticmethod
    def describe_on():
        return "next to wall"

    @staticmethod
    def describe_off():
        return "away from wall"


class FullyObservableMiniGridAwayFromWall(Predicate):

    # up, down, left, right
    DIRECTIONAL_ADJUSTMENT = np.array([[[-1], [0]], [[1], [0]], [[0], [1]], [[0], [-1]]])

    def __init__(self):
        super().__init__()

    # Check if the agent is walking next to the wall
    @staticmethod
    def away_from_wall(s):
        # Squeeze out extra dimensions
        s = s.cpu().squeeze()
        # Make sure we've got the channels in front
        assert(s.shape[0] == 3)
        # Check if the wall is next to the agent
        locs = np.array(np.where(s[0] == 10)) + FullyObservableMiniGridAwayFromWall.DIRECTIONAL_ADJUSTMENT
        if np.all(1 - (s[0][locs[:, 0], locs[:, 1]] == 2).cpu().numpy()):
            return True
        return False

    @staticmethod
    def predict(s, a, sp):
        return FullyObservableMiniGridAwayFromWall.away_from_wall(s)

    @staticmethod
    def describe_on():
        return "away from wall"

    @staticmethod
    def describe_off():
        return "next to wall"


class FullyObservableMiniGrid5x5v0AtSquare(Predicate):

    def __init__(self, i, j):
        self.i = i
        self.j = j
        super().__init__()

    @staticmethod
    def at_square_ij(s, i, j):
        # Squeeze out extra dimensions
        s = s.cpu().squeeze()
        # Make sure we've got the channels in front
        assert (s.shape[0] == 3)
        # Check if the agent is at the right square
        agent_loc = np.where(s[0] == 10)
        if agent_loc == (i, j):
            return True
        return False

    def predict(self, s, a, sp):
        return FullyObservableMiniGrid5x5v0AtSquare.at_square_ij(sp, self.i, self.j)

    def describe(self):
        return f"at square {(self.j, self.i)}"

    def name(self):
        return f"FullyObservableMiniGrid5x5v0AtSquare({self.j},{self.i})"


class FullyObservableMiniGridHitWall(Predicate):

    def __init__(self):
        super().__init__()

    # Check if the agent is walking next to the wall
    @staticmethod
    def next_to_wall(s):
        # Check if the lava is in front of the agent
        if s[(s.shape[0] - 1) // 2, -2, 0] == 9:
            return True
        return False

    @staticmethod
    def predict(s, a, sp):
        return FullyObservableMiniGridHitWall.lava_in_front(s)

    @staticmethod
    def describe():
        return "lava in front"


class Constant(Predicate):

    def __init__(self):
        super(Constant, self).__init__()

    @staticmethod
    def predict(s, a, sp):
        return 1.0


class FullyObservableMiniGridFacing(Predicate):
    
    def __init__(self, object, radius):
        super(FullyObservableMiniGridFacing, self).__init__()
        self.object = object
        self.radius = radius

    def predict(self, s, a, sp):
        # Moving to a state where agent is facing object
        return facing(sp, self.object, self.radius)

    def name(self):
        if self.radius == 1:
            return f"Facing{self.object.capitalize()}"
        else:
            return f"{self.object.capitalize()}Ahead"


class FullyObservableMiniGridNear(Predicate):

    def __init__(self, object, radius):
        super(FullyObservableMiniGridNear, self).__init__()
        self.object = object
        self.radius = radius

    def predict(self, s, a, sp):
        # Moving to a state where the object is within radius
        return near(sp, self.object, self.radius)

    def name(self):
        return f"{self.object.capitalize()}WithinRadius{int(self.radius)}"


class FullyObservableMiniGridAt(Predicate):

    def __init__(self, object):
        super(FullyObservableMiniGridAt, self).__init__()
        self.object = object

    def predict(self, s, a, sp):
        # Moving to a state where the agent is at object
        return at(sp, self.object)

    def name(self):
        return f"At{self.object.capitalize()}"


class FullyObservableMiniGridNextTo(Predicate):

    def __init__(self, object):
        super(FullyObservableMiniGridNextTo, self).__init__()
        self.object = object

    def predict(self, s, a, sp):
        # Moving to a state where the agent is next to object
        return next_to(sp, self.object)

    def name(self):
        return f"NextTo{self.object.capitalize()}"


class FullyObservableMiniGridAttemptPickUp(Predicate):

    def __init__(self, object):
        super(FullyObservableMiniGridAttemptPickUp, self).__init__()
        self.object = object

    def predict(self, s, a, sp):
        # Agent attempts to pick up the object
        return pickup_attempt(s, a, self.object)

    def name(self):
        return f"AttemptPickUp{self.object.capitalize()}"


class FullyObservableMiniGridPickUp(Predicate):

    def __init__(self, object):
        super(FullyObservableMiniGridPickUp, self).__init__()
        self.object = object

    def predict(self, s, a, sp):
        # Agent attempts to pick up the object
        return pickup(s, a, sp, self.object)

    def name(self):
        return f"PickUp{self.object.capitalize()}"


class FullyObservableMiniGridOpenDoor(Predicate):

    def __init__(self):
        super(FullyObservableMiniGridOpenDoor, self).__init__()

    def predict(self, s, a, sp):
        return open_door(s, a, sp)

    def name(self):
        return f"OpenDoor"


class FullyObservableMiniGridAttemptOpenDoor(Predicate):

    def __init__(self):
        super(FullyObservableMiniGridAttemptOpenDoor, self).__init__()

    def predict(self, s, a, sp):
        return open_door_attempt(s, a)

    def name(self):
        return f"AttemptOpenDoor"


def near(s, object, radius):
    # Clean up the state to make it ready for use
    s = sanitize_state(s)
    # Find the agent's location
    agent_loc = np.array(np.where(s[0] == 10)).flatten()

    # Slice out the part of the state near the agent
    substate = s[0, max(agent_loc[0]-radius, 0):agent_loc[0]+radius+1, max(agent_loc[1]-radius, 0):agent_loc[1]+radius+1]

    # Look for the object in the substate
    object_idx = MINIGRID_OBJECT_TO_IDX[object]
    if np.any(substate == object_idx):
        return True
    return False


def at(s, object):
    return near(s, object, 0)


def facing(s, object, distance):
    assert(distance > 0)
    # Clean up the state to make it ready for use
    s = sanitize_state(s)
    # Find the agent's location and orientation
    agent_loc = np.array(np.where(s[0] == 10)).flatten()
    orientation = s[2, agent_loc[0], agent_loc[1]]

    if orientation == 0:
        # 0 = right in rendering but 0 = down in representation
        substate = s[0, agent_loc[0] + 1:agent_loc[0] + distance + 1, agent_loc[1]]
    elif orientation == 1:
        # 1 = down in rendering but 1 = right in representation
        substate = s[0, agent_loc[0], agent_loc[1] + 1:agent_loc[1] + distance + 1]
    elif orientation == 2:
        # 2 = left in rendering but 2 = up in representation
        substate = s[0, max(agent_loc[0] - distance, 0):agent_loc[0], agent_loc[1]]
    elif orientation == 3:
        # 3 = up in rendering but 3 = left in representation
        substate = s[0, agent_loc[0], max(agent_loc[1] - distance, 0):agent_loc[1]]

    # Look for the object in the substate
    object_idx = MINIGRID_OBJECT_TO_IDX[object]
    if np.any(substate == object_idx):
        return True
    return False


def next_to(s, object):
    # Clean up the state to make it ready for use
    s = sanitize_state(s)
    # Find the agent's location
    agent_loc = np.array(np.where(s[0] == 10))

    # Check if the object is next to the agent
    object_idx = MINIGRID_OBJECT_TO_IDX[object]
    possible_locs = agent_loc + DIRECTIONAL_ADJUSTMENT
    if np.any(s[0][possible_locs[:, 0], possible_locs[:, 1]] == object_idx):
        return True
    return False


def carrying(s, object):
    pass


def pickup_attempt(s, a, object):
    # If you're facing an object and do the pickup action
    if facing(s, object, 1) and a == 3:
        return True
    return False


def pickup(s, a, sp, object):
    # If you're facing an object, do the pickup action and don't see the object anymore
    if facing(s, object, 1) and a == 3 and not facing(sp, object, 1):
        return True
    return False


# TODO: this is not working because facing(sp, door, 1) returns True
def open_door(s, a, sp):
    # You're facing the door, toggle it and it opens
    if facing(s, 'door', 1) and a == 5 and not facing(sp, 'door', 1):
        return True
    return False


def open_door_attempt(s, a):
    # You're facing the door, toggle it
    if facing(s, 'door', 1) and a == 5:
        return True
    return False


def took_action(a, action):
    if MINIGRID_ACTION_TO_IDX[action] == a:
        return True
    return False


def sanitize_state(s):
    # Squeeze out extra dimensions
    try:
        s = s.cpu().squeeze().numpy()
    except AttributeError:
        pass
    # Make sure we've got the channels in front
    assert (s.shape[0] == 3)
    return s


if __name__ == '__main__':
    import gym
    import gym_minigrid
    from envs import *
    from gym_minigrid.wrappers import *
    import matplotlib.pyplot as plt
    env = gym.make('MiniGrid-KeyCorridorS3R2-v0')
    for wrapper in ['FullyObsWrapper', 'ImageChannelSwapWrapper', 'ReseedWrapper']:
        env = globals()[wrapper](env)
    env = TorchObsWrapper(env, "cpu")
    env.seed(1)

    s = env.reset()
    # plt.imshow(env.render(mode='rgb_array'))
    # plt.show()
    s, r, t, _ = env.step(1)
    print (r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(2)
    print (r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(1)
    print (r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(5)
    print (r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(2)
    print (r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(2)
    print (r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(3)
    print (r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(1)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(1)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(2)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(2)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(0)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(2)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(2)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(1)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(5)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(2)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(2)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()

    s, r, t, _ = env.step(1)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(1)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(4)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(1)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(1)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()
    s, r, t, _ = env.step(3)
    print(r)
    plt.imshow(env.render(mode='rgb_array'))
    plt.show()

    print ("Facing down")
    print(facing(s, 'wall', 1))
    print(facing(s, 'wall', 2))
    s, _, _, _ = env.step(2)
    print(facing(s, 'wall', 1))
    print(facing(s, 'wall', 2))

    s, _, _, _ = env.step(1)
    print("Facing left")
    print(facing(s, 'wall', 1))
    print(facing(s, 'wall', 2))
    print(facing(s, 'door', 1))
    print(facing(s, 'door', 2))
    print(facing(s, 'key', 1))
    print(facing(s, 'key', 2))
    s, _, _, _ = env.step(1)
    print("Facing up")
    print(facing(s, 'wall', 1))
    print(facing(s, 'wall', 2))
    s, _, _, _ = env.step(1)
    print("Facing right")
    print(facing(s, 'wall', 1))
    print(facing(s, 'wall', 2))
    s, _, _, _ = env.step(2)
    for i in range(6):
        for j in list(MINIGRID_OBJECT_TO_IDX.keys()):
            if near(s, j, i):
                print (f'Object {j} in radius {i}')
            if next_to(s, j):
                print (f'Object {j} next to agent')

    # env.step(1)
    # env.step(2)
    # plt.imshow(env.render(mode='rgb_array'))
    # plt.show()
    # s = env.reset()
    # print(s)
    # print(s.shape)
    # plt.imshow(env.render(mode='rgb_array'))
    # plt.show()



