import gym
import torch
import gym_minigrid
from gym_minigrid.minigrid import *
from gym_minigrid.envs.keycorridor import *
from gym_minigrid.envs.distshift import *
import numpy as np
from abc import ABC, abstractmethod
from torch.multiprocessing import Process, Pipe
from gym_minigrid.register import register
from predicates import *


class PredicateWrapper(gym.Wrapper):

    def __init__(self, env, predicates):
        super().__init__(env)
        self.predicates = predicates
        self.obs = None

    def reset(self, **kwargs):
        self.obs = self.env.reset(**kwargs)
        return self.obs

    def step(self, action):
        next_obs, rew, done, info = self.env.step(action)
        info['predicates'] = [int(e.predict(self.obs, action, next_obs)) for e in self.predicates]
        self.obs = next_obs
        return next_obs, rew, done, info


class MiniGridRescaleObservationWrapper(gym.Wrapper):

    def __init__(self, env):
        super().__init__(env)

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)/10.

    def step(self, action):
        next_obs, rew, done, info = self.env.step(action)
        return next_obs/10., rew, done, info


class MiniGridTimeLimitWrapper(gym.Wrapper):

    def __init__(self, env, limit):
        super(MiniGridTimeLimitWrapper, self).__init__(env)
        self.env.unwrapped.max_steps = limit
        self.limit = limit
        self.steps = 0

    def reset(self, **kwargs):
        self.steps = 0
        return self.env.reset(**kwargs)

    def step(self, action):
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        next_obs, rew, done, info = self.env.step(action)
        if done and self.steps < self.limit:
            if rew < 0.01:
                done = False
            if action == self.actions.forward and fwd_cell != None and fwd_cell.type == 'lava':
                done = True
        # if done and rew < 0.01 and self.steps < self.limit:
        #     done = False
        if self.steps >= self.limit:
            done = True
        self.steps += 1
        self.env.unwrapped.step_count = self.steps
        return next_obs, rew, done, info


class MiniGridStateBonus(gym.core.Wrapper):

    def __init__(self, env):
        super().__init__(env)
        self.counts = {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)

        # Compute the hash code
        code = hash(obs.tostring())

        # Get the count for this key
        pre_count = 0
        if code in self.counts:
            pre_count = self.counts[code]

        # Update the count for this key
        new_count = pre_count + 1
        self.counts[code] = new_count

        bonus = 1 / math.sqrt(new_count)
        reward += bonus

        return obs, reward, done, info

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)


class MiniGridRewardWrapper(gym.RewardWrapper):

    def __init__(self, env):
        super(MiniGridRewardWrapper, self).__init__(env)

    def reward(self, reward):
        if reward > 0.:
            return 1.0
        elif reward == -1:
            return -1.0
        else:
            return reward - 0.9 / self.env.max_steps


class MiniGridBallFinding(RoomGrid):

    TRAIN_KEY_COLORS = ['red', 'green']
    TEST_KEY_COLORS = ['grey', 'purple', 'yellow', 'blue']

    TRAIN_BALL_COLORS = ['red', 'green']
    TEST_BALL_COLORS = ['grey', 'purple', 'yellow', 'blue']

    LEFT_DOORS_COLORS = ['red', 'green', 'blue']

    def __init__(self, train=True, seed=None):
        self.train = train
        super(MiniGridBallFinding, self).__init__(room_size=4, num_rows=2, num_cols=4, max_steps=1000, seed=seed)

    def sample_config(self):
        if self.train:
            self.key_color = self._rand_elem(self.TRAIN_KEY_COLORS)
            self.left_door_1_color = self.left_door_2_color = self._rand_elem(self.LEFT_DOORS_COLORS)
            self.ball_color = self._rand_elem(self.TRAIN_BALL_COLORS)
            self.right_door_1_color = 'yellow'
        else:
            self.key_color = self._rand_elem(self.TEST_KEY_COLORS)
            self.left_door_1_color = self._rand_elem(self.LEFT_DOORS_COLORS)
            while True:
                self.left_door_2_color = self._rand_elem(self.LEFT_DOORS_COLORS)
                if self.left_door_1_color != self.left_door_2_color:
                    break
            self.ball_color = self._rand_elem(self.TEST_BALL_COLORS)
            self.right_door_1_color = self._rand_elem(COLOR_NAMES)

    def fill_lava(self, i, j):
        room = self.get_room(i, j)

        topX, topY = room.top
        sizeX, sizeY = room.size

        for x in range(topX + 1, topX + sizeX - 1):
            for y in range(topY + 1, topY + sizeY - 1):
                self.grid.set(x, y, Lava())

    def fill_lava_above(self, i, j):
        room = self.get_room(i, j)

        topX, topY = room.top
        sizeX, sizeY = room.size

        for x in range(topX + 1, topX + sizeX - 1):
            self.grid.set(x, topY, Lava())

    def add_key(self):
        room = self.get_room(0, 0)
        topX, topY = room.top
        key = Key(self.key_color)
        self.grid.set(topX + 1, topY + 1, key)
        return key

    def add_ball(self):
        room = self.get_room(self.num_cols - 1, 0)
        topX, topY = room.top
        ball = Ball(self.ball_color)
        self.grid.set(topX + 1, topY + 1, ball)
        return ball

    def fill_lava_left(self, i, j):
        room = self.get_room(i, j)

        topX, topY = room.top
        sizeX, sizeY = room.size

        for y in range(topY + 1, topY + sizeY - 1):
            self.grid.set(topX + 1, y, Lava())

    def fill_lava_left_corner(self, i, j):
        room = self.get_room(i, j)

        topX, topY = room.top
        sizeX, sizeY = room.size

        for y in range(topY + sizeY - 2, topY + sizeY - 1):
            self.grid.set(topX + 1, y, Lava())

    def fill_lava_right(self, i, j):
        room = self.get_room(i, j)

        topX, topY = room.top
        sizeX, sizeY = room.size

        for y in range(topY + 1, topY + sizeY - 1):
            self.grid.set(topX + sizeX - 2, y, Lava())

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        # Connect the rooms vertically for all except the last column
        for i in range(0, self.num_cols - 1):
            for j in range(1, self.num_rows):
                # Removes the upper wall (3) of the room (i, j)
                self.remove_wall(i, j, 3)

        self.sample_config()

        # Add locked door and key
        self.locked_door, _ = self.add_door(3, 0, 2, locked=True, color=self.key_color)
        self.key = self.add_key()

        # Add ball
        self.ball = self.add_ball()

        # Add doors
        self.left_door_1, _ = self.add_door(1, 0, 2, locked=False, color=self.left_door_1_color)
        self.left_door_2, _ = self.add_door(1, 1, 0, locked=False, color=self.left_door_2_color)
        self.right_door_1, _ = self.add_door(3, 1, 2, locked=False, color=self.right_door_1_color)

        self.fill_lava(0, 1)
        self.fill_lava_above(0, 1)
        self.fill_lava(0, 1)
        self.fill_lava_right(1, 0)
        # self.fill_lava_left_corner(1, 1)
        self.fill_lava(3, 1)
        # self.fill_lava_left(2, 0)
        # self.fill_lava_right(3, 0)

        self.place_agent(1, 1)

    def reset(self):
        self.key_picked = False
        self.left_door_1_passed = False
        self.left_door_2_passed = False
        self.locked_door_passed = False
        self.key_dropped = False
        return super().reset()

    def step(self, action):
        terminal = False
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        agent_cell = self.grid.get(*self.agent_pos) 
        next_obs, rew, done, info = super().step(action)
        if action == self.actions.forward and fwd_cell != None and fwd_cell.type == 'lava':
            done = terminal = False
            rew = -0.1
        elif agent_cell != None and agent_cell.type == 'lava':
            done = terminal = False
            rew = -0.1
        elif action == self.actions.pickup and fwd_cell == self.key and not self.key_picked:
            rew = 1
            self.key_picked = True
        elif action == self.actions.drop and not self.locked_door_passed and self.key_picked:
            rew = -0.1
        elif action == self.actions.toggle and fwd_cell == self.left_door_2 and self.key_picked and not self.left_door_2_passed:
            rew = 1
            self.left_door_2_passed = True
        elif action == self.actions.forward and fwd_cell == self.left_door_1 and self.key_picked and not self.left_door_1_passed:
            rew = 1
            self.left_door_1_passed = True
        elif action == self.actions.toggle and fwd_cell == self.locked_door and self.key_picked and not self.locked_door_passed:
            rew = 1
            self.locked_door_passed = True
        elif action == self.actions.drop and fwd_cell == None and self.locked_door_passed and not self.key_dropped:
            rew = 1
            self.key_dropped = True
        elif action == self.actions.pickup and fwd_cell != None and fwd_cell.type == 'ball':
            done = terminal = True
            rew = 1
        # rew -= 0.0001
        info['terminal'] = terminal
        return next_obs, rew, done, info


class MiniGridBallFindingTrain(MiniGridBallFinding):
    
    def __init__(self, seed=None):
        super(MiniGridBallFindingTrain, self).__init__(train=True, seed=seed)


class MiniGridBallFindingTest(MiniGridBallFinding):

    def __init__(self, seed=None):
        super(MiniGridBallFindingTest, self).__init__(train=False, seed=seed)


register(
    id='MiniGrid-BallFindingTrain-v0',
    entry_point='envs:MiniGridBallFindingTrain'
)

register(
    id='MiniGrid-BallFindingTest-v0',
    entry_point='envs:MiniGridBallFindingTest'
)


class MiniGridKeyCorridorS3R2Manip(KeyCorridorS3R2):

    def __init__(self, key_color='blue', ball_color='red', door_color=None, seed=None):
        self.key_color = key_color
        self.ball_color = ball_color
        self.door_color = door_color
        super(MiniGridKeyCorridorS3R2Manip, self).__init__(seed=seed)

    def _gen_grid(self, width, height):
        super(MiniGridKeyCorridorS3R2Manip, self)._gen_grid(width, height)
        self.set_ball_color(self.ball_color)
        self.set_key_color(self.key_color)
        self.set_door_color(self.key_color)
        if self.door_color:
            self.set_unlocked_door_colors(self.door_color)

    def set_ball_color(self, color):
        self.obj.color = color

    def set_key_color(self, color):
        # Find the room with the key
        for i in range(self.num_cols):
            for j in range(self.num_rows):
                room = self.get_room(i, j)
                for obj in room.objs:
                    if obj.type == 'key':
                        obj.color = color

    def set_door_color(self, color):
        for j in range(0, self.height):
            for i in range(0, self.width):
                obj = self.grid.get(i, j)
                if obj is not None and obj.type == 'door' and obj.is_locked:
                    obj.color = color

    def set_unlocked_door_colors(self, color):
        for j in range(0, self.height):
            for i in range(0, self.width):
                obj = self.grid.get(i, j)
                if obj is not None and obj.type == 'door' and not obj.is_locked:
                    obj.color = color


class MiniGridKeyCorridorS3R2Manip1(MiniGridKeyCorridorS3R2Manip):
    
    def __init__(self, seed=None):
        super(MiniGridKeyCorridorS3R2Manip1, self).__init__(key_color='red', seed=seed)


class MiniGridKeyCorridorS3R2Manip2(MiniGridKeyCorridorS3R2Manip):

    def __init__(self, seed=None):
        super(MiniGridKeyCorridorS3R2Manip2, self).__init__(key_color='green', seed=seed)


class MiniGridKeyCorridorS3R2Manip3(MiniGridKeyCorridorS3R2Manip):

    def __init__(self, seed=None):
        super(MiniGridKeyCorridorS3R2Manip3, self).__init__(ball_color='blue', seed=seed)


class MiniGridKeyCorridorS3R2Manip4(MiniGridKeyCorridorS3R2Manip):

    def __init__(self, seed=None):
        super(MiniGridKeyCorridorS3R2Manip4, self).__init__(ball_color='green', seed=seed)


class MiniGridKeyCorridorS3R2Manip5(MiniGridKeyCorridorS3R2Manip):

    def __init__(self, seed=None):
        super(MiniGridKeyCorridorS3R2Manip5, self).__init__(ball_color='green', key_color='red', seed=seed)


class MiniGridKeyCorridorS3R2Manip6(MiniGridKeyCorridorS3R2Manip):

    def __init__(self, seed=None):
        super(MiniGridKeyCorridorS3R2Manip6, self).__init__(ball_color='green', key_color='green', seed=seed)


class MiniGridKeyCorridorS3R2Manip7(MiniGridKeyCorridorS3R2Manip):

    def __init__(self, seed=None):
        super(MiniGridKeyCorridorS3R2Manip7, self).__init__(ball_color='blue', key_color='red', seed=seed)


class MiniGridKeyCorridorS3R2Manip8(MiniGridKeyCorridorS3R2Manip):

    def __init__(self, seed=None):
        super(MiniGridKeyCorridorS3R2Manip8, self).__init__(ball_color='blue', key_color='green', seed=seed)


class MiniGridKeyCorridorS3R2Manip9(MiniGridKeyCorridorS3R2Manip):

    def __init__(self, seed=None):
        super(MiniGridKeyCorridorS3R2Manip9, self).__init__(door_color='yellow', seed=seed)


class MiniGridKeyCorridorS3R2Manip10(MiniGridKeyCorridorS3R2Manip):

    def __init__(self, seed=None):
        super(MiniGridKeyCorridorS3R2Manip10, self).__init__(door_color='grey', seed=seed)


register(
    id='MiniGrid-KeyCorridorS3R2M1-v0',
    entry_point='envs:MiniGridKeyCorridorS3R2Manip1'
)

register(
    id='MiniGrid-KeyCorridorS3R2M2-v0',
    entry_point='envs:MiniGridKeyCorridorS3R2Manip2'
)

register(
    id='MiniGrid-KeyCorridorS3R2M3-v0',
    entry_point='envs:MiniGridKeyCorridorS3R2Manip3'
)

register(
    id='MiniGrid-KeyCorridorS3R2M4-v0',
    entry_point='envs:MiniGridKeyCorridorS3R2Manip4'
)

register(
    id='MiniGrid-KeyCorridorS3R2M5-v0',
    entry_point='envs:MiniGridKeyCorridorS3R2Manip5'
)

register(
    id='MiniGrid-KeyCorridorS3R2M6-v0',
    entry_point='envs:MiniGridKeyCorridorS3R2Manip6'
)

register(
    id='MiniGrid-KeyCorridorS3R2M7-v0',
    entry_point='envs:MiniGridKeyCorridorS3R2Manip7'
)

register(
    id='MiniGrid-KeyCorridorS3R2M8-v0',
    entry_point='envs:MiniGridKeyCorridorS3R2Manip8'
)

register(
    id='MiniGrid-KeyCorridorS3R2M9-v0',
    entry_point='envs:MiniGridKeyCorridorS3R2Manip9'
)

register(
    id='MiniGrid-KeyCorridorS3R2M10-v0',
    entry_point='envs:MiniGridKeyCorridorS3R2Manip10'
)


class MiniGridKeyCorridorLavaS4R3(KeyCorridor):

    def __init__(self, seed=None):
        super().__init__(
            room_size=4,
            num_rows=3,
            seed=seed
        )

    def _gen_grid(self, width, height):
        super()._gen_grid(width, height)

        self.place_in_room(1, 0, Lava())
        self.place_in_room(1, self.num_rows // 2, Lava())
        self.place_in_room(1, self.num_rows - 1, Lava())

    def reset(self):
        super().reset()
        self.key_picked = False

    def step(self, action):
        fwd_pos = self.front_pos
        fwd_cell = self.grid.get(*fwd_pos)
        next_obs, rew, done, info = super().step(action)
        # if action == self.actions.forward and fwd_cell != None and fwd_cell.type == 'lava':
        #     rew = -1
        if action == self.actions.pickup and fwd_cell != None and fwd_cell.type == 'key' and not self.key_picked:
            rew = 1
            self.key_picked = True
        return next_obs, rew, done, info


register(
    id='MiniGrid-KeyCorridorLavaS4R3-v0',
    entry_point='envs:MiniGridKeyCorridorLavaS4R3'
)


class MiniGridLavaSafetyRewardWrapper(gym.Wrapper):

    def __init__(self, env, safety_radius):
        super(MiniGridLavaSafetyRewardWrapper, self).__init__(env)
        self.safety_radius = safety_radius

    def step(self, action):
        next_obs, rew, done, info = super().step(action)

        if self.safety_radius > 0 and near(next_obs, 'lava', self.safety_radius):
            rew -= 0.05

        return next_obs, rew, done, info


class MiniGridKeyCorridorRewardWrapper(gym.Wrapper):

    def __init__(self):
        super(MiniGridKeyCorridorRewardWrapper, self).__init__(env)
        self.obs = None
        self.key_pickup = False

    def reset(self, **kwargs):
        self.obs = self.env.reset(**kwargs)
        self.key_pickup = False
        return self.obs

    def step(self, action):
        next_obs, rew, done, info = self.env.step(action)

        if pickup(self.obs, action, next_obs, 'key') and not self.key_pickup:
            rew = 1.
            self.key_pickup = True

        self.obs = next_obs
        return next_obs, rew, done, info


class ImageChannelSwapWrapper(gym.ObservationWrapper):

    def __init__(self, env):
        super().__init__(env)

    def observation(self, observation):
        return observation.transpose((2, 0, 1))


class TorchObsWrapper(gym.ObservationWrapper):

    def __init__(self, env, device):
        super().__init__(env)
        self.device = device

    def observation(self, observation):
        observation = torch.from_numpy(observation).type(torch.FloatTensor)
        # Resize, and add a batch dimension (BCHW)
        return observation.unsqueeze(0).to(self.device)


def get_env_observation_shape(env):
    if type(env.observation_space) == gym.spaces.Dict:
        if 'image' in env.observation_space.spaces:
            # This is for the gym-minigrid environment
            return env.observation_space['image'].shape
        else:
            raise NotImplementedError

    elif type(env.observation_space) == gym.spaces.Box:
        return env.observation_space.shape

    else:
        raise NotImplementedError


class VecEnv(ABC):
    """
    An abstract asynchronous, vectorized environment.
    """

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a tuple of observation arrays.
        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.
        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().
        Returns (obs, rews, dones, infos):
         - obs: an array of observations, or a tuple of
                arrays of observations.
         - rews: an array of rewards
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    @abstractmethod
    def close(self):
        """
        Clean up the environments' resources.
        """
        pass

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self):
        pass

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self


class VecEnvWrapper(VecEnv):
    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        VecEnv.__init__(self,
                        num_envs=venv.num_envs,
                        observation_space=observation_space or venv.observation_space,
                        action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self):
        self.venv.render()


def manual_reseed(env, seeds):
    # Keep reseeding all the wrappers until the env is unwrapped
    # This assumes that a ReseedWrapper layer is being used
    if env != env.unwrapped:
        env.seeds = seeds
        env.seed_idx = 0
        manual_reseed(env.env, seeds)


def worker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            remote.send((ob, reward, done, info))
        elif cmd == 'reset':
            if data:
                manual_reseed(env, [data])
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_space, env.action_space))
        elif cmd == 'seed':
            remote.send(env.seeds[(env.seed_idx - 1) % len(env.seeds)])
        else:
            raise NotImplementedError


class SubprocVecEnv(VecEnv):
    def __init__(self, env_fns, spaces=None):
        """
        envs: list of gym environments to run in subprocesses
        """
        self.waiting = False
        self.closed = False
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, remote, env_fn))
                   for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()

        self.remotes[0].send(('get_spaces', None))
        observation_space, action_space = self.remotes[0].recv()
        VecEnv.__init__(self, len(env_fns), observation_space, action_space)

    def step_async(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self, seeds=None):
        for i, remote in enumerate(self.remotes):
            if seeds is not None:
                remote.send(('reset', seeds[i]))
            else:
                remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def reset_subset(self, indices, seeds=None):
        subset_remotes = [self.remotes[i] for i in indices]
        for i, remote in enumerate(subset_remotes):
            if seeds is not None:
                remote.send(('reset', seeds[i]))
            else:
                remote.send(('reset', None))
        return np.stack([remote.recv() for remote in subset_remotes])

    def get_seeds(self, indices=None):
        if indices is not None:
            subset_remotes = [self.remotes[i] for i in indices]
        else:
            subset_remotes = self.remotes
        for remote in subset_remotes:
            remote.send(('seed', None))
        return np.stack([remote.recv() for remote in subset_remotes])

    def reset_task(self):
        for remote in self.remotes:
            remote.send(('reset_task', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        if self.closed:
            return
        if self.waiting:
            for remote in self.remotes:
                remote.recv()
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()
        self.closed = True

# class MiniGridMaze(gym_minigrid.envs.MiniGridEnv):
#
#     def __init__(self):

if __name__ == '__main__':
    import gym
    import gym_minigrid
    from gym.wrappers import *
    from gym_minigrid.wrappers import *
    import matplotlib.pyplot as plt
    import time
    from predicates import *

    env = gym.make('MiniGrid-KeyCorridorS3R2-v0')
    for wrapper in ['FullyObsWrapper', 'ImageChannelSwapWrapper']:
        env = globals()[wrapper](env)
    env = ReseedWrapper(env, [1])
    env = PredicateWrapper(env, [FullyObservableMiniGridNextTo('door'), FullyObservableMiniGridNextTo('ball'),
                                 FullyObservableMiniGridNextTo('key'), FullyObservableMiniGridAttemptOpenDoor(),
                                 FullyObservableMiniGridFacing('door', 1)])
    env = MiniGridTimeLimitWrapper(env, 272)
    env = MiniGridRewardWrapper(env)
    n_envs = 1

    env.reset()
    # print(env.step(2)[1:])
    # print(env.step(2)[1:])
    # print(env.step(1)[1:])
    # print(env.step(5)[1:])
    # plt.imshow(env.render(mode='rgb_array'))
    # plt.show()

    envs = SubprocVecEnv([lambda: env] * n_envs)

    start = time.time()
    envs.reset()
    arr1 = []
    arr2 = []
    for i in range(500):
        t1 = time.time()
        envs.step_async(np.random.randint(0, 3, 1))
        envs.step_async(np.random.randint(0, 3, 1))
        envs.step_async(np.random.randint(0, 3, 1))
        envs.step_async(np.random.randint(0, 3, 1))
        envs.step_async(np.random.randint(0, 3, 1))
        envs.step_async(np.random.randint(0, 3, 1))
        envs.step_async(np.random.randint(0, 3, 1))
        envs.step_async(np.random.randint(0, 3, 1))
        envs.step_async(np.random.randint(0, 3, 1))
        envs.step_async(np.random.randint(0, 3, 1))
        envs.step_async(np.random.randint(0, 3, 1))
        envs.step_async(np.random.randint(0, 3, 1))
        t2 = time.time()
        time.sleep(0.004)
        arr1.append(t2 - t1)
        envs.step_wait()
        envs.step_wait()
        envs.step_wait()
        envs.step_wait()
        envs.step_wait()
        envs.step_wait()
        envs.step_wait()
        envs.step_wait()
        envs.step_wait()
        envs.step_wait()
        envs.step_wait()
        envs.step_wait()
        t3 = time.time()
        arr2.append(t3 - t2)
    end = time.time()
    print(end - start)
    print (np.mean(arr1))
    print(np.mean(arr2))

    # for _ in range(1):
    #     s = env.reset()
    #     # plt.imshow(env.render(mode='rgb_array'))
    #     # plt.show()
    #     for i in range(502):
    #         obs, rew, done, info = env.step(np.random.randint(0, 3))
    #         print (i, rew, info, done, obs)
    #         if done:
    #             break

