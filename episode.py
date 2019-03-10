""" Contains the Episodes for Navigation. """
import random
import torch
import time
import sys
from constants import GOAL_SUCCESS_REWARD, STEP_PENALTY, BASIC_ACTIONS
from environment import Environment
from utils.net_util import gpuify


class Episode:
    """ Episode for Navigation. """
    def __init__(self, args, gpu_id, rank, strict_done=False):
        super(Episode, self).__init__()

        self._env = None

        self.gpu_id = gpu_id
        self.strict_done = strict_done
        self.task_data = None
        self.glove_embedding = None

        self.seed = args.seed + rank
        random.seed(self.seed)

        with open('./datasets/objects/int_objects.txt') as f:
            int_objects = [s.strip() for s in f.readlines()]
        with open('./datasets/objects/rec_objects.txt') as f:
            rec_objects = [s.strip() for s in f.readlines()]
        
        self.objects = int_objects + rec_objects

        self.actions_list = [{'action':a} for a in BASIC_ACTIONS]
        self.actions_taken = []

    @property
    def environment(self):
        return self._env

    def state_for_agent(self):
        return self.environment.current_frame

    def step(self, action_as_int):
        action = self.actions_list[action_as_int]
        self.actions_taken.append(action)
        return self.action_step(action)

    def action_step(self, action):
        self.environment.step(action)
        reward, terminal, action_was_successful = self.judge(action)

        return reward, terminal, action_was_successful

    def slow_replay(self, delay=0.2):
        # Reset the episode
        self._env.reset(self.cur_scene, change_seed = False)
        
        for action in self.actions_taken:
            self.action_step(action)
            time.sleep(delay)
    
    def judge(self, action):
        """ Judge the last event. """
        # immediate reward
        reward = STEP_PENALTY 
        done = False
        action_was_successful = self.environment.last_action_success

        if action['action'] == 'Done':
            #done = True
            objects = self._env.last_event.metadata['objects']
            
            for o in objects:
                if not self.success1 and o['visible'] and o['pickupable'] and o['objectType'] == self.target1:
                    
                    try: 
                        self._env.pickup_object(o['objectId']) 
                        self.target1 = o['objectId']

                        reward += GOAL_SUCCESS_REWARD
                        self.success1 = True
                        
                    except Exception:
                        print('Pickup err')
                    
                if self.success1 and not self.success2 and o['visible'] and o['openable'] and o['objectType'] == self.target2:
                    
                    try:
                        self._env.put_in_receptacle(self.target1, o['objectId'])
                        reward += GOAL_SUCCESS_REWARD
                        self.success2 = True
                    except Exception:
                        print('Something went wrong with opening microwave, putting tomato in, or closing the microwave')
                        
            
            done = self.success1 and self.success2
            if done:
                reward += GOAL_SUCCESS_REWARD

        return reward, done, action_was_successful

    def new_episode(self, args, scene):
        
        if self._env is None:
            if args.arch == 'osx':
                local_executable_path = './datasets/builds/thor-local-OSXIntel64.app/Contents/MacOS/thor-local-OSXIntel64'
            else:
                local_executable_path = './datasets/builds/thor-local-Linux64'
            
            self._env = Environment(
                    grid_size=args.grid_size,
                    fov=args.fov,
                    local_executable_path=local_executable_path,
                    randomize_objects=args.randomize_objects,
                    seed=self.seed)
            self._env.start(scene, self.gpu_id)
        else:
            self._env.reset(scene)

        # For now, single target.
        self.target1 = 'Tomato'
        self.target2 = 'Microwave'
        self.success1 = False
        self.success2 = False
        self.cur_scene = scene
        self.actions_taken = []
        
        return True
