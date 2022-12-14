import numpy as np
from simglucose.controller.base import Controller, Action
from stable_baselines import PPO2


class PPOController(Controller):
    def __init__(self, init_state, model_path, padding=32):
        self.init_state = init_state
        self.state = init_state
        self.model = PPO2.load(model_path)
        self.padded_obs = np.zeros((padding, 1))

    def policy(self, observation, reward, done, **info):
        self.padded_obs[0] = observation[0]
        self.state = observation
        action, _ = self.model.predict(self.padded_obs, deterministic=True)
        return Action(basal=action[0], bolus=0)

    def reset(self):
        '''
        Reset the controller state to inital state, must be implemented
        '''
        self.state = self.init_state
