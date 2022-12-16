import random

from simglucose.simulation.env import T1DSimEnv as _T1DSimEnv
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.controller.base import Action
import pandas as pd
import numpy as np
import pkg_resources
import gym
from gym import error, spaces, utils
from gym.utils import seeding
from datetime import datetime

PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')

patient_params = pd.read_csv(PATIENT_PARA_FILE)
patient_names = patient_params['Name'].values


class T1DSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            patient_name = random.choice(patient_names)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()

    def step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        return self.env.step(act, reward_fun=self.reward_fun)

    def reset(self):
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        return obs

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2 ** 31
        seed3 = seeding.hash_seed(seed2 + 1) % 2 ** 31
        seed4 = seeding.hash_seed(seed3 + 1) % 2 ** 31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=2, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,))


class T1DSimDiffEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            patient_name = random.choice(patient_names)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()
        self.prev_cgm = None

    def step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=self.action_list[action], bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        observation = self.update_to_full_obs(observation)
        return observation, reward, done, info

    def reset(self):
        self.prev_cgm = None
        self.env, _, _, _ = self._create_env_from_random_state()
        par_obs, _, _, _ = self.env.reset()
        return self.update_to_full_obs(par_obs)

    def update_to_full_obs(self, partial_obs):
        diff = self.calculate_cgm_diff(partial_obs[0])
        self.prev_cgm = partial_obs[0]
        return [partial_obs[0], diff]

    def calculate_cgm_diff(self, current_cgm):
        if self.prev_cgm is None:
            self.prev_cgm = current_cgm
        diff = current_cgm - self.prev_cgm

        return diff

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2 ** 31
        seed3 = seeding.hash_seed(seed2 + 1) % 2 ** 31
        seed4 = seeding.hash_seed(seed3 + 1) % 2 ** 31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(2,))


class T1DSimHistoryEnv(gym.Env):
    '''
        A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
        '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None, number_of_last_obs=15):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            patient_name = random.choice(patient_names)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()
        self.number_of_last_obs = number_of_last_obs
        self.last_n_observations = np.ones([self.number_of_last_obs, ])

    def step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        observation = self.update_obs_history(observation[0])
        if done:
            reward = -1000
        return observation, reward, done, info

    def update_obs_history(self, observation):
        self.last_n_observations = np.roll(self.last_n_observations, -1)
        self.last_n_observations[-1] = observation
        return self.last_n_observations

    def reset(self):
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        self.last_n_observations[:] = obs
        return self.last_n_observations

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2 ** 31
        seed3 = seeding.hash_seed(seed2 + 1) % 2 ** 31
        seed4 = seeding.hash_seed(seed3 + 1) % 2 ** 31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(self.number_of_last_obs,))


class T1DDiscreteSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            patient_name = random.choice(patient_names)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()

    def step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)

        return [np.int(observation[0])], reward, done, info

    def reset(self):
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        return [np.int(obs[0])]

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2 ** 31
        seed3 = seeding.hash_seed(seed2 + 1) % 2 ** 31
        seed4 = seeding.hash_seed(seed3 + 1) % 2 ** 31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.int)


class T1DAdultSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            adult_patients = [p for p in patient_names if "adult" in p]
            patient_name = random.choice(adult_patients)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()

    def step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        # if done:
        #     reward = -10  # -10000
        return observation, reward, done, info

    def reset(self):
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        return obs

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2 ** 31
        seed3 = seeding.hash_seed(seed2 + 1) % 2 ** 31
        seed4 = seeding.hash_seed(seed3 + 1) % 2 ** 31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,))


class T1DAdolescentSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # todo: maybe change what patients we run
            # patient_name = 'adolescent#001'
            adolescent_patients = [p for p in patient_names if "adolescent" in p]
            patient_name = random.choice(adolescent_patients)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()

    def step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        # if done:
        #     reward = -10  # -10000
        return observation, reward, done, info

    def reset(self):
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        return obs

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2 ** 31
        seed3 = seeding.hash_seed(seed2 + 1) % 2 ** 31
        seed4 = seeding.hash_seed(seed3 + 1) % 2 ** 31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,))


class T1DChildSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # todo: maybe change what patients we run
            # patient_name = 'adolescent#001'
            child_patients = [p for p in patient_names if "child" in p]
            patient_name = random.choice(child_patients)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()

    def step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        # if done:
        #     reward = -10  # -10000
        return observation, reward, done, info

    def reset(self):
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        return obs

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2 ** 31
        seed3 = seeding.hash_seed(seed2 + 1) % 2 ** 31
        seed4 = seeding.hash_seed(seed3 + 1) % 2 ** 31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,))


class T1DAdultSimV2Env(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None, repeat_steps=4):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            adult_patients = [p for p in patient_names if "adult" in p]
            patient_name = random.choice(adult_patients)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.repeat_steps = repeat_steps
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()

    def step(self, action):
        # This gym only controls basal insulin
        observation = 0
        reward = 0
        done = False
        info = None
        act = Action(basal=action, bolus=0)
        for i in range(self.repeat_steps):
            if self.reward_fun is None:
                return self.env.step(act)
            observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
            if done:
                reward = -10
        return observation, reward, done, info

    def reset(self):
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        return obs

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2 ** 31
        seed3 = seeding.hash_seed(seed2 + 1) % 2 ** 31
        seed4 = seeding.hash_seed(seed3 + 1) % 2 ** 31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,))


class T1DDiscreteEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            adult_patients = [p for p in patient_names if "adult" in p]
            patient_name = random.choice(adult_patients)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.actions = [0.001, 0.005, 0.01, 0.1, 0.2, 0.5, 0.8, 1.0]
        self.env, _, _, _ = self._create_env_from_random_state()

    def step(self, action):
        # This gym only controls basal insulin
        action = self.actions[action]
        act = Action(basal=action, bolus=0)

        if self.reward_fun is None:
            return self.env.step(act)
        observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        if done:
            reward = -2000
        return observation, reward, done, info

    def reset(self):
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        return obs

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2 ** 31
        seed3 = seeding.hash_seed(seed2 + 1) % 2 ** 31
        seed4 = seeding.hash_seed(seed3 + 1) % 2 ** 31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):

        return spaces.Discrete(6)

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(1,))


class T1DInsObsSimEnv(gym.Env):
    '''
    A wrapper of simglucose.simulation.env.T1DSimEnv to support gym API
    '''
    metadata = {'render.modes': ['human']}

    SENSOR_HARDWARE = 'Dexcom'
    INSULIN_PUMP_HARDWARE = 'Insulet'

    def __init__(self, patient_name=None, reward_fun=None, seed=None):
        '''
        patient_name must be 'adolescent#001' to 'adolescent#010',
        or 'adult#001' to 'adult#010', or 'child#001' to 'child#010'
        '''
        # have to hard code the patient_name, gym has some interesting
        # error when choosing the patient
        if patient_name is None:
            # patient_name = 'adolescent#001'
            adult_patients = [p for p in patient_names if "adult" in p]
            patient_name = random.choice(adult_patients)
            print(patient_name)
        self.patient_name = patient_name
        self.reward_fun = reward_fun
        self.np_random, _ = seeding.np_random(seed=seed)
        self.env, _, _, _ = self._create_env_from_random_state()
        self.last_insulin = 0

    def step(self, action):
        # This gym only controls basal insulin
        act = Action(basal=action, bolus=0)
        if self.reward_fun is None:
            return self.env.step(act)
        observation, reward, done, info = self.env.step(act, reward_fun=self.reward_fun)
        # if done:
        #     reward = -10  # -10000
        observation = self.add_last_insulin_val_to_obs(observation[0])
        self.last_insulin = action[0]
        return observation, reward, done, info

    def reset(self):
        self.last_insulin = 0
        self.env, _, _, _ = self._create_env_from_random_state()
        obs, _, _, _ = self.env.reset()
        return self.add_last_insulin_val_to_obs(obs[0])

    def add_last_insulin_val_to_obs(self, obs):
        return [obs, self.last_insulin]

    def seed(self, seed=None):
        self.np_random, seed1 = seeding.np_random(seed=seed)
        self.env, seed2, seed3, seed4 = self._create_env_from_random_state()
        return [seed1, seed2, seed3, seed4]

    def _create_env_from_random_state(self):
        # Derive a random seed. This gets passed as a uint, but gets
        # checked as an int elsewhere, so we need to keep it below
        # 2**31.
        seed2 = seeding.hash_seed(self.np_random.randint(0, 1000)) % 2 ** 31
        seed3 = seeding.hash_seed(seed2 + 1) % 2 ** 31
        seed4 = seeding.hash_seed(seed3 + 1) % 2 ** 31

        hour = self.np_random.randint(low=0.0, high=24.0)
        start_time = datetime(2018, 1, 1, hour, 0, 0)
        patient = T1DPatient.withName(self.patient_name, random_init_bg=True, seed=seed4)
        sensor = CGMSensor.withName(self.SENSOR_HARDWARE, seed=seed2)
        scenario = RandomScenario(start_time=start_time, seed=seed3)
        pump = InsulinPump.withName(self.INSULIN_PUMP_HARDWARE)
        env = _T1DSimEnv(patient, sensor, pump, scenario)
        return env, seed2, seed3, seed4

    def render(self, mode='human', close=False):
        self.env.render(close=close)

    @property
    def action_space(self):
        ub = self.env.pump._params['max_basal']
        return spaces.Box(low=0, high=ub, shape=(1,))

    @property
    def observation_space(self):
        return spaces.Box(low=0, high=np.inf, shape=(2,))
