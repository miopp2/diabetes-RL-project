import os
from datetime import datetime, timedelta

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

from test.analysis.report import report
# from simglucose.analysis.report import report
from simglucose.simulation.env import T1DSimEnv
from simglucose.controller.pid_ctrller import PIDController
from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.sim_engine import SimObj, batch_sim
from datetime import timedelta
from datetime import datetime
import pkg_resources
import numpy as np
import pandas as pd
import os
import copy

from test.controller import PPOController

# import patient parameters
PATIENT_PARA_FILE = pkg_resources.resource_filename(
    'simglucose', 'params/vpatient_params.csv')

# define start date as hour 0 of the current day
now = datetime.now()
start_time = datetime.combine(now.date(), datetime.min.time())


def create_result_folder():
    """create results folder"""
    _folder_name = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    _path = os.path.join(os.path.abspath('./results/'), _folder_name)
    os.makedirs(_path, exist_ok=True)
    return _path


def build_envs(name, _scenario):
    """create environments for simulation
    name: patient name, e.g. 'adolescent#001
    scenario: simglucose.simulation.scenario.CustomScenario"""
    patient = T1DPatient.withName(name)
    sensor = CGMSensor.withName('Dexcom', seed=1)
    pump = InsulinPump.withName('Insulet')
    scen = copy.deepcopy(_scenario)
    env = T1DSimEnv(patient, sensor, pump, scen)
    return env


def create_scenario(_base_scen, _sim_days, vary=True):
    repeat_scen = []
    vary_scen = []
    for simDay in range(_sim_days):
        for time, mealsize in _base_scen:
            repeat_scen.append((24 * simDay + time, mealsize))
    if vary == True:
        for meal, vals in enumerate(repeat_scen):
            time, CHO = vals
            time += np.random.normal(0.0, 0.25)
            if not (meal - 3) % 4 == 0:
                CHO += np.random.normal(0.0, 10)
            else:
                CHO += np.random.normal(0.0, 5)
            vary_scen.append((float(f'{time:.2f}'), float(f'{CHO:.2f}')))
        return CustomScenario(start_time=start_time, scenario=vary_scen), vary_scen
    return CustomScenario(start_time=start_time, scenario=repeat_scen), repeat_scen


def write_log(_envs, _patient_names, _base_scen, _mod_scen, _controllers, save_path, model):
    """Generate log file containing infos to simulation. Contains patient names, sensor type, pump type, base scenario
    modified scenario, and controllers"""
    ctrllers = [controller.__class__.__name__ for controller in _controllers]
    log = ['patient name: ', str(patient_names),
           'sensor: ', _envs[0].sensor.name,
           'pump: ', _envs[0].pump._params[0],
           'base scen: ', str(base_scen),
           'scen: ', str(_mod_scen),
           'controllers: ', str(ctrllers),
           'Model', model]

    with open(os.path.join(save_path, 'scen.txt'), 'w') as f:
        f.write('\n'.join(log))


def select_patients(_patient_group='All'):
    """Select patients to run simulation for.
    Valid choices: 'All' (default), 'Adolescents', 'Adults', 'Children'"""
    patient_params = pd.read_csv(PATIENT_PARA_FILE)
    all_patients = list(patient_params['Name'].values)
    if _patient_group == 'All':
        return all_patients
    elif _patient_group == 'Adolescents':
        return all_patients[:10]
    elif _patient_group == 'Adults':
        return all_patients[10:20]
    elif _patient_group == 'Children':
        return all_patients[20:30]


def create_ctrllers(_ctrllers):
    """Enable to run same scenario with multiple controllers"""
    _controllers = []
    for _controller in _ctrllers:
        as_list = [copy.deepcopy(_controller) for _ in range(len(envs))]
        _controllers.extend(as_list)
    return _controllers


if __name__ == '__main__':
    path = create_result_folder()
    latest_saved_model = './models/best_model_T1DDiscreteSimEnv.zip'

    # select controller to run simulation with
    controllers = [BBController(), PIDController(P=-0.0001, I=-0.000000275, D=-0.1), PPOController(0, latest_saved_model)]
    # controllers = [PPOController(0, latest_saved_model)]

    # Select parameters to run simulation for
    patient_group = 'All'
    sim_days = 7
    patient_names = select_patients(patient_group)
    # patient_names = ['adolescent#001', 'adult#001', 'child#007']

    # set base scenario and add variability and repeat for as many days as necessary
    base_scen = [(7, 70), (10, 30), (14, 110), (21, 90)]
    scenario, mod_scen = create_scenario(base_scen, sim_days, vary=True)

    for num, controller in enumerate(controllers):
        folder_name = controller.__class__.__name__
        path_ctrl = os.path.join(path, folder_name)
        os.makedirs(path_ctrl, exist_ok=True)

        # create environment for each patient
        envs = [build_envs(patient, scenario) for patient in patient_names]

        if num == 0:
            write_log(envs, patient_names, base_scen, mod_scen, controllers, path, latest_saved_model)

        # copy controller for each environment
        if controller.__class__ == PPOController:
            ctrllers = [PPOController(0, latest_saved_model) for _ in range(len(envs))]
        else:
            ctrllers = [copy.deepcopy(controller) for _ in range(len(envs))]

        # create simulation objects
        sim_instances = [
            SimObj(env, ctrl, timedelta(days=sim_days), animate=False, path=path_ctrl)
            for (env, ctrl) in zip(envs, ctrllers)
        ]

        if controller.__class__ == PPOController:
            results = batch_sim(sim_instances, parallel=False)
        else:
            results = batch_sim(sim_instances, parallel=True)

        df = pd.concat(results, keys=[s.env.patient.name for s in sim_instances])
        results, ri_per_hour, figs, axes = report(df, path_ctrl)
