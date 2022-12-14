import glob
import os
from datetime import datetime, timedelta
from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

from simglucose.actuator.pump import InsulinPump
from simglucose.controller.pid_ctrller import PIDController
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.sensor.cgm import CGMSensor
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario import CustomScenario
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.sim_engine import SimObj, sim
from simglucose.simulation.user_interface import simulate

from test.controller import PPOController
from train.env.simglucose_gym_env import T1DAdultSimEnv
from train.reward.custom_rewards import partial_negativity

# list_of_files = glob.glob('.train/training_ws/*.zip')  # * means all if need specific format then *.csv
#latest_saved_model = max(list_of_files, key=os.path.getctime)
latest_saved_model = './test/rl_model_1187840_steps.zip'
# 1187840_steps
controller = PPOController(0, latest_saved_model)

now = datetime.now()
start_time = datetime.combine(now.date(), datetime.min.time())
path = './results'

patient = T1DPatient.withName('adult#002')
sensor = CGMSensor.withName('Dexcom', seed=1)
pump = InsulinPump.withName('Insulet')
#scen = [(0, 10),  (12, 22), (18, 15), (23, 10)]

scen = [(1, 70),  (12, 70), (18, 70), (23, 70)]
scenario = CustomScenario(start_time=start_time, scenario=scen)
env = T1DSimEnv(patient, sensor, pump, scenario)

s1 = SimObj(env, controller, timedelta(days=1), animate=True, path=path)

sim(s1)

pid_controller = PIDController(P=0.001, I=0.00001, D=0.001)
s2 = SimObj(env, pid_controller, timedelta(days=1), animate=True, path=path)
sim(s2)
