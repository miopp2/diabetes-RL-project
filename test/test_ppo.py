from simglucose.simulation.env import risk_diff
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecNormalize, SubprocVecEnv

from train.env.simglucose_gym_env import T1DSimEnv, T1DSimDiffEnv, T1DSimHistoryEnv
import glob
import os

from train.reward.custom_rewards import custom_reward, shaped_reward_around_normal_bg, \
    shaped_negative_reward_around_normal_bg, no_negativity

list_of_files = glob.glob('./training_ws/*.zip')  # * means all if need specific format then *.csv
latest_saved_model = max(list_of_files, key=os.path.getctime)


def main():
    vec_env_kwargs = {'start_method': 'spawn'}
    # env = make_vec_env(T1DSimEnv, n_envs=40, monitor_dir='./training_ws', vec_env_cls=SubprocVecEnv,
    #                   vec_env_kwargs=vec_env_kwargs)
    env_kwargs = {'reward_fun': no_negativity}
    env = make_vec_env(T1DSimHistoryEnv, n_envs=1, monitor_dir='./training_ws', vec_env_cls=SubprocVecEnv,
                       vec_env_kwargs=vec_env_kwargs, env_kwargs=env_kwargs)

    model = PPO.load(latest_saved_model, env=env)

    observation = env.reset()

    for t in range(1000):

        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        print('Obs = {}'.format(observation[0][-1]) + ' Action = {}'.format(action[0][0]))
        print("Reward = {}".format(reward[0]))

        if done:
            print("Episode finished after {} timesteps".format(t + 1))
            break


if __name__ == '__main__':
    main()