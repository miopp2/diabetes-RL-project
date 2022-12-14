import os

import numpy as np
from stable_baselines3 import DDPG
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.td3 import MlpPolicy

from train.env.simglucose_gym_env import T1DSimEnv
from train.reward.custom_rewards import custom_reward
from train.save_on_best_result_callback import SaveOnBestTrainingRewardCallback
import torch as th


def main():
    save_folder = 'training_ws_ddpg/'
    if save_folder is not None:
        os.makedirs(save_folder, exist_ok=True)
    env = Monitor(T1DSimEnv('adult#003', reward_fun=custom_reward), filename=save_folder)

    checkpoint_callback = CheckpointCallback(save_freq=2000, save_path=save_folder,
                                             name_prefix="rl_ddpg_model")
    best_model_save_callback = SaveOnBestTrainingRewardCallback(check_freq=2000, log_dir=save_folder)

    n_actions = env.action_space.shape[-1]
    # net_arch = {'qf': [32, 32], 'pi': [32, 32]}
    net_arch = [32, 32]
    action_noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma=0.3 * np.ones(n_actions))
    policy_kwargs = dict(activation_fn=th.nn.ReLU6, net_arch=net_arch)
    model = DDPG(MlpPolicy, env, gamma=0.99, buffer_size=10000, batch_size=32,
                 tensorboard_log="./simglucose_ddpg_tensorboard/", policy_kwargs=policy_kwargs,
                 action_noise=action_noise, learning_rate=1e-6)

    model.learn(total_timesteps=1200000, callback=[best_model_save_callback, checkpoint_callback])


if __name__ == "__main__":
    main()
