from warnings import simplefilter

simplefilter(action='ignore', category=FutureWarning)
simplefilter(action='ignore', category=UserWarning)

from simglucose.simulation.env import risk_diff as orig_risk_diff
from stable_baselines import PPO2
from stable_baselines.common import make_vec_env
from stable_baselines.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines.common.policies import MlpLnLstmPolicy, MlpLstmPolicy
from stable_baselines.common.schedules import LinearSchedule, PiecewiseSchedule
from stable_baselines.common.vec_env import SubprocVecEnv, VecNormalize


from train.env.simglucose_gym_env import T1DSimEnv, T1DDiscreteSimEnv, T1DAdultSimEnv, T1DAdultSimV2Env, T1DDiscreteEnv, \
    T1DInsObsSimEnv
from train.reward.custom_rewards import shaped_reward_around_normal_bg, shaped_negative_reward_around_normal_bg, \
    smooth_reward, no_negativity, risk_diff, no_negativityV2, partial_negativity
from train.save_on_best_result_callback_v2 import SaveOnBestTrainingRewardCallback


def main():
    save_folder = 'training_ws_disc/'
    checkpoint_callback = CheckpointCallback(save_freq=128, save_path=save_folder,
                                             name_prefix="rl_model")
    env_class = T1DDiscreteSimEnv
    reward_func = orig_risk_diff
    vec_env_kwargs = {'start_method': 'spawn'}
    env_kwargs = {'reward_fun': reward_func}
    n_envs = 32
    env = make_vec_env(env_class, n_envs=n_envs, monitor_dir='./training_ws_disc', vec_env_cls=SubprocVecEnv,
                       vec_env_kwargs=vec_env_kwargs, env_kwargs=env_kwargs)
    # env = VecNormalize(env, clip_obs=350, clip_reward=10000, gamma=0.9999)
    # todo: change network architecture
    policy_kwargs = {'net_arch': [64, 64, 'lstm', dict(vf=[64, 64, 64, 64], pi=[64, 64, 64, 64])], 'n_lstm': 32}
    # policy_kwargs = {'net_arch': [32, 'lstm', dict(vf=[32, 32], pi=[32, 32])], 'n_lstm': 32}
    #cliprange_scd = PiecewiseSchedule(
    #    endpoints=[(0.01, 0.0001), (0.992, 0.001), (0.994, 0.01), (0.998, 0.1), (0.999, 0.13)],
    #    outside_value=0.15).value

    model = PPO2(MlpLnLstmPolicy, env, verbose=1, tensorboard_log="./simglucose_ppo_tensorboard/",
                 n_steps=128, learning_rate=3e-5, ent_coef=0.0001, gamma=0.999, nminibatches=n_envs,
                 policy_kwargs=policy_kwargs, vf_coef=0.8, lam=0.91, cliprange=0.1,
                 cliprange_vf=0.1)

    model.learn(total_timesteps=1200000, callback=[checkpoint_callback])


if __name__ == "__main__":
    main()
#  , dict(vf=[32, 32], pi=[32, 32])
