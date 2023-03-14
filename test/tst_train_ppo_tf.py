"""
Changes that have been made:
- Refactor script name (from test_with_ins_obs_ppo_tf.py)
- Remove unused imports and variables
"""
from simglucose.simulation.env import risk_diff as orig_risk_diff
from stable_baselines import PPO2, DDPG
from stable_baselines.common import make_vec_env
from stable_baselines.common.vec_env import SubprocVecEnv

from train.env.simglucose_gym_env import T1DSimEnv, T1DDiscreteSimEnv, T1DAdultSimEnv, T1DDiscreteEnv, T1DInsObsSimEnv

from train.reward.custom_rewards import risk_diff, partial_negativity

latest_saved_model = 'training_ws/best_model/best_model_ddpg_0_3.zip'

print(latest_saved_model)


def main():
    vec_env_kwargs = {'start_method': 'fork'}
    env_kwargs = {'reward_fun': partial_negativity}
    env = make_vec_env(T1DInsObsSimEnv, n_envs=32, vec_env_cls=SubprocVecEnv,
                       vec_env_kwargs=vec_env_kwargs, env_kwargs=env_kwargs)
    model = PPO2.load(latest_saved_model, env=env)

    observation = env.reset()

    env.training = False
    total_reward = 0
    total_time_steps = 10000
    for t in range(total_time_steps):

        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        total_reward += reward
        print('Meal = {},'.format(info[0]['meal']) + ' Obs = {},'.format(observation[0]) + ' Action = {}'.format(
            action[0]))
        print("Reward = {}".format(reward[0]))

        if done[0] or t == total_time_steps:
            print("Episode finished after {} timesteps".format(t + 1))
            print(total_reward[0])
            break


if __name__ == '__main__':
    main()
