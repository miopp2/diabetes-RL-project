from stable_baselines3 import PPO, DDPG

from train.env.simglucose_gym_env import T1DSimEnv
import glob
import os

from train.reward.custom_rewards import custom_reward

# list_of_files = glob.glob('./training_ws/*ddpg*.zip')  # * means all if need specific format then *.csv
# latest_saved_model = max(list_of_files, key=os.path.getctime)
latest_saved_model = './test/models/best_model_ddpg.zip'
print(latest_saved_model)


def main():
    vec_env_kwargs = {'start_method': 'spawn'}
    # env = make_vec_env(T1DSimEnv, n_envs=40, monitor_dir='./training_ws', vec_env_cls=SubprocVecEnv,
    #                   vec_env_kwargs=vec_env_kwargs)
    env = T1DSimEnv(patient_name='adult#004', reward_fun=custom_reward)
    model = DDPG.load(latest_saved_model, env=env)
    env = env
    observation = env.reset()

    env.training = False

    for t in range(1000):
        action, _states = model.predict(observation, deterministic=True)
        observation, reward, done, info = env.step(action)
        print(observation)
        print("Reward = {}".format(reward))
        print("Action = {}".format(action))
        # print('Info = {}'.format(info))

        env.render(mode='human')
        # if done:
        #    print("Episode finished after {} timesteps".format(t + 1))
        #    break


if __name__ == '__main__':
    main()
