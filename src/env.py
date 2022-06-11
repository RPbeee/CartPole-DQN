import gym
MAX_STEPS=600

def get_env():
    # 環境の生成
    env = gym.make('CartPole-v1')
    env._max_episode_steps=MAX_STEPS
    nb_actions = env.action_space.n
    return env,nb_actions