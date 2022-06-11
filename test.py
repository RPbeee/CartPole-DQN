import src.env
import src.model
import src.agent

env,nb_actions=src.env.get_env()
model=src.model.get_model(env,nb_actions)
memory,policy,dqn=src.agent.get_agent(model,nb_actions)

# モデルをテスト
dqn.load_weights("dqn_weights.h5f")
dqn.test(env, nb_episodes=8, visualize=True)