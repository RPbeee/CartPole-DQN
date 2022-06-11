import src.env
import src.model
import src.agent
import src.settings

env,nb_actions=src.env.get_env()
model=src.model.get_model(env,nb_actions)
memory,policy,dqn=src.agent.get_agent(model,nb_actions)

# 学習
dqn.fit(env, nb_steps=src.settings.STEPS, visualize=True, verbose=2)
print("学習が終わりました")
dqn.save_weights("dqn_weights.h5f", overwrite=True)