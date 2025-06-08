# Reinforcement Learning for Aircraft Flight Control

This is a reinforcement learning flight control project for EEC256 at UC Davis, Spring 2025. 

The aircraft learns to preserve flight time and altitude following an engine out scenario.

I used JSBSim(https://jsbsim.sourceforge.net/) and stable baselines 3 to provide the environment and dynamics, respectively, for this project.

## Execution
To run the code, execute the train.py script in the scripts directory. The script will train the agent and save the model to a file.

To evaluate the trained agent, run the evaluate.py script in the scripts directory. This will load the saved model and run it in the environment. You will need to tweak the `model_meta` variable in the `evaluate.py` script to point to the correct model file. The naming is based on the ppo config file in the configs directory.

`evaluate.py` will also save plots of the agent's performance during evaluation, which can be found in the `plots` directory.

To run a 1,000,000 timestep training session takes roughly 20 minutes per condition. With 4 conditions, this will take about 1 hour and 20 minutes. The training is CPU intensive only.

## Requirements
- Python 3.8+
- JSBSim
- Stable Baselines 3
- NumPy
- Matplotlib
- Pandas
- Gym
- TensorFlow or PyTorch (depending on the version of Stable Baselines 3 you are using)