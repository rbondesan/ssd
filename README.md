# ssd

Status report:
Tested modules/, currently testing and improving learning of the neural network.

Dependencies:
python3, NumPy, PyTorch (see web page for install instructions).

Testing is done on latest Ubuntu Linux.

Description of files:
modules/dqn_file.py : implementation of deep q network
modules/gathering.py : implementation of gathering game class
modules/q_learner_file.py : implementation of q_learner class, which wraps
			    several properties of agents
modules/train_agents.py : main train loop
modules/test_units.ipynb : ipython notebook which tests units
run.sh : bash script to run the train_agents module

Other files are temporary and should be ignored. They will be removed
when the initial development is over.

