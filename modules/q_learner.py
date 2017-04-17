"""Definitions and functions used during Q-learning.

"""

import random
import numpy as np
import torch
import torch.optim as optim
import torch.autograd as autograd

#
# Experience replay memory 
#

# This will be used during the training when the loss function to be
# minimized will be averaged over a minibatch (sample) of experiences
# drawn randomly from the replay_memory .memory object using method
# .sample

# namedtuple: tuple subclass with elements accessible by name with
# . operator (here name class=name instance) e_t = (s_t, a_t, r_t,
# s_{t+1}) globally defined and used by replay_memory
experience = namedtuple('Experience',
                        ('observation', 'action', 'reward', 'next_observation'))

class replay_memory(object):
    """A cyclic buffer of bounded size that holds the transitions
    observed recently.  It also implements a .sample() method for
    selecting a random batch of transitions for training.

    """
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = experience(*args)
        # cyclicity:
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class q_learner():
    """Class q_learner. An agent instance will have the possibility of
    perceiving, using eps_greedy policy.

    members:
    - replay memory
    - loss function
    - optimizer
    - neural network

    will have a main method "perceive" which takes e=(s,r,a,s') and t.  it
    stores e in its memory and computes the next action according to an
    eps-greedy policy. Then it performs an optimization step of its neural
    network to reduce its loss function.

    """

    def __init__(self, gpars, pars):
        # assign parameters
        self.C_in = pars['C_in']
        self.C_H = pars['C_H'] 
        self.C_out = pars['C_out']
        self.kernel_size = pars['kernel_size']
        self.stride = pars['stride']
        self.obs_window_W = pars['obs_window_W'] 
        self.obs_window_H = pars['obs_window_H']
        self.gamma = pars['gamma'] 
        self.capacity = pars['capacity']
        self.batch_size = pars['batch_size']
        self.eps_start = pars['eps_start']
        self.eps_end = pars['eps_end']
        self.decay_rate = pars['decay_rate']
        self.n_actions = gpars['n_actions']
        # memory, model (nn approximator for Q), loss_fn, optimizer
        self.rpl_memory = replay_memory(self.capacity)
        self.model = dqn(C_in, C_H, C_out, kernel_size, stride,
                         obs_window_H, obs_window_W)
        self.loss_fn = torch.nn.MSELoss(size_average=False)
        self.optimizer = optim.SGD(Q_0.parameters(),lr=0.01)

    #
    # Policy
    #
    def eps_decay(self, t):
        """Returns the value of eps at time t according to epsilon decay from
        eps_start to eps_end with decay rate gamma
        
        """
        ret = self.eps_end + \
              (self.eps_start - self.eps_end) * np.exp(-1. * t / self.decay_rate)
        return ret

    def eps_greedy(self, obs, eps):
        """epsilon-greedy policy. Input:
        obs : an observation, already preprocessed tensor below promoted to autograd.Variable 
        n_action : the number of possible actions (gathering, = 8)
        t : time.
        Returns an action."""
        assert(0 <= eps <= 1)
        random_num = random.random()
        print('rand',random_num, 'eps',eps)
        if random_num > eps:
            # to be adjusted eventually. 
            # volatile: Boolean indicating that the Variable should be used in
            # inference mode (forward), i.e. don't save the history. See
            # :ref:`excluding-subgraphs` for more details.
            # Can be changed only on leaf Variables.
            print('In eps_greedy')
            y_pred = self.model(autograd.Variable(obs, volatile=True))
            # data.max(1) returns an array with 0 component the maximum values for each sample in the batch
            # and 1 component their indices, which is selected here, so giving which action maximizes the model for Q.
            return y_pred.data.max(1)[1].cpu()
        else:
            print('In rand policy')
            return torch.LongTensor([[random.randrange(n_actions)]])

    #
    # preprocess 
    #
    def preprocess_obs(self, raw_obs):
        """preprocessed input raw observation window of player from game.
        Convert to float, convert to torch tensor (this doesn't require a
        copy) and add a batch dimension

        """
        ret = np.ascontiguousarray(raw_obs, dtype=np.float32) 
        ret = torch.from_numpy(ret).unsqueeze(0)
        #print('my_obs',my_obs.size(),type(my_obs))
        return ret

    #
    # optimize 
    #
    def optimize(self):
        """ ..."""
        # if the memory is smaller than wanted, don't do anything and keep building memory
        print('In optimize: len(rpl_memory), bacth_size', len(self.rpl_memory), self.batch_size)
        if len(self.rpl_memory) < self.batch_size:
            return
        #otherwise get minibatch of experiences
        # this is a list of namedtuples
        sample_experience = self.rpl_memory.sample(self.batch_size)
        # Transpose the batch (see http://stackoverflow.com/a/19343/3343043 for
        # detailed explanation).
        # This is a namedtuple of lists
        minibatch = experience(*zip(*sample_experience))
        print('minibatch.reward:',minibatch.reward)

        # get obs,action,next_obs,reward batches in Variable
        for s in minibatch.next_observation:
            if s is None:
                print('########### None')
        next_obs_batch = autograd.Variable(torch.cat(minibatch.next_observation),
                                           volatile=True)
        obs_batch = autograd.Variable(torch.cat(minibatch.observation))
        action_batch = autograd.Variable(torch.cat(minibatch.action))
        reward_batch = autograd.Variable(torch.cat(minibatch.reward))
    
        # Compute Q(obs, action) - the model computes Q(obs), then we select the
        # columns of actions taken
        print("In optimize: obs_batch", obs_batch.data.size())
        obs_action_values = self.model(obs_batch).gather(1, action_batch)
    
        # Compute V(obs')=max_a Q(obs, a) for all next states.
        next_obs_values = self.model(next_obs_batch).max(1)[0]
        # Now, we don't want to mess up the loss with a volatile flag, so let's
        # clear it. After this, we'll just end up with a Variable that has
        # requires_grad=False
        next_obs_values.volatile = False
        # Compute y
        y = (next_obs_values * self.gamma) + reward_batch
    
        # Compute loss
        loss = self.loss_fn(obs_action_values, y)
    
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.model.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    #
    # perceive
    #
    def perceive(self, s, r, a, sp, t):
        """input: e=(s,r,a,sp) and t.  it stores e in its memory and computes
        the next action according to an eps-greedy policy. Then it
        performs an optimization step of its neural network to reduce
        its loss function.

        """
        # ...
