"""Code to train the agents.

"""

import argparse
from q_learner_file import q_learner
from gathering import gathering_game
import time
import matplotlib.pyplot as plt


def verbose(a0, a1, r0, r1, t, game, show_screen):
    """Print useful information about training.

    """
    print("---------------------------------------")
    print("Time", t)
    print("Action performed by player 0", game.get_key_action(a0))
    print("Action performed by player 1", game.get_key_action(a1))
    print("Reward player 0:",r0, " player 1:",r1)
    if show_screen:
        print("Printing current screen and directions:")
        game.show_screen()
        #to_show = game.s.transpose((2,1,0))
        #plt.imshow(to_show,origin='lower')
        plt.pause(.01)
    # returns nothing

def main(game_pars, q_learn_pars):
    """...

    """
    print('game', game_pars)
    print('q_learn_pars', q_learn_pars)
    show_screen = False

    # init game
    game = gathering_game(game_pars)
    # interactive mode for matplotlib on
    if show_screen == True:
        plt.ion()
    
    # init agents as q_learner, using values of epsilon, loss_fn, optimizer etc
    agent0 = q_learner(q_learn_pars)
    agent1 = q_learner(q_learn_pars)
    # set initial experience to None
    s0 = None
    s1 = None
    a0 = None
    a1 = None
    r0 = None
    r1 = None
    s0p = game.obs_0()
    s1p = game.obs_1()
    verbose(a0, a1, r0, r1, -1, game, show_screen) #t=-1 means boundary condition
    # init vec with reward
    vec_r0 = []
    vec_r1 = []
    
    # training loop over episodes
    for episode in range(game_pars['n_episodes']):
        for t in range(game_pars['tmax_episode']):
            # perceive 
            a0 = agent0.perceive(s0, a0, r0, s0p, t)
            a1 = agent1.perceive(s1, a1, r1, s1p, t)
            # execute action in emulator. (a_* is a 1x1 tensor)
            r0, r1 = game.transition_and_get_reward(a0[0,0], a1[0,0])
            # print some info
            verbose(a0[0,0], a1[0,0], r0, r1, t, game, show_screen)
            # accumulate statistics
            vec_r0.append(r0)
            vec_r1.append(r1)
            # update observations
            s0 = s0p
            s1 = s1p
            s0p = game.obs_0()
            s1p = game.obs_1()

    # plot reward
    plt.ioff()
    plt.close()
    plt.plot(vec_r0,'ro')
    plt.plot(vec_r1,'b*')
    plt.show()
    # end of main
    
            
# When it is executed from command line
if __name__ == '__main__':
    # Get from command line
    parser = argparse.ArgumentParser(description='train_agents')
    #
    list_args = []
    list_helps = []
    list_args += ['n_apples']
    list_helps += ['number of frames after which apples respawn']
    list_args += ['n_tagged']
    list_helps += ['num frames after which tagged player comes back']
    list_args += ['width']
    list_helps += ['width of the screen, always odd']
    list_args += ['height']
    list_helps += ['height of the screen, always odd']
    list_args += ['size_obs_ahead']
    list_helps += ['size of obs window front player']
    list_args += ['size_obs_side']
    list_helps += ['size of obs window each side player']
    #
    list_args += ['C_in']
    list_helps += ['in channel into the nn (3 for RGB)']
    list_args += ['C_H']
    list_helps += ['number of hidden units (or channels)']
    list_args += ['C_out']
    list_helps += ['out channels = number of actions']
    list_args += ['kernel_size']
    list_helps += ['kernel size']
    list_args += ['stride']
    list_helps += ['stride']
    list_args += ['gamma']
    list_helps += ['discount of reward']
    list_args += ['n_episodes']
    list_helps += ['total number of episodes']
    list_args += ['tmax_episode']
    list_helps += ['max time of an episode']
    #
    list_args += ['capacity']
    list_helps += ['capacity memory']
    list_args += ['batch_size']
    list_helps += ['size memory batch']
    #
    list_args += ['eps_start']
    list_helps += ['starting value of eps for policy']
    list_args += ['eps_end']
    list_helps += ['final value of eps for policy']
    list_args += ['decay_rate']
    list_helps += ['decay_rate of eps for policy']

    for a,h in zip(list_args,list_helps):
        if a == 'eps_start' or a == 'eps_end' or a == 'gamma':
            parser.add_argument('--'+a, type=float, help=h)
        else:
            parser.add_argument('--'+a, type=int, help=h)
    args = parser.parse_args()

    # def dictionaries containing the parameters
    game_pars={}
    game_pars['N_apples'] = vars(args)['n_apples']
    game_pars['N_tagged'] = vars(args)['n_tagged']
    game_pars['W'] = vars(args)['width']
    game_pars['H'] = vars(args)['height']
    game_pars['size_obs_ahead'] = vars(args)['size_obs_ahead']
    game_pars['size_obs_side'] = vars(args)['size_obs_side']
    game_pars['n_episodes'] = vars(args)['n_episodes']
    game_pars['tmax_episode'] = vars(args)['tmax_episode']
    
    # and q-learner pars
    q_learn_pars = {}
    q_learn_pars['C_in'] = vars(args)['C_in']
    q_learn_pars['C_H'] = vars(args)['C_H']
    q_learn_pars['C_out'] = vars(args)['C_out']
    q_learn_pars['kernel_size'] = vars(args)['kernel_size']
    q_learn_pars['stride'] = vars(args)['stride']
    q_learn_pars['obs_window_W'] = 2*vars(args)['size_obs_side'] + 1
    q_learn_pars['obs_window_H'] = vars(args)['size_obs_ahead'] + 1
    q_learn_pars['gamma'] = vars(args)['gamma']
    #
    q_learn_pars['capacity'] = vars(args)['capacity'] 
    q_learn_pars['batch_size'] =vars(args)['batch_size']
    #
    q_learn_pars['eps_start'] = vars(args)['eps_start']
    q_learn_pars['eps_end'] = vars(args)['eps_end']
    q_learn_pars['decay_rate'] = vars(args)['decay_rate']
    
    main(game_pars, q_learn_pars)
