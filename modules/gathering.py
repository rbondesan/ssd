"""gathering : a Markov game class

Define first a Markov game class for 2 players. (Define right away the
gathering game class, no need of a parent class. This replaces the
environment class e.g. OpenAI gym.)

Members are:

- States
    - A state is the screen image s \in S.
    - It also has actions of each player: A_1, A_2

- Public methods
    - observation function O(s,i) which returns a the portion of screen visible by a player (same size for any i)
    - transition function: T : S x A_1 x A_2 -> \Delta(S). We will consder a deterministic process, which returns next state.
    - Reward function r_i : S x A_1 x A_2 -> R
    
    Note: I think that in gym reward and transition are put together
    in env.step(action) which returns reward and updates the state.

Author: Roberto Bondesan
Date  : 13 April 2017

"""


import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt

class gathering_game():
    """
    Gathering game class. An object has the following attributes:
    
    - s : ndarray representing the image, the current state of the screen
    - dir : vector with directions of view of player 0,1. one of 0,1,2,3 = up,down,left,right
    - pars:
        - N_apples = number of frames after which apple respawns
        - N_tagged = number of frames a player hit by beam is removed from game
        - W = width screen, odd
        - H = height screen, odd
        - size_obs_front = number of sites the players can see in front of them
        - size_obs_side = number of sites the players can see on their side
    
    - action : a dictionary returning a onehot vector for each action among
      step forward, step back-ward, step left, step right, rotate left, rotate right, use beam and stand still.
    
    It also has methods to update the state (transition function) and to return the reward.
    
    Random thoughts:
    -agents are not transparent, so if i cannot move to position of j, i,j in {1,2}. Similarly rigid boundary.
    
    """
    
    #
    # Init functions
    #
    def get_init_screen(self):
        """Initialize the screen. 
        
        Apples are green and each player sees himself blue and the opponent red. 
        We choose to save player 1 blue and player 2 red, and to exchange
        blue with red in the observation function of player 2.
        
        Returns the initial frame.
        
        """        
        s = np.zeros((3,self.pars['W'],self.pars['H']),dtype=np.uint8) # black background
        # from top to bottom put apples
        s[1,self.pos_apples[:,0],self.pos_apples[:,1]] = 255 # green
        # Starting positions of players, symmetric at left border, so x = 0
        s[self.channel_pl[0],self.init_pos_x[0],self.init_pos_y[0]] = 255 # blue
        s[self.channel_pl[1],self.init_pos_x[1],self.init_pos_y[1]] = 255 # red
        return s
    
    def __init__(self, pars):
        #
        # assign parameters
        #
        self.pars = pars
        # initial directions, set also a dictionary used later which sets the absolute notions of 
        # up, down, left and right for an absolute observer.
        self.dir_dict = {'up':1,'down':3,'left':2,'right':0} # exp of 4th roots of 1
        #movement actions are chosen to easily compose with dir_dict as done in get_new_pos
        self.n_actions = 8
        self.actions_dict = {'step_forward':0, 
                             'step_back-ward':2, 
                             'step_left':1, 
                             'step_right':3, 
                             'rotate_left':4, 
                             'rotate_right':5, 
                             'use_beam':6,
                             'stand_still':7}
        # group actions by category
        self.action_cat = {'movement':[self.actions_dict['step_forward'],
                                       self.actions_dict['step_back-ward'],
                                       self.actions_dict['step_left'],
                                       self.actions_dict['step_right']],
                           'rotation_or_still':[self.actions_dict['rotate_left'],
                                                self.actions_dict['rotate_right'],
                                                self.actions_dict['stand_still']],
                           'beam':[self.actions_dict['use_beam']]}
        self.mid_x = int((self.pars['W']-1)/2)+1
        self.mid_y = int((self.pars['H']-1)/2)+1
        mid = np.array([self.mid_x,self.mid_y]) # coordinate central point
        dx = np.array([1,0])
        dy = np.array([0,1])
        self.pos_apples = np.array([mid+2*dy,
                      mid+dy-dx,mid+dy,mid+dy+dx,
                      mid-2*dx,mid-dx,mid,mid+dx,mid+2*dx,
                      mid-dy-dx,mid-dy,mid-dy+dx,
                      mid-2*dy])
        #
        # Init values that can be modified later on
        #
        self.dir = np.array([self.dir_dict['right'],self.dir_dict['right']])
        self.t_apples = [] # list of times elapsed since i-th apple removed. should be decreasing.
        self.rem_apples_pos = []
        self.is_tagged = [False,False] 
        self.t_tagged = [-1,-1] # init value of time elapsed since pl was tagged by beam
        self.global_time = 0
        # 2 players, so associate a channel to a player: 
        # choose blue with 0 and red with 1. index of list is pl num.
        self.channel_pl = [2,0]
#        self.init_pos_x = [0,0] # initial x position of pl 0 and 1
        self.init_pos_x = [self.mid_x-3,self.mid_x-3] # initial x position of pl 0 and 1
        self.init_pos_y = [int((self.pars['H']-1)/2)+2,int((self.pars['H']-1)/2)] # initial y position of pl 0 and 1
        # Last init screen
        self.s = self.get_init_screen()
        self.prev_s = None
    
    def reset(self):
        """Reset to init values as in class constructor"""
        self.dir = np.array([self.dir_dict['right'],self.dir_dict['right']])
        self.t_apples = [] # list of times elapsed since i-th apple removed. should be decreasing.
        self.rem_apples_pos = []
        self.is_tagged = [False,False] 
        self.t_tagged = [-1,-1] # init value of time elapsed since pl was tagged by beam
        self.global_time = 0
        # 2 players, so associate a channel to a player: 
        # choose blue with 0 and red with 1. index of list is pl num.
        self.channel_pl = [2,0]
        self.init_pos_x = [0,0] # initial x position of pl 0 and 1
        self.init_pos_y = [int((self.pars['H']-1)/2)+2,int((self.pars['H']-1)/2)] # initial y position of pl 0 and 1
        # Last init screen
        self.s = self.get_init_screen()
        self.prev_s = None
        
    #
    # observation functions
    #
    def pad_and_slice(self, obs_window_up, obs_window_down,
                      obs_window_left, obs_window_right, s):
        """determine how many extra black pixels have to be added at
        top,bottom,left,right in case the observation window exceeds
        the dimension of the screen.  Then screen acts then as a
        "fence" and inside and outside it all the pixels are seen
        black by the players.

        """
        pad = [0,0,0,0]
        dup = obs_window_up - self.pars['H']
        if dup > 0:
            pad[self.dir_dict['up']] = dup
            obs_window_up = self.pars['H']
        ddown = 0 - obs_window_down
        if ddown > 0:
            pad[self.dir_dict['down']] = ddown
            obs_window_down = 0
        dright = obs_window_right - self.pars['W']
        if dright > 0:
            pad[self.dir_dict['right']] = dright
            obs_window_right = self.pars['W']
        dleft = 0 - obs_window_left
        if dleft > 0:
            pad[self.dir_dict['left']] = dleft
            obs_window_left = 0

        # print('In pad_and_slice: obs_window_up, obs_window_down, obs_window_right, obs_window_left',
        #      obs_window_up,obs_window_down,obs_window_right,obs_window_left)
        # print('In pad_and_slice: pad_up, pad_down, pad_right, pad_left',
        #       pad[self.dir_dict['up']],pad[self.dir_dict['down']],
        #       pad[self.dir_dict['right']],pad[self.dir_dict['left']])

        # slc_screen_x, slc_screen_y: slices giving the slice of the
        # screen observed in x and y direction.
        slc_screen_x = slice(obs_window_left,obs_window_right)
        slc_screen_y = slice(obs_window_down,obs_window_up)
        # print('In pad_and_slice: slc_screen_x, slc_screen_y', slc_screen_x, slc_screen_y)
        ret = s[:,slc_screen_x, slc_screen_y]
        # pad 
        # with 4 components telling how many zeros need to be padded to
        # fill the observed region outside the screen.
        ret = np.lib.pad(ret, ((0,0),(pad[self.dir_dict['left']],pad[self.dir_dict['right']]),
                               (pad[self.dir_dict['down']], pad[self.dir_dict['up']])), 
                         'constant', constant_values=(255))
        return ret

    def get_obs(self, d, pos_x, pos_y, s):
        """d = direction of player pl, pos_x and pos_y its position.  s =
        screen.
        
        It should be size_obs_front ahead and size_obs_side to
        left/right, taking into account boundaries. The portion of
        screen is encoded in obs_window_* and the padding zeros in
        pad.  Flip is done so that the player has coordinates of space
        increasing from its left to its right and ahead of him.
        
        Returns: 
        ret: the observation window.

        """

        # determine case by case the distance to boundary ahead and player's left/right side
        # given a direction, the observation
        if d == self.dir_dict['up']:
            obs_window_up = pos_y + self.pars['size_obs_ahead']+1
            obs_window_down = pos_y
            obs_window_right = pos_x + self.pars['size_obs_side']+1
            obs_window_left = pos_x - self.pars['size_obs_side']
            ret = self.pad_and_slice(obs_window_up, obs_window_down,
                                     obs_window_left, obs_window_right, s)
            # no flip needed
            
        elif d == self.dir_dict['down']:
            obs_window_up = pos_y+1
            obs_window_down = pos_y - self.pars['size_obs_ahead']
            obs_window_right = pos_x + self.pars['size_obs_side']+1
            obs_window_left = pos_x - self.pars['size_obs_side']
            ret = self.pad_and_slice(obs_window_up, obs_window_down,
                                     obs_window_left, obs_window_right, s)
            # flip both directions x,y
            ret = np.flip(ret, 1)
            ret = np.flip(ret, 2)

        elif d == self.dir_dict['left']:
            obs_window_up = pos_y + self.pars['size_obs_side']+1
            obs_window_down = pos_y - self.pars['size_obs_side']
            obs_window_right = pos_x+1
            obs_window_left = pos_x - self.pars['size_obs_ahead']
            ret = self.pad_and_slice(obs_window_up, obs_window_down,
                                     obs_window_left, obs_window_right, s)
            # flip both directions x
            ret = np.flip(ret, 1)
            
        elif d == self.dir_dict['right']:
            obs_window_up = pos_y + self.pars['size_obs_side']+1
            obs_window_down = pos_y - self.pars['size_obs_side']
            obs_window_right = pos_x + self.pars['size_obs_ahead']+1
            obs_window_left = pos_x
            ret = self.pad_and_slice(obs_window_up, obs_window_down,
                                     obs_window_left, obs_window_right, s)
            # flip y
            ret = np.flip(ret, 2)

        # finally transpose to canonical view if direction up or down.
        if ret.shape == (3, 2*self.pars['size_obs_side']+1, self.pars['size_obs_ahead']+1):
            ret = ret.transpose((0,2,1))
            
        return ret
    
    def get_pos_pl(self,pl,s):
        """Get position of player pl given screen s.  Namely, pl 0 is where
        blue, player 1 where red.

        """
        assert(pl==0 or pl==1)
        ch_pl = self.channel_pl[pl] # channel of player
        other_chs = [x for x in [0,1,2] if x != ch_pl]
        tmp = np.argwhere(s[ch_pl,:,:] == 255)
#        print("In get_pl_pos, tmp", tmp)
        # check other are zero since beam makes yellow line and
        # outside window is white. Assume at most 1 pixel blue or
        # red. [-1,-1] is value returned if tmp is empty, meaning that
        # player has been tagged.
        ret = [-1,-1] 
        for [x,y] in tmp:
            if s[other_chs[0],x,y]==0 and s[other_chs[1],x,y]==0:
                ret = [x,y]
                break
#        print("In get_pl_pos, ret", ret)
        return ret
    
    def obs_0(self):
        """Observation function of player 0. The observation window is from
        the perspective of the player.

        """
        [pos_0_x,pos_0_y] = self.get_pos_pl(0,self.s)
        print('In obs_0, pos_0:',pos_0_x,pos_0_y)
        ret = self.get_obs(self.dir[0], pos_0_x, pos_0_y, self.s)        
        # print('In obs_0: ret.shape, right one', ret.shape, 
        #       (3, self.pars['size_obs_ahead']+1, 2*self.pars['size_obs_side']+1))
        assert(ret.shape == (3, self.pars['size_obs_ahead']+1, 2*self.pars['size_obs_side']+1))
        return ret
    
    def obs_1(self):
        """Observation function of player 1. The observation window is from
        the perspective of the player.  Player 1 sees himself as blue
        and player 0 as red. What is stored in self.s is the color
        setting of player 0, so here invert.

        """
        # determine position of player 1: 
        [pos_1_x,pos_1_y] = self.get_pos_pl(1,self.s)        
        print('In obs_1, pos_1:',pos_1_x,pos_1_y)
        # invert red with blue: first get positions of players in ret:
        inv_s = deepcopy(self.s)
        [pos_0_x,pos_0_y] = self.get_pos_pl(0,self.s)
        inv_s[self.channel_pl[1],pos_1_x, pos_1_y] = 0
        inv_s[self.channel_pl[0],pos_1_x, pos_1_y] = 255
        inv_s[self.channel_pl[1],pos_0_x, pos_0_y] = 255
        inv_s[self.channel_pl[0],pos_0_x, pos_0_y] = 0
        ret = self.get_obs(self.dir[1], pos_1_x, pos_1_y, inv_s)
        # print('In obs_1: ret.shape, right one', ret.shape, 
        #       (3, self.pars['size_obs_ahead']+1, 2*self.pars['size_obs_side']+1))
        assert(ret.shape == (3, self.pars['size_obs_ahead']+1, 2*self.pars['size_obs_side']+1))
        return ret
    
    #
    # transition functions
    #
    def update_dir(self,pl,a):
        """Update the direction of player pl after action a. If still, do nothing
        Use roots of unity coding of positions.
        TESTED
        """
        assert(pl==0 or pl==1)
        assert(a in self.action_cat['rotation_or_still'])
        if a == self.actions_dict['rotate_left']:
            self.dir[pl] = (self.dir[pl] + 1) % 4
        elif a == self.actions_dict['rotate_right']:
            self.dir[pl] = (self.dir[pl] - 1) % 4
    
    def opponent(self,pl):
        """Returns opponent of player pl"""
        assert(pl==0 or pl==1)
        opp = 1 if pl == 0 else 0 
        return opp
    
    def beam_grid(self,pl,pos_pl_x,pos_pl_y):
        """Compute the grid of positions in front of player pl and check if hits opponent.
        In that case, check tagged and remove. Returns grid"""
        assert(pl==0 or pl==1)
        if self.dir[pl] == self.dir_dict['up']:
            grid_hit = np.mgrid[pos_pl_x:pos_pl_x+1,pos_pl_y+1:self.pars['H']].squeeze()
        elif self.dir[pl] == self.dir_dict['down']:
            grid_hit = np.mgrid[pos_pl_x:pos_pl_x+1,0:pos_pl_y].squeeze()
        elif self.dir[pl] == self.dir_dict['left']:
            grid_hit = np.mgrid[0:pos_pl_x,pos_pl_y:pos_pl_y+1].squeeze()
        elif self.dir[pl] == self.dir_dict['right']:
            grid_hit = np.mgrid[pos_pl_x+1:self.pars['W'],pos_pl_y:pos_pl_y+1].squeeze()
        gr_xs = grid_hit[0]; gr_ys = grid_hit[1]
#        print('sh gr',gr_xs.shape,gr_ys.shape)
        # if just one point, reshape so that check below works
        if gr_xs.shape == ():
            gr_xs=gr_xs.reshape((1,1))
        if gr_ys.shape == ():
            gr_ys=gr_ys.reshape((1,1))
        # if no points in the grid (eg at boundary), just skip check
        if gr_xs.size == 0 and gr_ys.size == 0:
            return gr_xs,gr_ys
        # check if opponent is in there
        opp = self.opponent(pl)
        [pos_opp_x,pos_opp_y] = self.get_pos_pl(opp,self.s)
#        print('pos_pl',[pos_pl_x,pos_pl_y],'pos_opp',[pos_opp_x,pos_opp_y],'gr_xs,gr_ys', gr_xs,gr_ys)
        if pos_opp_x in gr_xs and pos_opp_y in gr_ys:
            print('In use_beam: opponent hit!')
            if self.is_tagged[opp]:
                # remove from the game and init t_tagged[opp]
                self.s[self.channel_pl[opp],pos_opp_x,pos_opp_y] = 0
                self.t_tagged[opp] = 0
            else:
                self.is_tagged[opp] = True
        # returns grid
        return gr_xs, gr_ys
    
    def use_beam(self, pl_a, pl_b=None):
        """ if pos_opp is in the beam, check if opp has to be removed and update tagged.
            else, do nothing.
        TESTED ?   
        """
        if pl_b == None: # only pl_a has used the beam
            [pos_pl_a_x,pos_pl_a_y] = self.get_pos_pl(pl_a,self.s)
            gr_xs,gr_ys = self.beam_grid(pl_a, pos_pl_a_x, pos_pl_a_y)
            # color hit grid in yellow to show visually the action of the beam.
            # Before that save the status of the screen, since next time, it will be started from there.
            self.prev_s = deepcopy(self.s)
            # one after the other and the current 
            #print(gr_xs,gr_ys)
            sh = max(gr_xs.shape,gr_ys.shape)
            self.s[0,gr_xs,gr_ys]=np.ones(sh)*255
            self.s[1,gr_xs,gr_ys]=np.ones(sh)*255
            self.s[2,gr_xs,gr_ys]=np.zeros(sh)
        else:
            # both use the beam. get positions of pl_a and pl_b
            # before using the beam, so that if one gets hit, the
            # result of the action of its beam on the other is not
            # affected.
            [pos_pl_a_x,pos_pl_a_y] = self.get_pos_pl(pl_a,self.s)
            [pos_pl_b_x,pos_pl_b_y] = self.get_pos_pl(pl_b,self.s)
            gr_a_xs,gr_a_ys = self.beam_grid(pl_a,pos_pl_a_x,pos_pl_a_y)
            gr_b_xs,gr_b_ys = self.beam_grid(pl_b,pos_pl_b_x,pos_pl_b_y)
            # color hit grid in yellow to show visually the action of the beam.
            # Before that save the status of the screen, since next time it will be started from there.
            self.prev_s = deepcopy(self.s)
            # one after other
            sh_a = max(gr_a_xs.shape,gr_a_ys.shape)
            sh_b = max(gr_b_xs.shape,gr_b_ys.shape)
            self.s[0,gr_a_xs,gr_a_ys]=np.ones(sh_a)*255
            self.s[1,gr_a_xs,gr_a_ys]=np.ones(sh_a)*255
            self.s[2,gr_a_xs,gr_a_ys]=np.zeros(sh_a)
            self.s[0,gr_b_xs,gr_b_ys]=np.ones(sh_b)*255
            self.s[1,gr_b_xs,gr_b_ys]=np.ones(sh_b)*255
            self.s[2,gr_b_xs,gr_b_ys]=np.zeros(sh_b)
        # returns nothing
            
    def move_and_update_apples(self, pl, new_pos, cur_pos):
        """ If one gets an apple, apple removed, create a new t_apple. 
        return also the reward which is one only if an apple has been taken.        
        Update values of pixels as well.
        TESTED
        """
        assert(pl==0 or pl==1)
        [new_pos_x,new_pos_y] = new_pos
        [cur_pos_x,cur_pos_y] = cur_pos
        r = 0
        # if new_pos coincides with that of an apple, create the new t_apple, remove the apple and set r=1
        if self.s[1,new_pos_x,new_pos_y] == 255:
            print('In move_and_update_apples, got an apple!')
            self.t_apples.append(0) # add at the end an element whose value is zero.
            self.rem_apples_pos.append([new_pos_x,new_pos_y])
            self.s[1,new_pos_x,new_pos_y] = 0
            r = 1
        # move pl from cur_pos to new_pos
        ch_pl = self.channel_pl[pl]
#        print('in move_and_update_apples, pl', pl, 'pos', cur_pos_x, cur_pos_y)
#        print('s:',self.s[ch_pl,cur_pos_x,cur_pos_y])
        assert(self.s[ch_pl,cur_pos_x,cur_pos_y] == 255)
        self.s[ch_pl,cur_pos_x,cur_pos_y] = 0
        self.s[ch_pl,new_pos_x,new_pos_y] = 255
        return r
        
    def get_new_pos(self, pl, a, cur_pos):
        """Returns the new position after move a. This depends on a and the direction of player pl.
        Using the numerical convention of the init function to assign values to directions and moves,
        it can be easily implemented using trigonometry.
        It already checks if out of the screen and then do nothing.
        TESTED
        """
        assert(a in self.action_cat['movement'])
        assert(pl==0 or pl==1)
        # get current dir
        cur_dir = self.dir[pl]
        x,y = cur_pos[0],cur_pos[1]
#        print('in get_new_pos, pl',pl,'cur_pos',x,y)
#        print('cur_dir,a',cur_dir,a)
        new_x = int(round(x + np.cos(np.pi*(cur_dir+a)/2.)))
        new_y = int(round(y + np.sin(np.pi*(cur_dir+a)/2.)))
#        print('new_x,new_y',new_x,x + np.cos(np.pi*(cur_dir+a)/2.),new_y,y + np.sin(np.pi*(cur_dir+a)/2.))
        # in case point out of box, just do nothing, reassign old position
        if new_x > self.pars['W']-1 or new_x < 0 or new_y > self.pars['H']-1 or new_y < 0:
            print('out of box')
            new_x = x
            new_y = y
        return [new_x,new_y]
        
    def update_pos(self, pl_a,action_a,pl_b=None,action_b=None):
        """Update position of pl_a and pl_b (if is not None).
        Returns rewards for eventual apples in the order of players specified. 
        TESTED
        """
        assert(pl_a == 0 or pl_a == 1)
        if pl_b == None and action_b == None:
            cur_pos_a = self.get_pos_pl(pl_a,self.s)
            new_pos_a = self.get_new_pos(pl_a,action_a,cur_pos_a)
            opp = self.opponent(pl_a)
            pos_opp = self.get_pos_pl(opp,self.s)
            r=0
            if new_pos_a[0] == pos_opp[0] and new_pos_a[1] == pos_opp[1]:
                pass # do nothing.
            else:
                # check if apples
                r = self.move_and_update_apples(pl_a,new_pos_a,cur_pos_a) 
            return r
        else: # both players a and b specified
            ra,rb = 0,0
            cur_pos_a = self.get_pos_pl(pl_a,self.s)
            cur_pos_b = self.get_pos_pl(pl_b,self.s)
            new_pos_a = self.get_new_pos(pl_a,action_a,cur_pos_a)
            new_pos_b = self.get_new_pos(pl_b,action_b,cur_pos_b)
            if new_pos_a[0] == new_pos_b[0] and new_pos_a[1] == new_pos_b[1]:
                print('collide, do nothing')
                pass # do nothing
            else:
                # check if apples
                ra = self.move_and_update_apples(pl_a,new_pos_a,cur_pos_a) 
                rb = self.move_and_update_apples(pl_b,new_pos_b,cur_pos_b) 
            return ra,rb
    
    def update_time(self):
        """update global time, t_apples and t_tagged"""
        self.global_time += 1
        for i in range(len(self.t_apples)):
            self.t_apples[i] += 1
        for pl in range(len(self.t_tagged)):
            if self.t_tagged[pl] >= 0:
                self.t_tagged[pl] += 1
        # returns nothing
    
    def update_status_apples(self):
        """check if some apple has to respawn. 
        if yes, delete the corresponding element of t_apples and place the new
        apple on screen."""
        print('In update_status_apples: t_apples',self.t_apples)
        new_rem_pos = []
        new_t_apples = []
        for t,pos in zip(self.t_apples,self.rem_apples_pos):
            if t>self.pars['N_apples']:
                x = pos[0] 
                y = pos[1]
                print('Try to respawn apple at', pos, "values of screen",self.s[:,x,y],
                      'comparison',all(self.s[:,x,y] == np.zeros(3)))
                if all(self.s[:,x,y] == np.zeros(3)):
                    # respawn the apple:
                    self.s[1,x,y] = 255
                else: 
                    # wait, put back this apple to the list of apples removed and t_apples
                    new_rem_pos.append(pos)
                    new_t_apples.append(t)
            else:
                new_rem_pos.append(pos)
                new_t_apples.append(t)        
        # assign updated lists
        self.t_apples = new_t_apples
        self.rem_apples_pos = new_rem_pos
# OLD
#         for pos in pos_to_be_respawned:
#             x = pos[0] 
#             y = pos[1]
#             print('Try to respawn apple at', pos, "values of screen",self.s[:,x,y],
#                   'comparison',any(self.s[:,x,y] != np.zeros(3)))            
#             # check if position is free, otherwise try a random in a window around apples
#             window_h = 3
#             window_w = 3
#             while any(self.s[:,x,y] != np.zeros(3)):
#                 x = self.mid_x + int(np.random.random()*window_w)
#                 y = self.mid_y + int(np.random.random()*window_h)
#                 print('not free, try', x,y)
#             self.s[1,x,y] = 255
    
    def update_tagged(self):
        """check if some player is still tagged and in case its t_tagged > N_tagged, put it back into game
        at init position."""
        print('In update_tagged: t_tagged',self.t_tagged)
        for pl in range(len(self.t_tagged)):
            if self.t_tagged[pl] > self.pars['N_tagged']:
                # put player pl back if position not yet occupied and init t_tagged, is_tagged
                self.is_tagged[pl] = False
                self.t_tagged[pl] = -1
                if self.s[self.channel_pl[self.opponent(pl)],self.init_pos_x[pl],self.init_pos_y[pl]] != 255:
                    self.s[self.channel_pl[pl],self.init_pos_x[pl],self.init_pos_y[pl]] = 255
                else:
                    self.s[self.channel_pl[pl],self.init_pos_x[pl],self.init_pos_y[pl]+1] = 255
                    
    def act_freely(self,pl,a):
        """Player pl acts freely with action a since the opponent is out of the game.
        Returns reward."""
        assert(pl==0 or pl==1)
        r = 0
        if a in self.action_cat['movement']:
            r = self.update_pos(pl,a) # this works ok if opp has position -1,-1
        elif a in self.action_cat['rotation_or_still']:
            self.update_dir(pl,a) # also works ok if opp has position -1,-1
        elif self.action_cat['beam']:
            self.use_beam(pl) # this also works of if opp has position -1,-1
        return r
    
    def transition_and_get_reward(self, a0, a1):
        """
        The transition function which given state of system s and two actions a0, a1. It updates
        the state self.s and returns the rewards r0,r1, which are 1 only if apple removed from map.
        
        We proceed as follows: we group actions in 3 groups: movement, rotate_or_still, beam and check
        which combinations occur case by case.
        
        Players act simultaneously
        """
        # update time counter
        self.update_time()
        
        # remove the yellow beam
        if self.prev_s is not None:
            self.s = deepcopy(self.prev_s)
            self.prev_s = None
        
        #
        self.update_status_apples()
        
        #
        self.update_tagged()
        
        pl_0 = 0; pl_1 = 1
        r0 = 0; r1 = 0 # init reward
        if self.t_tagged[pl_0] >= 0:
            # pl_1 opponent acts freely
            r1 = self.act_freely(pl_1,a1)
        elif self.t_tagged[pl_1] >= 0:
            # pl_0 acts freely
            r0 = self.act_freely(pl_0,a0)
        else:
            # all actions need to be considered.
            # implement massive case-by-case. Not elegant but clear for 1st implementation.
            if a0 in self.action_cat['movement'] and a1 in self.action_cat['movement']:
                print('a0,a1 mov')
                r0,r1 = self.update_pos(0,a0,1,a1)
            elif a0 in self.action_cat['movement'] and a1 in self.action_cat['rotation_or_still']:
                print('a0 mov,a1 rot')
                r0 = self.update_pos(0,a0)
                self.update_dir(1,a1)
            elif a0 in self.action_cat['movement'] and a1 in self.action_cat['beam']:
                print('a0 mov,a1 beam')
                r0 = self.update_pos(0,a0)
                self.use_beam(1)
            elif a0 in self.action_cat['rotation_or_still'] and a1 in self.action_cat['movement']:
                print('a0 rot,a1 mov')
                r1 = self.update_pos(1,a1)
                self.update_dir(0,a0)            
            elif a0 in self.action_cat['rotation_or_still'] and a1 in self.action_cat['rotation_or_still']:
                print('a0 rot,a1 rot')
                self.update_dir(0,a0)            
                self.update_dir(1,a1)
            elif a0 in self.action_cat['rotation_or_still'] and a1 in self.action_cat['beam']:
                print('a0 rot,a1 beam')
                self.update_dir(0,a0)
                self.use_beam(1)
            elif a0 in self.action_cat['beam'] and a1 in self.action_cat['movement']:
                print('a0 beam,a1 mov')
                r1 = self.update_pos(1,a1)
                self.use_beam(0)
            elif a0 in self.action_cat['beam'] and a1 in self.action_cat['rotation_or_still']:
                print('a0 beam,a1 rot')
                self.update_dir(1,a1)
                self.use_beam(0)
            elif a0 in self.action_cat['beam'] and a1 in self.action_cat['beam']:
                print('a0 beam,a1 beam')
                self.use_beam(0,1)
            
        return r0, r1
    
    #
    # Output functions:
    #
    def show_screen(self, show=False):
        """Shows current state s using matplotlib

        """
        to_show = self.s.transpose((2,1,0))
        plt.imshow(to_show,origin='lower')
        if show:
            plt.show()
        for k,v in self.dir_dict.items():
            if self.dir[0] == v:
                print('Direction 0:',k)
            if self.dir[1] == v:
                print('Direction 1:',k)

    def get_key_action(self, a):
        """Returns the key associated to action a (a number between 0 and 7)

        """
        for k,v in self.actions_dict.items():
            if a == v:
                return k
        return None # If not found

    # str? 
