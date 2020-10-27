from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import statistics
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import pickle
import numpy as np


class QL:
    def __init__(self, alpha = 0.1, gamma = 1, epsilon = 0.1):
        self.action = [0,1]
        self.flap_vector = np.array([1,1,1,1])
        self.noop_vector = np.array([1,1,1,1])
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = (1-(epsilon/2))*100

                            
    def update_q_reward(self, s1, a, s2, r):
        if a == 0:
            q= self.alpha*( r+ self.gamma*self.max_next(s2) - np.inner(s1, self.flap_vector) )
            self.flap_vector = self.flap_vector + q * s1
        else:
            q =self.alpha*( r+ self.gamma*self.max_next(s2) - np.inner(s1, self.noop_vector) )
            self.noop_vector = self.noop_vector + q * s1
        

        # self.state_action_q_dict[s1][a] = self.state_action_q_dict[s1][a] + self.alpha*(r + self.gamma*max(self.state_action_q_dict[s2]) - self.state_action_q_dict[s1][a])
        
    def max_next(self, s):
        flap = np.inner(self.flap_vector, s)
        noop = np.inner(self.noop_vector, s)
        if flap > noop:
            return flap
        return noop

    def get_action(self, s1):
        flap = np.dot(self.flap_vector, s1)
        noop = np.dot(self.noop_vector, s1)
        if flap > noop:
            if random.randint(1,100) > self.epsilon:
                return 1
            return 0
        if random.randint(1,100) > self.epsilon:
            return 0
        return 1
    
    def get_policy(self, s1):
        flap = np.dot(self.flap_vector, s)
        noop = np.dot(self.noop_vector, s)
        if flap > noop:
            return flap
        return noop


class FlappyAgent:
    def __init__(self):
        self.results = []
        self.discountFactor = 0.1
        self.QL = QL()
        self.actions = [0,1]
        self.curr_epi = dict()
        self.score = 0

    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    
    def observe(self, s1, a, r, s2, end):
        """ this function is called during training on each step of the game where
            the state transition is going from state s1 with action a to state s2 and
            yields the reward r. If s2 is a terminal state, end==True, otherwise end==False.
            
            Unless a terminal state was reached, two subsequent calls to /observe will be for
            subsequent steps in the same episode. That is, s1 in the second call will be s2
            from the first call.
            """
        #(next_pipe_top_y, next_pipe_dist_to_player, player_y, player_vel, action)
        if end:
            s2 = "terminal"
        self.QL.update_q_reward(s1, a, s2, r)
        return #ok

    def state_binner(self, state):
        """splits the y-postion of the bird, y postion of the next gap and horizontal distanze between bird and pipe into 15 bins."""
        dist_bin =  int(state["next_pipe_dist_to_player"])
        player_bin = int(state["player_y"])
        pipe_bin = int(state["next_pipe_top_y"])
        vel = int(state["player_vel"])
        
        binned_state = np.array([pipe_bin,  dist_bin, player_bin, vel])
        return binned_state 

    def training_policy(self, s1):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        #print("state: %s" % (s1,))
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.


        return self.QL.get_action(s1)


    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        #print("state: %s" % state)
        # TODO: 
        return self.QL.get_policy(state) 
    

def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    # TODO: when training use the following instead:
    # reward_values = agent.reward_values
    
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True 
    env.init()
    totalscore = 0
    count = nb_episodes
    score = 0
    
    while nb_episodes > 0:
        # pick an action
        # TODO: for training using agent.training_policy instead
        action = agent.policy(agent.state_binner(env.game.getGameState()))

        # step the environment
        reward = env.act(env.getActionSet()[action])
        #print("reward=%d" % reward)

        # TODO: for training let the agent observe the current state transition

        score += reward
    
        # reset the environment if the game is over
        if env.game_over():
            totalscore += score
            print(count)
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0
    print("average for this run is :%d" % (totalscore/count))
    

def train(nb_frames, agent):
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    env.init()

    score = 0
    biggest_score = -5
    avg_score = 0
    avrage = []
    count = []
    number_of_frames = 0
    nb_episodes =0
    while number_of_frames <nb_frames:
        # pick an action
        state = env.game.getGameState()
        state = agent.state_binner(state)
        action = agent.training_policy(state)

        # step the environment
        reward = env.act(env.getActionSet()[action])
        #print("reward=%d" % reward)

        # let the agent observe the current state transition
        newState = env.game.getGameState()
        newState = agent.state_binner(newState)
        agent.observe(state, action, reward, newState, env.game_over())
        
        
        score += reward
        number_of_frames+= 1
        # reset the environment if the game is over
        if env.game_over():
            nb_episodes += 1
            avg_score += score
            if score > biggest_score:
                biggest_score = score
                print(biggest_score)
                print(nb_episodes)
                print(number_of_frames)
            if nb_episodes %100 == 0:
                print(avg_score/100)
                avrage.append(avg_score/100)
                count.append(number_of_frames)
                avg_score = 0
               

            #print("score for this episode: %d" % score)
            env.reset_game()
            
            
            score = 0
            
    data = {"Count":count, "Avrage":avrage}
    df = pd.DataFrame(data)

    sns.relplot(x="Count", y="Avrage", ci=None, kind="line", data=df)
    # sns.displot(df, x="Count",y="Avrage",kind="ecdf")
    
        
            
    print(biggest_score)



agent = FlappyAgent()
train(20000000, agent)
run_game(70, agent)
# pickle.dump(agent, open('QL.txt',"wb"))
# plt.show()