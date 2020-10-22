from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import statistics

class OPMCC:
    def __init__(self, epsilon = 0.1):
        self.action = [0,1]
        self.action_reward_dict = dict()
        self.state_action_count = dict()
        self.pi = 1-epsilon+epsilon/len(self.action)
        for i in range(15):
            for j in range(15):
                for k in range(15):
                    for l in range(-8,11):
                        for m in self.action:
                            new_state = (i, j, k, l, m)
                            # (next_pipe_top_y, next_pipe_dist_to_player, player_y, player_vel, action)
                            self.action_reward_dict[new_state] = 0
                            self.state_action_count[new_state] = 0
                            

                            
    def add_to_list(self, deli, score):
        self.state_action_count[deli]  += 1
        self.action_reward_dict[deli] += (score - self.action_reward_dict[deli])/self.state_action_count[deli] 

    def get_action(self, s1):


        test1 = s1 + (0,)
        test2 = s1 + (1,)
       
        if self.action_reward_dict[test1] > self.action_reward_dict[test2]:
            if random.randint(1,100)>self.pi*100:
                return 1
            else:
                return 0
        elif self.action_reward_dict[test1] < self.action_reward_dict[test2]:
            if random.randint(1,100)>self.pi*100:
                return 0
            else:
                return 1
        else:
            if random.randint(1,100)>50:
                return 1
            else:
                return 0
    
    def get_policy(self, s1):
        test1 = s1 + (0,)
        test2 = s1 + (1,)
        if self.action_reward_dict[test1] > self.action_reward_dict[test2]:
            return 0
        elif self.action_reward_dict[test1] < self.action_reward_dict[test2]:
            return 1
        else:
            if random.randint(1,100)>50:
                return 1
            else:
                return 0


class FlappyAgent:
    def __init__(self):
        self.results = []
        self.discountFactor = 0.1
        self.opmcc = OPMCC()
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
        
        s1_tup = s1 + (a,)
        if s1_tup not in self.curr_epi.keys():
            self.curr_epi[s1_tup] = self.score
        self.score += r 
        return #ok

    def state_binner(self, state):
        """splits the y-postion of the bird, y postion of the next gap and horizontal distanze between bird and pipe into 15 bins."""
        dist_bin =  int(state["next_pipe_dist_to_player"]/9.6)
        if dist_bin >14:
            dist_bin = 14
        player_bin = int(state["player_y"]/25.46)
        if player_bin >14:
            player_bin = 14
        pipe_bin = int(state["next_pipe_top_y"]/12.84)
        if pipe_bin > 14:
            pipe_bin = 14
        vel = state["player_vel"]
        if vel < -8:
            vel = -8
        binned_state = (pipe_bin,  dist_bin, player_bin, vel)
        return binned_state 

    def training_policy(self, s1):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
        #print("state: %s" % (s1,))
        # TODO: change this to to policy the agent is supposed to use while training
        # At the moment we just return an action uniformly at random.


        return self.opmcc.get_action(s1)


    def policy(self, state):
        """ Returns the index of the action that should be done in state when training is completed.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            policy is called once per frame in the game (30 times per second in real-time)
            and needs to be sufficiently fast to not slow down the game.
        """
        #print("state: %s" % state)
        # TODO: 
        return self.opmcc.get_policy(state) 
    
    def calculate(self):
        for state, minus in self.curr_epi.items():
            self.opmcc.add_to_list(state, self.score-self.curr_epi[state])
        self.curr_epi = dict()

def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    # TODO: when training use the following instead:
    # reward_values = agent.reward_values
    
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=False, rng=None,
            reward_values = reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True 
    env.init()

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
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0
    

def train(nb_episodes, agent):
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    env.init()

    score = 0
    biggest_score = -5
    avg_score = 0
    while nb_episodes > 0:
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
        
        # reset the environment if the game is over
        if env.game_over():
            avg_score += score
            if score > biggest_score:
                biggest_score = score
                print(biggest_score)
                print(nb_episodes)
            if nb_episodes %100 == 0:
                print(avg_score/100)
                avg_score = 0

            #print("score for this episode: %d" % score)
            agent.calculate()
            env.reset_game()
            
            nb_episodes -= 1
            score = 0
    print(biggest_score)



agent = FlappyAgent()
train(500000, agent)
run_game(70, agent)
