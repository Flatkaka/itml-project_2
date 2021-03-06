from ple.games.flappybird import FlappyBird
from ple import PLE
import random
import statistics
import pickle

class QL:
    def __init__(self, alpha = 0.1, gamma = 0.9, epsilon = 0.1):
        self.action = [0,1]
        self.state_action_q_dict = dict()
        self.state_action_count = dict()
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        for i in range(20):
            for j in range(15):
                for k in range(15):
                    for l in range(-8,11):
                        new_state = (i, j, k, l)
                    # (next_pipe_top_y, next_pipe_dist_to_player, player_y, player_vel, action)
                        self.state_action_q_dict[new_state] = [0,0]
        self.state_action_q_dict["terminal"] = [0]

                            
    def update_q_reward(self, s1, a, s2, r):
        self.state_action_q_dict[s1][a] = self.state_action_q_dict[s1][a] + self.alpha*(r + self.gamma*max(self.state_action_q_dict[s2]) - self.state_action_q_dict[s1][a])
        
    def get_action(self, s1):
        if self.state_action_q_dict[s1][0] >self.state_action_q_dict[s1][1]:
            if random.randint(1,100)>(1 -(self.epsilon/2)) *100:
               return 1
            else:
                return 0
            
        elif self.state_action_q_dict[s1][0] <self.state_action_q_dict[s1][1]:
            if random.randint(1,100)>(1 -(self.epsilon/2)) *100:
                return 0
            else:
                return 1
        else:
            if random.randint(1,100)>50:
                return 1
            else:
                return 0
    
    def get_policy(self, s1):

        if self.state_action_q_dict[s1][0] >self.state_action_q_dict[s1][1]:
            return 0
        elif self.state_action_q_dict[s1][0] <self.state_action_q_dict[s1][1]:
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
        self.QL = QL()
        self.actions = [0,1]
        self.curr_epi = dict()
        self.score = 0
        self.frames = 0

    def reward_values(self):
        """ returns the reward values used for training
        
            Note: These are only the rewards used for training.
            The rewards used for evaluating the agent will always be
            1 for passing through each pipe and 0 for all other state
            transitions.
        """
        return {"positive": 1.0, "tick": 0.0, "loss": -5.0}
    
    def change_eps(self):
        self.QL.epsilon /= 2
    
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
        
        dist_bin =  int(state["next_pipe_dist_to_player"]/9.6)
        if dist_bin>=14:
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
        diff_bin = pipe_bin - player_bin
        binned_state = (pipe_bin,  dist_bin, player_bin, vel)
        return binned_state 

    def training_policy(self, s1):
        """ Returns the index of the action that should be done in state while training the agent.
            Possible actions in Flappy Bird are 0 (flap the wing) or 1 (do nothing).

            training_policy is called once per frame in the game while training
        """
  
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

    score = 0
    tot_nb_episodes = nb_episodes
    average = 0
    highscore = 0
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
            average += score
            if score > highscore:
                highscore = score
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0
    print("Average for 100 runs {}".format(average/tot_nb_episodes))
    return highscore
    

def train(nb_episodes, agent):
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    env.init()

    score = 0
    biggest_score = -50000
    avg_score = 0
    episodes = 0
    to_break = False
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
        agent.frames += 1
        score += reward
        
        if ((agent.frames %10000) == 0):
            to_break = True
        # reset the environment if the game is over
        if env.game_over():
            avg_score += score
            if score > biggest_score:
                biggest_score = score
                if biggest_score > 450:
                    break
                print(biggest_score)
                print(nb_episodes)
            if nb_episodes %100 == 0:
                print(avg_score/100)
                if avg_score/100 >= 5:
                    break
                avg_score = 0
            if to_break:
                break
                

            #print("score for this episode: %d" % score)
            env.reset_game()
            
            nb_episodes -= 1
            score = 0
            
    return biggest_score



agent = FlappyAgent()
while (agent.frames < 1000000):
    biggest_train_score = train(30000, agent)
    avg_run_game = run_game(100, agent)
    if abs(biggest_train_score+5-avg_run_game)>10:
        agent.change_eps()
pickle.dump(agent, open("flott.txt", "wb"))
