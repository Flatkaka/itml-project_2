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
        for i in range(8): #height
            for j in range(16): #width
                for l in range(-8,11):
                    for h in range(4):
                        new_state = (i, j, l, h)
                # (next_pipe_top_y, next_pipe_dist_to_player, player_y, player_vel, action)
                        self.state_action_q_dict[new_state] = [0,0]
        
        self.state_action_q_dict["terminal"] = [0]

                            
    def update_q_reward(self, s1, a, s2, r):
        if not self.state_action_q_dict[s1]:
            self.state_action_q_dict[s1] = [0,0]
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
        self.train_high = -6

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
    
    def increment_frames(self):
        self.frames  += 1

    def state_binner(self, state, extra_reward = 0):
        """splits the y-postion of the bird, y postion of the next gap and horizontal distanze between bird and pipe into 15 bins."""
        return_reward = 0
        vel = (state["player_vel"] + 8)//4
        if vel < 0:
            vel = 0
        diff_bin =  int(state["next_pipe_top_y"]-state["player_y"])
        if diff_bin < -125:
            diff_bin = 0
        elif diff_bin >25:
            diff_bin = 1
        else:
            diff_bin += 125
            diff_bin /= 30
            diff_bin = int(diff_bin)+2
        if diff_bin == 0 or diff_bin == 1:
            binned_state = (diff_bin, 0,vel,0)
        else:
            if diff_bin == 4:
                return_reward = extra_reward
            pipe_bin = int(state["next_pipe_dist_to_player"]/10)+1
            if pipe_bin > 15:
                pipe_bin = 15
            
            next_next = int((state["next_next_pipe_top_y"]-state["next_pipe_top_y"]))
            if next_next < 30 and next_next > -30:
                next_next = 1
            elif next_next>= 30:
                next_next = 2
            else:
                next_next = 3
            binned_state = (diff_bin, pipe_bin,vel, next_next)
        return binned_state, return_reward

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
    
    def get_frames(self):
        return self.frames
    
    def get_train_high(self):
        return self.train_high
    
    def set_train_high(self, score):
        self.train_high = score

def run_game(nb_episodes, agent):
    """ Runs nb_episodes episodes of the game with agent picking the moves.
        An episode of FlappyBird ends with the bird crashing into a pipe or going off screen.
    """

    reward_values = {"positive": 1.0, "negative": 0.0, "tick": 0.0, "loss": 0.0, "win": 0.0}
    # TODO: when training use the following instead:
    # reward_values = agent.reward_values
    
    env = PLE(FlappyBird(), fps=30, display_screen=True, force_fps=True, rng=None,
            reward_values = reward_values)
    # TODO: to speed up training change parameters of PLE as follows:
    # display_screen=False, force_fps=True 
    env.init()

    score = 0
    tot_nb_episodes = nb_episodes
    average = 0
    highscore = 0
    over_50_count = 0
    while nb_episodes > 0:
        # pick an action
        # TODO: for training using agent.training_policy instead
        state, ignore = agent.state_binner(env.game.getGameState())
        action = agent.policy(state)

        # step the environment
        reward = env.act(env.getActionSet()[action])
        #print("reward=%d" % reward)

        # TODO: for training let the agent observe the current state transition

        score += reward

        # reset the environment if the game is over
        if env.game_over() or score >= 60:
            average += score
            if score > highscore:
                highscore = score
            if score >= 50:
                over_50_count += 1
            print("score for this episode: %d" % score)
            env.reset_game()
            nb_episodes -= 1
            score = 0
    print("Average for {} runs {:.2f}".format(tot_nb_episodes, average/tot_nb_episodes))
    over_50_p = (over_50_count/tot_nb_episodes)*100
    print("The percentage of scores over 50 is: %d" % (over_50_p))
    return over_50_p
    

def train(nb_episodes, agent, stop_frame):
    reward_values = agent.reward_values()
    
    env = PLE(FlappyBird(), fps=30, display_screen=False, force_fps=True, rng=None,
            reward_values = reward_values)
    env.init()

    score = 0
    biggest_score = -50000
    avg_score = 0
    episodes = 0
    frames = 0
    break_bool = False
    while nb_episodes > 0:
        
        # pick an action
        state = env.game.getGameState()

        state, extra_reward = agent.state_binner(state, 0.1)
        action = agent.training_policy(state)

        # step the environment
        reward = env.act(env.getActionSet()[action])
        #print("reward=%d" % reward)

        # let the agent observe the current state transition
        newState = env.game.getGameState()
        newState, ignore = agent.state_binner(newState)
        reward_extra = reward + extra_reward
        agent.observe(state, action, reward_extra, newState, env.game_over())
        agent.increment_frames()
        score += reward
        if (agent.get_frames() % stop_frame == 0):
            break_bool = True
        if(agent.get_frames() == 1000000):
            break
      
        # reset the environment if the game is over
        if env.game_over():
            avg_score += score
            if score > agent.get_train_high():
                agent.set_train_high(score)
                if biggest_score > 450:
                    break
                print("New highscore {}".format(agent.get_train_high()))
                print("Frames {}".format(agent.get_frames()))
            if nb_episodes %100 == 0:
                print("New average {}".format(avg_score/100))
                print("Frames {}".format(agent.get_frames()))
                if avg_score/100 >= 4:
                    break
                avg_score = 0
            if break_bool:
                break
                

            #print("score for this episode: %d" % score)
            env.reset_game()
            
            nb_episodes -= 1
            score = 0
            
    return biggest_score



# agent = FlappyAgent()
# i = 0
# while True:
#     train(20000, agent)
#     avg = run_game(100, agent)
#     if avg >= 90:
#         break
# pickle.dump(agent, open('opmc.txt',"wb"))
# while i < 10:
#     avg = run_game(100, agent)
#     i+=1

# agent = pickle.load(open('opmc.txt',"rb"))
agent = FlappyAgent()
train(20000, agent, 200000)
pickle.dump(agent, open('opmc.txt',"wb"))
run_game(100, agent)
    



