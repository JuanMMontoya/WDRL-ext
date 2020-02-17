"""
Author: Juan M. Montoya
Class structure based on PacmanDQN_Agents.py

The Pacman AI projects were developed at UC Berkeley found at
http://ai.berkeley.edu/project_overview.html

This new version integrates the memory replay into the data flow.
Thus, not saving it into the disk.
In addition, added Rank-based Prioritized Experience Replay and shift option.
The shift option permits to turn the wide component off and on again.
"""

from util import *

# Pacman Game
from game import Agent
from pacman import GameState
from stable_baselines.common.schedules import LinearSchedule
from stable_baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer
from rankBasedReplay import RankBasedReplay
import pickle
# Neural nets
import tensorflow as tf
from WDQN import WDQN
import os


class PacmanPWDQN(Agent):
    """
        Creates the Wide Deep Q-Network Agent that iterates with the environment
        In addition, this agent can be set up to purely Linear or DQN Agent
        """

    def __init__(self, args):
        # Load parameters from user-given arguments

        self.params = json_to_dict(args["path"])
        os.environ["CUDA_VISIBLE_DEVICES"] = str(self.params["GPU"])
        self.params['width'] = args['width']
        self.params['height'] = args['height']
        self.params['num_training'] = args['numTraining']
        self.params['num_games'] = args['numGames']
        self.path_extra = ""
        self.params["seed"] = args['seed']
        self.random = np.random.RandomState(self.params["seed"])
        self.beta_schedule = None
        # time started
        self.general_record_time = time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime())
        self.start_time = time.time()
        self.rank_sort = None

        if self.params["prioritized"]: # For using PrioritizedReplayBuffer
            if self.params["ranked"]:
                N_list = [self.params["batch_size"]] + [int(x) for x in np.linspace(100, self.params["mem_size"], 5)]
                save_quantiles(N_list=N_list, k=self.params["batch_size"],
                               alpha=self.params["prioritized_replay_alpha"], name=self.params["save_file"])
                self.replay_buffer = RankBasedReplay(self.params["mem_size"],
                                                     self.params["prioritized_replay_alpha"],
                                                     name=self.params["save_file"])
                if self.params["sort_rank"] == None:  # For sorting rankbased buffer
                    self.rank_sort = int(self.params["mem_size"] * 0.01)
                else:
                    self.rank_sort = self.params["sort_rank"]
            else:
                self.replay_buffer = PrioritizedReplayBuffer(self.params["mem_size"],
                                                             self.params["prioritized_replay_alpha"])
            if self.params["prioritized_replay_beta_iters"] is None:
                prioritized_replay_beta_iters = self.params['num_training']
            else:
                prioritized_replay_beta_iters = self.params['prioritized_replay_beta_iters']

            self.beta_schedule = LinearSchedule(prioritized_replay_beta_iters,
                                                initial_p=self.params['prioritized_replay_beta0'],
                                                final_p=1.0)
        else:
            self.replay_buffer = ReplayBuffer(self.params["mem_size"])
            self.beta_schedule = None

        if self.params["only_dqn"]:
            print("Initialise DQN Agent")
        elif self.params["only_lin"]:
            print("Initialise Linear Approximative Agent")
        else:
            print("Initialise WDQN Agent")

        print(self.params["save_file"])
        if self.params["prioritized"]:
            if self.params["ranked"]:
                print("Using Rank-Based Experience Replay Buffer")
            else:
                print("Using Prioritized Experience Replay Buffer")

        if self.params["model_shift"]:
            print("Using Model Shift")

        print("seed", self.params["seed"])
        print("Starting time:", self.general_record_time)

        # Start Tensorflow session
        tf.reset_default_graph()
        tf.set_random_seed(self.params["seed"])
        self.qnet = WDQN(self.params, "model")  # Q-network
        self.tnet = WDQN(self.params, "target_model")  # Q-target-network
        self.saver = tf.train.Saver()
        self.sess = tf.Session()
        self.qnet.set_session(self.sess)
        self.tnet.set_session(self.sess)
        self.sess.run(tf.global_variables_initializer())

        # Q and cost
        self.Q_global = []

        # Stats
        self.cnt = self.qnet.sess.run(self.qnet.global_step_dqn)
        self.local_cnt = 0
        self.wins = 0
        self.best_int = self.params["shift_best"]

        self.numeps = 0
        self.model_eps = 0
        self.episodeStartTime = time.time()
        self.last_steps = 0

        self.get_direction = lambda k: ['North', 'South', 'East', 'West', 'Stop'][k]
        self.get_value = {'North': 0, 'South': 1, 'East': 2, 'West': 3, 'Stop': 4}

        self.lastWindowAccumRewards = 0.0
        self.Q_accumulative = 0.0
        self.accumTrainRewards = 0.0
        self.sub_dir = str(self.params["save_interval"])

    def registerInitialState(self, state):
        """Inspects the starting state"""
        # Reset reward
        self.last_score = 0
        self.last_reward = 0.

        # Reset state
        self.last_state = None
        self.current_state = state

        # Reset actions
        self.last_action = None

        # Reset vars
        self.terminal = None
        self.won = True
        self.Q_global = []

        # Shift Model between WDQN and DQN during training
        if self.params["model_shift"] and (self.numeps + 1) <= self.params['num_training']:
            if (self.numeps +1) >= self.params["start_shift"] and (self.numeps +1)  % self.params["val_shift"] == 0:
                if self.params["only_dqn"]:
                    self.params["only_dqn"] = False
                    print("Using WDQN Agent starting from eps", (self.numeps + 1))
                else:
                    self.params["only_dqn"] = True
                    print("Using DQN Agent starting from eps", (self.numeps + 1))
            if self.params["model_shift"] and (self.numeps + 1) == self.params['num_training'] and not self.params["only_dqn"]:  # Back to WDQN at the end
                self.params["only_dqn"] = False
                print("Back to WDQN Agent for testing")

        # Load model
        self.load_mod()

        # Next
        self.numeps += 1


    def getQvalues(self, model, dropout):
        """Access Q Values by using the model prediction of WDQN.py"""
        if self.params["only_dqn"]:
            return model.predict_dqn(map_state_mat(self.current_state), dropout)[0]
        elif self.params["only_lin"]:
            return model.predict_lin(mat_features(self.current_state, ftrs=self.params["feat_val"]), dropout)[0]
        else:
            return model.predict_wdqn(map_state_mat(self.current_state),
                                      mat_features(self.current_state, ftrs=self.params["feat_val"]), dropout)[0]

    def getPolicy(self, model, dropout=1.0):
        """Pick up the policy """
        qValues = self.getQvalues(model, dropout)
        qVal = {self.get_value[l]: qValues[self.get_value[l]] for l in self.current_state.getLegalActions(0) if
                not l == "Stop"}
        maxValue = max(qVal.values())
        self.Q_global.append(maxValue)
        return self.get_direction(self.random.choice([k for k in qVal.keys() if qVal[k] == maxValue]))

    def getAction(self, state):
        """Exploit / Explore"""
        if self.random.rand() > self.params['eps']:
            # Exploit action
            move = self.getPolicy(self.qnet)  # dropout deactivated
        else:
            legal = [v for v in state.getLegalActions(0) if not v == "Stop"]
            move = self.random.choice(legal)
        # Save last_action
        self.last_action = self.get_value[move]
        return move

    def observationFunction(self, state):
        """Do observation"""
        self.terminal = False
        self.observation_step(state)
        return state

    def observation_step(self, state):
        """
        Realize the observation step
        Rewards are balanced in this part
        The training occurs in this section
        """
        if self.last_action is not None:
            # Process current experience state
            self.last_state = self.current_state.deepCopy()
            self.current_state = state
            # Process current experience reward
            reward = state.getScore() - self.last_score
            self.last_score = state.getScore()
            # Reward system
            if reward > 20:
                self.last_reward = 50  # 0.1 # Eat ghost
            elif reward > 0:
                self.last_reward = 10  # 0.02 # Eat food
            elif reward < -10:
                self.last_reward = -500.  # -1 # Get eaten
                self.won = False
            elif reward < 0:
                self.last_reward = -1  # -0.002 # Punish time
            if (self.terminal and self.won):
                self.last_reward = 100  # 0.2 # Won

            if self.isInTraining():
                # Copy values to target network
                if self.local_cnt % self.params["target_update_network"] == 0 \
                        and self.local_cnt > self.params['train_start']:
                    self.tnet.rep_network(self.qnet)
                    print("Copied model parameters to target network. total_t = %s, period = %s" % (
                        self.local_cnt, self.params["target_update_network"]))

                # Store last experience into memory
                if self.params["prioritized"] and self.params["ranked"]:
                    self.replay_buffer.add((self.last_state, self.last_action, float(self.last_reward),
                                            self.current_state,
                                            self.terminal))
                else:
                    self.replay_buffer.add(self.last_state, self.last_action, float(self.last_reward),
                                           self.current_state,
                                           self.terminal)
                # Train
                self.train()

                # Next
                self.local_cnt += 1
                if self.local_cnt == self.params['train_start']:
                    print("")
                    print("Memory Replay populated")
                    print("")
                    self.model_eps = self.numeps

                    # with open('data/lin_rb.pickle', 'wb') as handle:
                    #    pickle.dump(self.replay_buffer, handle)
                    #    print("Pickle Saved")
                    #    print(10 + "n")

                self.params['eps'] = max(self.params['eps_final'],
                                         1.00 - float(self.cnt) / float(self.params['eps_step']))

    def train(self):
        """Train different agents: WDQN, DQN and Linear"""
        if self.local_cnt > self.params['train_start']:
            if self.params["only_dqn"]:
                batch_s_dqn, batch_a, batch_t, qt_dqn, batch_r, batch_idxes, weights = extract_batches_per(self.params,
                                                                                                           self.tnet,
                                                                                                           self.replay_buffer,
                                                                                                           self.beta_schedule,
                                                                                                           (self.numeps-self.model_eps))
                self.cnt, td_errors = self.qnet.train(batch_s_dqn, None, batch_a, batch_t, qt_dqn, None, None, batch_r,
                                                      self.params["dropout"],
                                                      self.params["only_dqn"],
                                                      self.params["only_lin"], weights)

            elif self.params["only_lin"]:
                batch_s_lin, batch_a, batch_t, qt_lin, batch_r, batch_idxes, weights = extract_batches_per(self.params,
                                                                                                           self.tnet,
                                                                                                           self.replay_buffer,
                                                                                                           self.beta_schedule,
                                                                                                           (self.numeps-self.model_eps))
                self.cnt, td_errors = self.qnet.train(None, batch_s_lin, batch_a, batch_t, None, qt_lin, None, batch_r,
                                                      self.params["dropout"],
                                                      self.params["only_dqn"],
                                                      self.params["only_lin"], weights)
            else:
                batch_s_dqn, batch_s_lin, batch_a, batch_t, qt_lin, qt_dqn, qt_wdqn, batch_r, batch_idxes, weights = extract_batches_per(
                    self.params, self.tnet,
                    self.replay_buffer, self.beta_schedule, (self.numeps-self.model_eps))
                self.cnt, td_errors = self.qnet.train(batch_s_dqn, batch_s_lin, batch_a, batch_t, qt_dqn, qt_lin,
                                                      qt_wdqn,
                                                      batch_r,
                                                      self.params["dropout"],
                                                      self.params["only_dqn"],
                                                      self.params["only_lin"], weights)

            if self.params["prioritized"]:
                new_priorities = np.abs(td_errors) + self.params["prioritized_replay_eps"]
                self.replay_buffer.update_priorities(batch_idxes, new_priorities)
                if self.params["ranked"] and self.cnt % self.rank_sort == 0:
                    self.replay_buffer.sort()

    def final(self, state):
        """Inspects the last state"""
        # Do observation
        self.terminal = True
        self.observation_step(state)
        NUM_EPS_UPDATE = 100
        self.lastWindowAccumRewards += state.getScore()  #
        self.accumTrainRewards += state.getScore()
        self.Q_accumulative += max(self.Q_global, default=float('nan'))
        self.wins += self.won

        if self.numeps % NUM_EPS_UPDATE == 0:
            # Print stats
            eps_time = time.time() - self.episodeStartTime
            print('Reinforcement Learning Status:')
            if self.numeps <= self.params['num_training']:
                trainAvg = self.accumTrainRewards / float(self.numeps)
                print('\tCompleted %d out of %d training episodes' % (
                    self.numeps, self.params['num_training']))
                print('\tAverage Rewards over all training: %.2f' % (
                    trainAvg))

            windowAvg = self.lastWindowAccumRewards / float(NUM_EPS_UPDATE)
            windowQavg = self.Q_accumulative / float(NUM_EPS_UPDATE)
            window_steps = (self.cnt - self.last_steps) / float(NUM_EPS_UPDATE)
            print('\tAverage Rewards for last %d episodes: %.2f' % (
                NUM_EPS_UPDATE, windowAvg))
            print('\tEpisode took %.2f seconds' % (eps_time))
            print('\tEpisilon is %.8f' % self.params["eps"])
            print('\tLinear Decay learning Rate is %.8f' % self.sess.run(self.qnet.lr_lin))
            if self.params["save_logs"]:
                log_file = open('logs/' + self.params["save_file"] + "-" + str(self.general_record_time) + '-l-' + str(
                    (self.params["num_training"])) + '.log',
                                'a')
                log_file.write("# %4d |  s: %8d | t: %.2f  |  r: %12f | Q: %10f | won: %r \n" %
                               (self.numeps, window_steps, eps_time, windowAvg,
                                windowQavg, self.wins))

            # Save Best Model
            if windowAvg >= self.params["best_thr"]:
                self.params["best_thr"] = windowAvg
                self.last_steps = self.cnt
                self.save_mod(best_mod=True)
                print("Saving the model with:", self.params["best_thr"])
                self.params["best_thr"] = self.params["best_thr"] + self.best_int

            sys.stdout.flush()
            self.lastWindowAccumRewards = 0
            self.Q_accumulative = 0
            self.last_steps = self.cnt
            self.wins = 0
            self.episodeStartTime = time.time()

        if self.numeps >= self.params['num_training']:
            eps_time = time.time() - self.episodeStartTime
            if self.numeps == self.params['num_training']:
                print("Starting Date of Training:", self.general_record_time)
                print("Ending Date of Training:", time.strftime("%a_%d_%b_%Y_%H_%M_%S", time.localtime()))
                print("Training time duration in minutes:", (time.time() - self.start_time) / 60)
                print('Training Done (turning off epsilon)')
            self.params["eps"] = 0.0  # no exploration

            log_file = open(
                'logs/testedModels/' + self.params["save_file"] + '-s-' + str(self.params["seed"]) + '-n-' + str(
                    self.params['num_games'] - self.params["num_training"]) + '.log',
                'a')
            log_file.write("# %4d |  s: %8d | t: %.2f  |  r: %12f | Q: %10f | won: %r \n" %
                           (self.numeps, self.cnt - self.last_steps, eps_time, state.getScore(),
                            max(self.Q_global, default=float('nan'))
                            , int(self.won)))
            self.last_steps = self.cnt
        # save model
        self.save_mod(best_mod=False)

    def isInTraining(self):
        """Check is if agent is in training"""
        return self.numeps < self.params["num_training"]

    def isInTesting(self):
        """Check is if agent is in testing"""
        return not self.isInTraining()

    def load_mod(self):
        """ Load data and model"""
        if self.params["load"]:
            try:
                print("".join([self.path_extra, "model/", self.params["save_file"], "-", self.params["load_file"]]))

                self.saver.restore(self.sess,
                                   "".join([self.path_extra, "model/", self.params["save_file"], "-",
                                            self.params["load_file"]]))
                if not self.params["load_file"].lower() == "best":
                    print("Model Restored")
                else:
                    print("Best Model Restored")
                try:
                    load_path = "".join(
                        [self.path_extra, "parameters/", "params_", self.params["save_file"], "-",
                         self.params["load_file"].lower(), ".npy"])

                    # Parameters to be preserved for params when charging
                    save, save_interval, num_tr, load_data, best_thr, eps_final, dropout, decay_lr, decay_lr_val, only_dqn, only_lin, load_file, num_games, seed, save_logs = \
                        self.params["save"], self.params["save_interval"], \
                        self.params["num_training"], self.params["load_data"], \
                        self.params["best_thr"], self.params["eps_final"], self.params["dropout"], self.params[
                            "dcy_lrl"], \
                        self.params["dcy_lrl_val"], self.params["only_dqn"], self.params["only_lin"], self.params[
                            "load_file"], self.params["num_games"], self.params["seed"], self.params["save_logs"]

                    # Load saved parameters and hyperparameters of the new starting point
                    self.last_steps, self.accumTrainRewards, self.numeps, self.params, self.local_cnt, self.cnt, self.sub_dir = np.load(
                        load_path)  #
                    orig_num_training = self.params["num_training"]
                    # Load newest Parameters to params
                    orig_save_int = self.params["save_interval"]

                    self.params["save"], self.params["save_interval"], self.params["num_training"], \
                    self.params["load_data"], self.params["best_thr"], self.params["eps_final"], self.params["dropout"], \
                    self.params["dcy_lrl"], self.params["dcy_lrl_val"], self.params["only_dqn"], self.params[
                        "only_lin"], self.params["load_file"], \
                    self.params["num_games"], self.params["seed"], self.params["save_logs"] = save, save_interval, \
                                                                                              num_tr, load_data, best_thr, eps_final, \
                                                                                              dropout, decay_lr, decay_lr_val, only_dqn, \
                                                                                              only_lin, load_file, num_games, seed, save_logs

                    if self.sub_dir == "best":
                        print("Best Parameters Restored")
                    else:
                        print("Parameters Restored")

                    if self.params["load_data"]:  # Load data and starts with correct data
                        self.params["num_training"] += orig_num_training
                        if not self.params["load_file"].lower() == "best":
                            src = "".join(
                                [self.path_extra, "data/mem_rep_", self.params["save_file"], "/",
                                 self.params["save_file"], "-", str(self.sub_dir), ".pickle"])

                            print("Interval Data to be Restored")
                        else:
                            src = "".join(
                                [self.path_extra, "data/mem_rep_", self.params["save_file"], "/",
                                 self.params["save_file"], "-", self.sub_dir, ".pickle"])
                            print("Best Data to be Restored")
                        with open(src, 'rb') as handle:
                            self.replay_buffer = pickle.load(handle)
                        print("Data Restored")
                except Exception as e:
                    print(e)
                    print("Parameters don't exist or could not be properly loaded")
            except:
                print("Model don't exist or could not be properly loaded")
            self.params["load"] = False

    def save_mod(self, best_mod=False):
        """
        Saving model and parameters
        Possibility of saving the best model
        """
        if (self.numeps % self.params["save_interval"] == 0 and self.params["save"]) or (
                    best_mod and self.params["save"]):
            self.params["global_step"] = self.cnt
            save_files = [self.last_steps, self.accumTrainRewards, self.numeps, self.params, self.local_cnt, self.cnt]
            try:
                if best_mod:
                    self.saver.save(self.sess,
                                    "".join([self.path_extra, "model/", self.params["save_file"], "-", "best"]))
                    print("Best Model Saved")
                elif not self.sub_dir == "best":
                    self.saver.save(self.sess, "".join(
                        [self.path_extra, "model/", self.params["save_file"], "-", str(self.numeps)]))
                    print("Model Saved")
            except Exception as e:
                print("Model could not be saved")
                print("Error", e)
            try:
                if str(self.numeps) == self.sub_dir:  # Save memory replay and parameters
                    dic = "".join(
                        [self.path_extra, "data/mem_rep_", self.params["save_file"], "/"])
                    f_name = "".join([self.params["save_file"], "-", str(self.sub_dir), ".pickle"])
                    # 1
                    save_files.append(self.sub_dir)

                    np.save("".join(
                        [self.path_extra, "parameters/", "params_", self.params["save_file"], "-", str(self.numeps)]),
                        save_files)
                    print("Pameters Saved")
                    save_rep_buf(self.replay_buffer, dic, f_name)
                    self.sub_dir = str(self.numeps + self.params["save_interval"])
                    print("Memory Replay Saved")
                elif best_mod:  # Save memory replay of best model in directory "best" and parameters
                    dic = "".join(
                        [self.path_extra, "data/mem_rep_", self.params["save_file"], "/"])
                    f_name = "".join([self.params["save_file"], "-", "best", ".pickle"])
                    save_files.append("best")
                    np.save("".join([self.path_extra, "parameters/", "params_", self.params["save_file"], "-", "best"]),
                            save_files)
                    print("Best Pameters Saved")
                    save_rep_buf(self.replay_buffer, dic, f_name)
                    print("Best Memory Replay Saved")
            except Exception as e:
                print("Parameters could not be saved")
                print("Error", e)
