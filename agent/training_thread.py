import logging
import os
import pdb
import random
import signal
import sys
from collections import namedtuple

import h5py
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.functional as F

from agent.environment.ai2thor_file import \
    THORDiscreteEnvironment as THORDiscreteEnvironmentFile
from agent.method.aop import AOP
from agent.method.gcn import GCN
from agent.method.similarity_grid import SimilarityGrid
from agent.method.target_driven import TargetDriven
from agent.network import ActorCriticLoss, SceneSpecificNetwork, SharedNetwork, SceneSpecificNetwork_att2act
from torchvision import transforms


class ForkablePdb(pdb.Pdb):

    _original_stdin_fd = sys.stdin.fileno()
    _original_stdin = None

    def __init__(self):
        pdb.Pdb.__init__(self, nosigint=True)

    def _cmdloop(self):
        current_stdin = sys.stdin
        try:
            if not self._original_stdin:
                self._original_stdin = os.fdopen(self._original_stdin_fd)
            sys.stdin = self._original_stdin
            self.cmdloop()
        finally:
            sys.stdin = current_stdin


TrainingSample = namedtuple('TrainingSample', ('state', 'policy',
                                               'value', 'action_taken', 'goal', 'R', 'temporary_difference'))


class TrainingThread(mp.Process):
    """This thread is an agent, it will explore the world and backpropagate gradient
    """

    def __init__(self,
                 id: int,
                 networks: dict,
                 saver,
                 optimizer,
                 summary_queue: mp.Queue,
                 device,
                 method: str,
                 reward: str,
                 tasks: list,
                 kwargs):
        """TrainingThread constructor

        Arguments:
            id {int} -- UID of the thread
            network {torch.nn.Module} -- Master network shared by all TrainingThread
            saver {[type]} -- saver utils to to save checkpoint
            optimizer {[type]} -- Optimizer to use
            scene {str} -- Name of the current world
            summary_queue {mp.Queue} -- Queue to pass scalar to tensorboard logger
        """

        super(TrainingThread, self).__init__()

        # Initialize the environment
        self.envs = None
        self.init_args = kwargs
        self.saver = saver
        self.id = id
        self.device = device

        self.master_network = networks
        self.optimizer = optimizer

        self.exit = mp.Event()
        self.local_t = 0

        self.summary_queue = summary_queue
        self.method = method
        self.reward = reward
        self.tasks = tasks
        self.scenes = set([scene for (scene, target) in tasks])

    def _sync_network(self, scene):
        if self.init_args['cuda']:
            with torch.cuda.device(self.device):
                state_dict = self.master_network.state_dict()
                self.policy_networks.load_state_dict(state_dict)
        else:
            state_dict = self.master_network.state_dict()
            self.policy_networks.load_state_dict(state_dict)

    def get_action_space_size(self):
        return len(self.envs[0].actions)

    def _initialize_thread(self):
        # Disable OMP
        torch.set_num_threads(1)
        torch.manual_seed(self.init_args['seed'])
        if self.init_args['cuda']:
            torch.cuda.manual_seed(self.init_args['seed'])
        h5_file_path = self.init_args.get('h5_file_path')
        self.logger = logging.getLogger('agent')
        self.logger.setLevel(logging.INFO)
        self.init_args['h5_file_path'] = lambda scene: h5_file_path.replace(
            '{scene}', scene)

        self.mask_size = self.init_args.get('mask_size', 5)

        args = self.init_args
        args.pop("reward")
        args.pop("method")
        self.envs = [THORDiscreteEnvironmentFile(method=self.method,
                                                 reward=self.reward,
                                                 scene_name=scene,
                                                 terminal_state=task,
                                                 **args)
                     for (scene, task) in self.tasks]

        self.gamma: float = self.init_args.get('gamma', 0.99)
        self.grad_norm: float = self.init_args.get('grad_norm', 40.0)
        entropy_beta: float = self.init_args.get('entropy_beta', 0.01)
        self.max_t: int = self.init_args.get('max_t')
        self.local_t = 0
        self.action_space_size = self.get_action_space_size()

        self.criterion = ActorCriticLoss(entropy_beta)

        self.policy_networks = nn.Sequential(SharedNetwork(
            self.method, self.mask_size), SceneSpecificNetwork_att2act(self.get_action_space_size())).to(self.device)

        # Store action for each episode
        self.saved_actions = []
        # Initialize the episode
        for idx, _ in enumerate(self.envs):
            self._reset_episode(idx)
        for scene in self.scenes:
            self._sync_network(scene)

        self.method_class = None
        if self.method == 'word2vec' or self.method == 'word2vec_nosimi' or \
           self.method == 'word2vec_noconv' or self.method == 'word2vec_notarget' or \
           self.method == 'gcn_' or self.method == 'aop_we' or self.method == 'word2vec_notarget_lstm' or \
           self.method == 'word2vec_notarget_lstm_2layer' or self.method == 'word2vec_notarget_lstm_3layer' or \
           self.method == 'word2vec_notarget_rnn' or self.method == 'word2vec_notarget_gru' or self.method == 'ana':        
            self.method_class = SimilarityGrid(self.method)
        elif self.method == 'aop' or self.method == 'aop_we':
            self.method_class = AOP(self.method)
        elif self.method == 'target_driven':
            self.method_class = TargetDriven(self.method)
        elif self.method == 'gcn':
            self.method_class = GCN(self.method)

    def _reset_episode(self, idx):
        self.saved_actions = []
        self.episode_reward = 0
        self.episode_length = 0
        self.episode_max_q = torch.FloatTensor([-np.inf]).to(self.device)
        self.envs[idx].reset()

    def _forward_explore(self, scene, idx):
        # Does the evaluation end naturally?
        is_terminal = False
        terminal_end = False

        results = {"policy": [], "value": []}
        rollout_path = {"state": [], "action": [], "rewards": [], "done": []}

        # Plays out one game to end or max_t
        for t in range(self.max_t):

            policy, value, state = self.method_class.forward_policy(
                self.envs[idx], self.device, self.policy_networks)

            if (self.id == 0) and (self.local_t % 100) == 0:
                print(f'Local Step {self.local_t}')

            # Store raw network output to use in backprop
            results["policy"].append(policy)
            results["value"].append(value)

            with torch.no_grad():
                (_, action,) = policy.max(0)
                action = F.softmax(policy, dim=0).multinomial(1).item()

            policy = policy.data  # .numpy()
            value = value.data  # .numpy()

            # Makes the step in the environment
            self.envs[idx].step(action)

            # Save action for this episode
            self.saved_actions.append(action)

            # ad-hoc reward for navigation
            reward = self.envs[idx].reward

            # Receives the game reward
            is_terminal = self.envs[idx].is_terminal

            # Max episode length
            if self.episode_length > 200:
                is_terminal = True

            # Update episode stats
            self.episode_length += 1
            self.episode_reward += reward
            with torch.no_grad():
                self.episode_max_q = torch.max(
                    self.episode_max_q, torch.max(value))

            # clip reward
            reward = np.clip(reward, -1, 1)

            # Increase local time
            self.local_t += 1

            rollout_path["state"].append(state)
            rollout_path["action"].append(action)
            rollout_path["rewards"].append(reward)
            rollout_path["done"].append(is_terminal)

            # Episode is terminal
            # soft goal: means that the agent emits the done signal
            # other method: agent reach goal position
            if is_terminal:
                (_, task) = self.tasks[idx]
                scene_log = scene + '-' + \
                    str(task['object'])
                step = self.optimizer.get_global_step() * self.max_t

                if self.envs[idx].success:
                    print(
                        f"time {self.optimizer.get_global_step() * self.max_t} | thread #{self.id} | scene {scene} | target #{self.envs[idx].terminal_state['object']}")

                    print(
                        f'playout finished, success : {self.envs[idx].success}')
                    print(
                        f'episode length: {self.episode_length}')

                    # print(f'episode shortest length: {self.envs[idx].shortest_path_distance_start}')
                    print(f'episode reward: {self.episode_reward}')
                    print(
                        f'episode max_q: {self.episode_max_q.detach().cpu().numpy()[0]}')

                    hist_action, _ = np.histogram(
                        self.saved_actions, bins=self.action_space_size, density=False)
                    self.summary_queue.put(
                        (scene_log + '/actions', hist_action, step))

                    # Send info to logger thread
                    self.summary_queue.put(
                        (scene_log + '/episode_length', self.episode_length, step))
                self.summary_queue.put(
                    (scene_log + '/max_q', float(self.episode_max_q.detach().cpu().numpy()[0]), step))
                self.summary_queue.put(
                    (scene_log + '/reward', float(self.episode_reward), step))
                self.summary_queue.put(
                    (scene_log + '/learning_rate', float(self.optimizer.scheduler.get_lr()[0]), step))

                terminal_end = True
                self._reset_episode(idx)
                break

        if terminal_end:
            return 0.0, results, rollout_path, terminal_end
        else:
            policy, value, state = self.method_class.forward_policy(
                self.envs[idx], self.device, self.policy_networks)
            return value.data.item(), results, rollout_path, terminal_end

    def _optimize_path(self, scene, playout_reward: float, results, rollout_path):
        policy_batch = []
        value_batch = []
        action_batch = []
        temporary_difference_batch = []
        playout_reward_batch = []

        for i in reversed(range(len(results["value"]))):
            reward = rollout_path["rewards"][i]
            value = results["value"][i]
            action = rollout_path["action"][i]

            playout_reward = reward + self.gamma * playout_reward
            temporary_difference = playout_reward - value.data.item()

            policy_batch.append(results['policy'][i])
            value_batch.append(results['value'][i])
            action_batch.append(action)
            temporary_difference_batch.append(temporary_difference)
            playout_reward_batch.append(playout_reward)

        policy_batch = torch.stack(policy_batch, 0).to(self.device)
        value_batch = torch.stack(value_batch, 0).to(self.device)
        action_batch = torch.from_numpy(
            np.array(action_batch, dtype=np.int64)).to(self.device)
        temporary_difference_batch = torch.from_numpy(
            np.array(temporary_difference_batch, dtype=np.float32)).to(self.device)
        playout_reward_batch = torch.from_numpy(
            np.array(playout_reward_batch, dtype=np.float32)).to(self.device)

        # Compute loss
        loss = self.criterion.forward(
            policy_batch, value_batch, action_batch, temporary_difference_batch, playout_reward_batch)
        loss = loss.sum()

        # loss_value = loss.detach().numpy()
        self.optimizer.optimize(loss,
                                self.policy_networks,
                                self.master_network,
                                self.init_args['cuda'])

    def run(self, master=None):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        print(f'Thread {self.id} ready')

        # We need to silence all errors on new process
        h5py._errors.silence_errors()
        self._initialize_thread()

        if not master is None:
            print(f'Master thread {self.id} started')
        else:
            print(f'Thread {self.id} started')

        try:
            # random task order
            idx = [j for j in range(len(self.tasks))]
            random.shuffle(idx)
            j = 0

            # reset all env
            self.envs = [env for env in self.envs if env.reset()]
            while not self.exit.is_set() and self.optimizer.get_global_step() * self.max_t < self.init_args["total_step"]:
                # Load current task with scene
                (scene, target) = self.tasks[idx[j]]

                # Change episode if it's a terminal episode (goal reached or max step)
                terminal = False
                while not terminal and not self.exit.is_set() and self.optimizer.get_global_step() * self.max_t < self.init_args["total_step"]:
                    self._sync_network(scene)

                    # Plays some samples
                    playout_reward, results, rollout_path, terminal = self._forward_explore(
                        scene,
                        idx[j])

                    # Train on collected samples
                    self._optimize_path(scene, playout_reward,
                                        results, rollout_path)
                    if (self.id == 0) and (self.optimizer.get_global_step() % 100) == 0:
                        print(
                            f'Global Step {self.optimizer.get_global_step()}')

                    # Trigger save or other
                    self.saver.after_optimization(self.id)

                # New episode with different scene/task
                j = j + 1
                j = j % len(self.tasks)
                # pass
            self.stop()
            [env.stop() for env in self.envs]
        except Exception as e:
            # self.logger.error(e.msg)
            raise e

    def stop(self):
        print("Stop initiated")
        self.exit.set()
