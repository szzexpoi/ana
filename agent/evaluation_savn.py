import csv
import imp
import os
import random
import sys
from itertools import groupby
import json
import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.multiprocessing as mp
import torch.nn.functional as F
from PIL import Image
from tensorboardX import SummaryWriter
from torch.nn import Sequential

from agent.environment.ai2thor_file import \
    THORDiscreteEnvironment as THORDiscreteEnvironmentFile
from agent.gpu_thread import GPUThread
from agent.method.aop import AOP
from agent.method.gcn import GCN
from agent.method.similarity_grid import SimilarityGrid
from agent.method.target_driven import TargetDriven
from agent.network import SceneSpecificNetwork, SharedNetwork, SceneSpecificNetwork_att2act
from agent.training import TrainingSaver
from agent.utils import find_restore_points, get_first_free_gpu
from torchvision import transforms

#See https://stackoverflow.com/a/42721412
from _ctypes import PyObj_FromPtr  # see https://stackoverflow.com/a/15012814/355230
import json
import re
class NoIndent(object):
    """ Value wrapper. """
    def __init__(self, value):
        if not isinstance(value, (list, tuple)):
            raise TypeError('Only lists and tuples can be wrapped')
        self.value = value


class MyEncoder(json.JSONEncoder):
    FORMAT_SPEC = '@@{}@@'  # Unique string pattern of NoIndent object ids.
    regex = re.compile(FORMAT_SPEC.format(r'(\d+)'))  # compile(r'@@(\d+)@@')

    def __init__(self, **kwargs):
        # Keyword arguments to ignore when encoding NoIndent wrapped values.
        ignore = {'cls', 'indent'}

        # Save copy of any keyword argument values needed for use here.
        self._kwargs = {k: v for k, v in kwargs.items() if k not in ignore}
        super(MyEncoder, self).__init__(**kwargs)

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return (self.FORMAT_SPEC.format(id(obj)) if isinstance(obj, NoIndent)
                    else super(MyEncoder, self).default(obj))

    def iterencode(self, obj, **kwargs):
        format_spec = self.FORMAT_SPEC  # Local var to expedite access.

        # Replace any marked-up NoIndent wrapped values in the JSON repr
        # with the json.dumps() of the corresponding wrapped Python object.
        for encoded in super(MyEncoder, self).iterencode(obj, **kwargs):
            match = self.regex.search(encoded)
            if match:
                id = int(match.group(1))
                no_indent = PyObj_FromPtr(id)
                json_repr = json.dumps(no_indent.value, **self._kwargs)
                # Replace the matched id string with json formatted representation
                # of the corresponding Python object.
                encoded = encoded.replace(
                            '"{}"'.format(format_spec.format(id)), json_repr)

            yield encoded

def prepare_csv(file, scene_task):
    f = open(file, 'w', newline='')
    writer = csv.writer(f)
    header = ['']
    header2lvl = ['Checkpoints']
    values = ['_reward', '_length', '_collision', '_success', '_spl']
    for scene_scope, tasks_scope in scene_task:
        for task in tasks_scope:
            for val in values:
                header.append(scene_scope)
                header2lvl.append(task['object'] + val)
    writer.writerow(header)
    writer.writerow(header2lvl)
    return writer


def write_text(img, text, pos):
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = pos
    fontScale = 1
    fontColor = (255, 255, 255)
    lineType = 2

    cv2.putText(img, text,
                bottomLeftCornerOfText,
                font,
                fontScale,
                fontColor,
                lineType)
    text_width, text_height = cv2.getTextSize(text, font, fontScale, lineType)[0]
    return img, text_width, text_height

def overlay_heatmap(img,att,cmap=plt.cm.jet):
    gamma = 1.0
    att[att<0.3] = 0.01
    att = cv2.blur(att,(35,35))
    colorized = cmap(np.uint8(att*255))
    alpha = 0.7
    # overlaid = np.uint8(img*(1-alpha)+colorized[:,:,2::-1]*255*alpha) # for BGR
    overlaid = np.uint8(img*(1-alpha)+colorized[:,:,:3]*255*alpha) # for RGB
    return overlaid

def create_img(target, obs, obs_feature, word_embedding, simi_grid, attention, action=None):
    padding = 3
    base_img = np.zeros((720, 1280, 3), dtype=np.uint8)

    """OBSERVATION
    """
    # Set obs width and height
    width_obs, height_obs = 400, 300

    #  Set obs position
    pos_width_obs, pos_height_obs = int(width_obs*1.5)+padding*4, 200

    obs = cv2.resize(obs, dsize=(int(width_obs*1.5), int(height_obs*1.5)))
    attention = cv2.resize(attention, dsize=(int(width_obs*1.5), int(height_obs*1.5)))
    simi_grid = cv2.resize(simi_grid, dsize=(int(width_obs*1.5), int(height_obs*1.5)))
    attention /= attention.max()
    att_obs = overlay_heatmap(obs,attention)

    # Set obs width and height
    height_obs, width_obs, _ = obs.shape

    base_img, text_width, text_height = write_text(
        base_img, "Observation", (200+pos_width_obs, pos_height_obs))

    # # Merge obs in canva
    base_img[pos_height_obs+text_height:pos_height_obs+height_obs+text_height+padding*2,
             pos_width_obs:pos_width_obs+width_obs+padding*2, :] = np.pad(obs, ((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=255)

    # Display similarity grid instead of observation
    # simi_grid = (simi_grid - np.min(simi_grid))/(np.max(simi_grid) - np.min(simi_grid))
    # simi_grid =  overlay_heatmap(obs,simi_grid)
    # base_img[pos_height_obs+text_height:pos_height_obs+height_obs+text_height+padding*2,
    #          pos_width_obs:pos_width_obs+width_obs+padding*2, :] = np.pad(simi_grid, ((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=255)


    # Merge attention in canvas
    base_img[pos_height_obs+text_height:pos_height_obs+height_obs+text_height+padding*2,
             padding*2:pos_width_obs, :] = np.pad(att_obs, ((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=255)


    """
    #OBSERVATION FEATURE

    # Set observation feature position
    pos_width_feat, pos_height_feat = 100, 100

    # Normalize
    obs_feature = (obs_feature - np.min(obs_feature)) / (np.max(obs_feature) - np.min(obs_feature))
    obs_feature = obs_feature * 255

    # Vector to matrix and upscale
    obs_feature = obs_feature.reshape(
        (32, -1, 1)).repeat(3, axis=2).repeat(2, axis=0).repeat(2, axis=1)

    base_img, text_width, text_height = write_text(
        base_img, "Visual feature", (pos_width_feat, pos_height_feat))

    # Get obs width and height
    height_feat, width_feat, _ = obs_feature.shape

    base_img[pos_height_feat+text_height:pos_height_feat+text_height+height_feat+padding*2,
             pos_width_feat:pos_width_feat+width_feat+padding*2, :] = np.pad(obs_feature, ((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=255)

    # WORD EMBEDDING FEATURE

    # Set observation feature position
    pos_width_we, pos_height_we = 100, 250

    # Normalize
    word_embedding = (word_embedding - np.min(word_embedding)) / \
        (np.max(word_embedding) - np.min(word_embedding))
    word_embedding = word_embedding * 255

    # Vector to matrix and upscale
    word_embedding = word_embedding.reshape(
        (10, -1, 1)).repeat(3, axis=2).repeat(8, axis=0).repeat(8, axis=1)

    base_img, text_width, text_height = write_text(
        base_img, "Word embedding feature", (pos_width_we, pos_height_we))

    # Get obs width and height
    height_we, width_we, _ = word_embedding.shape

    base_img[pos_height_we+text_height:pos_height_we+text_height+height_we+padding*2,
             pos_width_we:pos_width_we+width_we+padding*2, :] = np.pad(word_embedding, ((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=255)

    # GRID

    # Set observation feature position
    pos_width_grid, pos_height_grid = 100, 400

    # Normalize
    simi_grid = (simi_grid - np.min(simi_grid)) / \
        (np.max(simi_grid) - np.min(simi_grid))
    simi_grid = simi_grid * 255

    # Vector to matrix and upscale
    simi_grid = simi_grid.repeat(3, axis=2).repeat(12, axis=0).repeat(12, axis=1)

    base_img, text_width, text_height = write_text(
        base_img, "Similarity grid", (pos_width_grid, pos_height_grid))

    # Get obs width and height
    height_grid, width_grid, _ = simi_grid.shape

    base_img[pos_height_grid+text_height:pos_height_grid+text_height+height_grid+padding*2,
             pos_width_grid:pos_width_grid+width_grid+padding*2, :] = np.pad(simi_grid, ((padding, padding), (padding, padding), (0, 0)), 'constant', constant_values=255)
    """

    """TARGET NAME
    """
    base_img, text_width, text_height = write_text(
        base_img, "TARGET : " + target, (200+pos_width_obs, pos_height_obs-100))

    action_dict = {0:"MoveAhead", 1:"RotateRight", 2:"RotateLeft", 3:"MoveBack",
            4:"LookUp", 5:"LookDown", 6:"MoveRight", 7:"MoveLeft", 8:"Done"}

    if action is not None:
        base_img, text_width, text_height = write_text(
            base_img, "Action : " + action_dict[action], (200+pos_width_obs, pos_height_obs-50))

    return base_img

class Logger(object):
    def __init__(self, path="logfile.log"):
        self.terminal = sys.stdout
        self.log = open(path, "w")

    def write(self, message, term='\n'):
        self.terminal.write(message + term)
        self.log.write(message + term)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    def __del__(self):
        self.log.close()

class Evaluation_savn:
    def __init__(self, config):
        self.config = config
        self.method = config['method']
        gpu_id = get_first_free_gpu(2000)
        if gpu_id is None:
            print("You need at least 2G of GPU RAM")
            exit()
        self.device = torch.device("cuda:" + str(gpu_id))
        if self.method != "random":
            self.shared_net = SharedNetwork(
                self.config['method'], self.config.get('mask_size', 5)).to(self.device)

            self.scene_net = SceneSpecificNetwork_att2act(
                self.config['action_size']).to(self.device)



        self.checkpoints = []
        self.checkpoint_id = 0
        self.saver = None
        self.chk_numbers = None

    @staticmethod
    def load_checkpoints(config, fail=True):
        evaluation = Evaluation_savn(config)
        checkpoint_path = config.get(
            'checkpoint_path', 'model/checkpoint-{checkpoint}.pth')

        checkpoints = []

        # # original script (not working)
        # (base_name, chk_numbers) = find_restore_points(checkpoint_path, fail)

        # simplified new script
        all_checkpoint = os.listdir(checkpoint_path)
        chk_numbers = [int(os.path.basename(cur)[:-4]) for cur in all_checkpoint]
        chk_numbers = sorted(chk_numbers)
        base_name = [str(cur)+'.pth' for cur in chk_numbers]

        if evaluation.method != "random":
            try:
                for chk_name in base_name:
                    # print(checkpoint_path)
                    # print(chk_name)
                    # assert 0
                    state = torch.load(
                        open(os.path.join(os.path.dirname(checkpoint_path), chk_name), 'rb'))
                    checkpoints.append(state)
            except Exception as e:
                print("Error loading", e)
                exit()
        evaluation.saver = TrainingSaver(evaluation.shared_net,
                                         evaluation.scene_net, None, evaluation.config)
        evaluation.chk_numbers = chk_numbers
        evaluation.checkpoints = checkpoints
        return evaluation

    def restore(self):
        print('Restoring from checkpoint',
              self.chk_numbers[self.checkpoint_id])
        self.saver.restore(self.checkpoints[self.checkpoint_id])

    def next_checkpoint(self):
        self.checkpoint_id = (self.checkpoint_id + 1) % len(self.checkpoints)

    def save_video(self, ep_lengths, ep_actions, ep_start, ind_succ_or_fail_ep, chk_id, env, scene_scope, task_scope, success=True):
        # Find episode based on episode length
        if not ind_succ_or_fail_ep:
            return
        ep_lengths = np.array(ep_lengths)
        sorted_ep_lengths = np.sort(ep_lengths[ind_succ_or_fail_ep])
        ep_lengths_succeed = ep_lengths[ind_succ_or_fail_ep]
        ep_actions_succeed = np.array(ep_actions)[ind_succ_or_fail_ep]
        ep_start_succeed = np.array(ep_start)[ind_succ_or_fail_ep]

        ind_list = []
        names_video = []
        if success:
            # Best is the first episode in the sorted list but we want more than 5 step
            index_best = 0
            for idx, ep_len in enumerate(sorted_ep_lengths):
                if ep_len >= 5:
                    index_best = idx
                    break
            index_best = np.where(
                ep_lengths_succeed == sorted_ep_lengths[index_best])
            index_best = index_best[0][0]
            # print("Best", ep_lengths_succeed[index_best])

            # Worst is the last episode in the sorted list
            index_worst = np.where(
                ep_lengths_succeed == sorted_ep_lengths[-1])
            index_worst = index_worst[0][0]
            # print("Worst", ep_lengths_succeed[index_worst])

            # Median is half the array size
            index_median = np.where(
                ep_lengths_succeed == sorted_ep_lengths[len(sorted_ep_lengths)//2])
            # Extract index
            index_median = index_median[0][0]
            # print("Median", ep_lengths_succeed[index_median])

            names_video = ['best', 'median', 'worst']
            ind_list = [index_best, index_median, index_worst]
        else:
            ind_list = [i for i in range(len(ind_succ_or_fail_ep))]
            names_video = ['Fail_' + str(i) for i in range(len(ind_succ_or_fail_ep))]

        # Create dir if not exisiting
        directory = os.path.join(
            self.config['base_path'], 'video', str(chk_id))
        if not os.path.exists(directory):
            os.makedirs(directory)
        for idx_name, idx in enumerate(ind_list):
            # Create video to save
            height, width, layers = 720, 1280, 3
            filename = os.path.join(directory, scene_scope + '_' +
                                        task_scope['object'] + '_' +
                                        names_video[idx_name] + '_' +
                                        str(ep_lengths_succeed[idx]))
            video_name = os.path.join(filename + '.avi')
            text_name = os.path.join(filename + '.json')
            FPS = 5
            video = cv2.VideoWriter(
                video_name, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (width, height))
            # Retrieve start position
            state_id_best = ep_start_succeed[idx]
            env.reset()

            # Set start position
            env.current_state_id = state_id_best
            for a in ep_actions_succeed[idx]:
                state, x_processed, object_mask = self.method_class.extract_input(
                    env, torch.device("cpu"))
                x_processed = x_processed.view(-1, 1).numpy()
                object_mask = object_mask.squeeze().unsqueeze(2).numpy()
                object_mask = np.flip(np.rot90(object_mask), axis=0)
                img = create_img(task_scope['object'], env.observation, x_processed,
                                    np.zeros((300, 1)), object_mask)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video.write(img)
                env.step(a)
            for i in range(10):
                video.write(img)
            video.release()

            data = {}
            data['start'] = ep_start_succeed[idx]
            data['stop'] = env.current_state_id
            data['action'] = [env.acts[i] for i in ep_actions_succeed[idx]]
            data['object_visible'] = [k.split("|")[0] for k in env.boudingbox.keys()]
            round_mask = np.squeeze(np.around(object_mask,2)).tolist()

            def fmt(v):
                return "%.2f" % (v,)
            vecfmt = np.vectorize(fmt)

            data['object_mask'] = [NoIndent(e) for e in vecfmt(round_mask).tolist()]

            with open(text_name, 'w') as outfile:
                json.dump(data, outfile, cls=MyEncoder, sort_keys=True, indent=4)

    def save_video_att(self, ep_lengths, ep_actions, ep_att, ep_start, ind_succ_or_fail_ep, chk_id, env, scene_scope, task_scope, success=True):
        # Find episode based on episode length
        if not ind_succ_or_fail_ep:
            return
        ep_lengths = np.array(ep_lengths)
        sorted_ep_lengths = np.sort(ep_lengths[ind_succ_or_fail_ep])
        ep_lengths_succeed = ep_lengths[ind_succ_or_fail_ep]
        ep_actions_succeed = np.array(ep_actions)[ind_succ_or_fail_ep]
        ep_att_succeed = np.array(ep_att)[ind_succ_or_fail_ep]
        ep_start_succeed = np.array(ep_start)[ind_succ_or_fail_ep]

        ind_list = []
        names_video = []
        if success:
            # Best is the first episode in the sorted list but we want more than 5 step
            index_best = 0
            for idx, ep_len in enumerate(sorted_ep_lengths):
                if ep_len >= 5:
                    index_best = idx
                    break
            index_best = np.where(
                ep_lengths_succeed == sorted_ep_lengths[index_best])
            index_best = index_best[0][0]
            # print("Best", ep_lengths_succeed[index_best])

            # Worst is the last episode in the sorted list
            index_worst = np.where(
                ep_lengths_succeed == sorted_ep_lengths[-1])
            index_worst = index_worst[0][0]
            # print("Worst", ep_lengths_succeed[index_worst])

            # Median is half the array size
            index_median = np.where(
                ep_lengths_succeed == sorted_ep_lengths[len(sorted_ep_lengths)//2])
            # Extract index
            index_median = index_median[0][0]
            # print("Median", ep_lengths_succeed[index_median])

            names_video = ['best', 'median', 'worst']
            ind_list = [index_best, index_median, index_worst]
        else:
            ind_list = [i for i in range(len(ind_succ_or_fail_ep))]
            names_video = ['Fail_' + str(i) for i in range(len(ind_succ_or_fail_ep))]

        # Create dir if not exisiting
        directory = os.path.join(
            self.config['base_path'], 'video', str(chk_id))
        if not os.path.exists(directory):
            os.makedirs(directory)
        for idx_name, idx in enumerate(ind_list):
            # Create video to save
            height, width, layers = 720, 1280, 3
            filename = os.path.join(directory, scene_scope + '_' +
                                        task_scope['object'] + '_' +
                                        names_video[idx_name] + '_' +
                                        str(ep_lengths_succeed[idx]))
            video_name = os.path.join(filename + '.avi')
            text_name = os.path.join(filename + '.json')
            FPS = 1
            video = cv2.VideoWriter(
                video_name, cv2.VideoWriter_fourcc(*"MJPG"), FPS, (width, height))
            # Retrieve start position
            state_id_best = ep_start_succeed[idx]
            env.reset()

            # Set start position
            env.current_state_id = state_id_best
            for s_idx, a in enumerate(ep_actions_succeed[idx]):
                state, x_processed, object_mask = self.method_class.extract_input(
                    env, torch.device("cpu"))
                x_processed = x_processed.view(-1, 1).numpy()
                object_mask = object_mask.squeeze().unsqueeze(2).numpy()
                object_mask = np.flip(np.rot90(object_mask), axis=0)
                cur_att = ep_att_succeed[idx][s_idx]
                img = create_img(task_scope['object'], env.observation, x_processed,
                                    np.zeros((300, 1)), object_mask, cur_att, a)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                video.write(img)
                env.step(a)
            for i in range(10):
                video.write(img)
            video.release()

            data = {}
            data['start'] = ep_start_succeed[idx]
            data['stop'] = env.current_state_id
            data['action'] = [env.acts[i] for i in ep_actions_succeed[idx]]
            data['object_visible'] = [k.split("|")[0] for k in env.boudingbox.keys()]
            round_mask = np.squeeze(np.around(object_mask,2)).tolist()

            def fmt(v):
                return "%.2f" % (v,)
            vecfmt = np.vectorize(fmt)

            data['object_mask'] = [NoIndent(e) for e in vecfmt(round_mask).tolist()]

            with open(text_name, 'w') as outfile:
                json.dump(data, outfile, cls=MyEncoder, sort_keys=True, indent=4)

    def run(self, show=False):
        self.method_class = None
        # Load policy network
        if self.method == 'word2vec' or self.method == 'word2vec_nosimi' or \
           self.method == 'word2vec_noconv' or self.method == 'word2vec_notarget' or \
           self.method == 'gcn' or self.method == 'aop_we' or self.method == 'word2vec_notarget_lstm' or \
           self.method == 'word2vec_notarget_lstm_2layer' or self.method == 'word2vec_notarget_lstm_3layer' or \
           self.method == 'word2vec_notarget_rnn' or self.method == 'word2vec_notarget_gru':
            self.method_class = SimilarityGrid(self.method)
        elif self.method == 'aop' or self.method == 'aop_we':
            self.method_class = AOP(self.method)
        elif self.method == 'target_driven':
            self.method_class = TargetDriven(self.method)
        elif self.method == 'gcn':
            self.method_class = GCN(self.method)

        # Init random seed
        random.seed(200)
        print(self.chk_numbers)


        for chk_id in self.chk_numbers:
            resultData = [chk_id]
            if self.config['train']:
                log = Logger(self.config['base_path'] + 'train' +
                             str(chk_id) + '.log') # add "_object" for unknown object evaluation
            else:
                log = Logger(self.config['base_path'] + 'eval' +
                             str(chk_id) + '.log')
            scene_stats = dict()
            if self.method != "random":
                self.restore()
                self.next_checkpoint()


            if self.method != "random":
                scene_net = self.scene_net
                scene_net.eval()

            network = Sequential(self.shared_net, scene_net)
            network.eval()

            room_type = ['kitchen','livingroom','bedroom','bathroom']
            room_id = {'kitchen':['FloorPlan22','FloorPlan24','FloorPlan26','FloorPlan28','FloorPlan30'],
                        'livingroom': ['FloorPlan222','FloorPlan224','FloorPlan226','FloorPlan228','FloorPlan230'],
                        'bedroom': ['FloorPlan322','FloorPlan324','FloorPlan326','FloorPlan328','FloorPlan330'],
                        'bathroom': ['FloorPlan422','FloorPlan424','FloorPlan426','FloorPlan428','FloorPlan430']}


            # SPL results
            spl_result = dict()
            for cur_type in room_type + ['avg']:
                spl_result[cur_type] = dict()
                spl_result[cur_type]['count'] = 0
                spl_result[cur_type]['spl'] = 0
                spl_result[cur_type]['sr'] = 0

            # SPL-5 results
            spl_5_result = dict()
            for cur_type in room_type  + ['avg']:
                spl_5_result[cur_type] = dict()
                spl_5_result[cur_type]['count'] = 0
                spl_5_result[cur_type]['spl'] = 0
                spl_5_result[cur_type]['sr'] = 0


            # attention correlation
            att_corr_result = dict()
            att_corr_result['room_type'] = dict()
            att_corr_result['floor_plan'] = dict()
            for cur_type in room_type  + ['avg']:
                att_corr_result['room_type'][cur_type] = dict()
                att_corr_result['room_type'][cur_type]['count'] = 0
                att_corr_result['room_type'][cur_type]['corr'] = 0

            for cur_type in room_type:
                for cur_room in room_id[cur_type]:
                    att_corr_result['floor_plan'][cur_room] = dict()
                    att_corr_result['floor_plan'][cur_room]['count'] = 0
                    att_corr_result['floor_plan'][cur_room]['corr'] = 0


            for cur_type in room_type:
                cur_eval_candidate = []
                num_episode = 250 # 250 episodes per room type

                for scene_scope, items in self.config['task_list'].items():
                    if scene_scope in room_id[cur_type]:
                        cur_eval_candidate.append([scene_scope,items])

                i_episode = 1
                while i_episode <= num_episode:
                    sample_room = np.random.randint(0,len(cur_eval_candidate),1)[0]
                    sample_obj = np.random.randint(0,len(cur_eval_candidate[sample_room][1]),1)[0]
                    scene_scope = cur_eval_candidate[sample_room][0]
                    task_scope = cur_eval_candidate[sample_room][1][sample_obj]

                    env = THORDiscreteEnvironmentFile(scene_name=scene_scope,
                                                      method=self.method,
                                                      reward=self.config['reward'],
                                                      h5_file_path=(lambda scene: self.config.get(
                                                          "h5_file_path").replace('{scene}', scene)),
                                                      terminal_state=task_scope,
                                                      action_size=self.config['action_size'],
                                                      mask_size=self.config.get(
                                                          'mask_size', 5))

                    ep_rewards = []
                    ep_lengths = []
                    ep_collisions = []
                    ep_actions = []
                    ep_start = []
                    ep_success = []
                    ep_spl = []
                    ep_shortest_distance = []
                    embedding_vectors = []
                    state_ids = list()

                    ep_fail_threshold = 300

                    if not env.reset():
                        continue

                    terminal = False
                    ep_reward = 0
                    ep_collision = 0
                    ep_t = 0
                    actions = []
                    ep_start.append(env.current_state_id)


                    while not terminal:
                        if self.method != "random":
                            policy, value, state = self.method_class.forward_policy(
                                env, self.device, network)

                            with torch.no_grad():
                                action = F.softmax(policy, dim=0).multinomial(
                                    1).data.cpu().numpy()[0]

                            if env.current_state_id not in state_ids:
                                state_ids.append(env.current_state_id)
                        else:
                            action = np.random.randint(env.action_size)

                        env.step(action)
                        actions.append(action)


                        ep_reward += env.reward
                        terminal = env.terminal

                        if ep_t == ep_fail_threshold:
                            break
                        if env.collided:
                            ep_collision += 1
                        ep_t += 1

                    ep_actions.append(actions)
                    ep_lengths.append(ep_t)
                    ep_rewards.append(ep_reward)
                    ep_shortest_distance.append(env.shortest_path_terminal(
                        ep_start[-1]))
                    ep_collisions.append(ep_collision)

                    # Compute SPL
                    spl = env.shortest_path_terminal(
                        ep_start[-1])/ep_t

                    if env.success:
                        spl_result[cur_type]['sr'] += 1
                        spl_result[cur_type]['spl'] += spl
                        spl_result['avg']['sr'] += 1
                        spl_result['avg']['spl'] += spl
                    spl_result[cur_type]['count'] += 1
                    spl_result['avg']['count'] += 1

                    # compute SPL-5
                    if env.shortest_path_terminal(ep_start[-1])>5:
                        if env.success:
                            spl_5_result[cur_type]['sr'] += 1
                            spl_5_result[cur_type]['spl'] += spl
                            spl_5_result['avg']['sr'] += 1
                            spl_5_result['avg']['spl'] += spl
                        spl_5_result[cur_type]['count'] += 1
                        spl_5_result['avg']['count'] += 1

                    i_episode += 1

            log.write('\nResults (average trajectory length):')
            for cur_type in room_type+['avg']:
                log.write('%s: %.4f spl |  %.2f%% success | %.4f spl-5 | %.2f%% success-5' %
                    (cur_type,spl_result[cur_type]['spl']/spl_result[cur_type]['count'],
                      spl_result[cur_type]['sr']/spl_result[cur_type]['count']*100,
                      spl_5_result[cur_type]['spl']/spl_5_result[cur_type]['count'],
                      spl_5_result[cur_type]['sr']/spl_5_result[cur_type]['count']*100))



'''
# Load weights trained on tensorflow
data = pickle.load(
    open(os.path.join(__file__, '..\\..\\weights.p'), 'rb'), encoding='latin1')
def convertToStateDict(data):
    return {key:torch.Tensor(v) for (key, v) in data.items()}

shared_net.load_state_dict(convertToStateDict(data['navigation']))
for key in TASK_LIST.keys():
    scene_nets[key].load_state_dict(convertToStateDict(data[f'navigation/{key}']))'''
