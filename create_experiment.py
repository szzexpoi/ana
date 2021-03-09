import argparse
import json
import re

import h5py
import numpy as np
import spacy
from scipy import spatial

KITCHEN_ID = 0
LIVINGROOM_ID = 200
BEDROOM_ID = 300
BATHROOM_ID = 400

names = []

# Splits for evaluation in environments with known semantics
TRAIN_SPLIT = (1, 21)
VAL_SPLIT = [21,23,25,27,29]
TEST_SPLIT = [22,24,26,28,30]


# Splits for evaluation in environments with unknown semantics
TRAIN_SPLIT_UNK = (1, 11)
VAL_SPLIT_UNK = [12,13,14,15,16]
TEST_SPLIT_UNK = [22,23,24,25,26]


KITCHEN_OBJECT_CLASS_LIST_TRAIN = [
    "Toaster",
    "Microwave",
    "Fridge",
    "CoffeeMachine",
    "GarbageCan",
    "Bowl",
]

KITCHEN_OBJECT_CLASS_LIST_EVAL = [
    "Mug",
    "Pot",
    "Cup"
]

LIVING_ROOM_OBJECT_CLASS_LIST_TRAIN = [
    "Pillow",
    "Laptop",
    "Television",
    "GarbageCan",
    "Bowl",
]

LIVING_ROOM_OBJECT_CLASS_LIST_EVAL = [
    "Sofa",
    "Box",
    "TableTop"
]

BEDROOM_OBJECT_CLASS_LIST_TRAIN = ["HousePlant", "Lamp", "Book", "AlarmClock"]

BEDROOM_OBJECT_CLASS_LIST_EVAL = ["Mirror", "CD", "CellPhone"]


BATHROOM_OBJECT_CLASS_LIST_TRAIN = [
    "Sink", "ToiletPaper", "SoapBottle", "LightSwitch"]

BATHROOM_OBJECT_CLASS_LIST_EVAL = [
    "Toilet", "Towel"]

scene_id_name = ["Kitchen", "LivingRoom", "Bedroom", "Bathroom"]


def extract_word_emb_vector(nlp, word_name):
    # Usee scapy to extract word embedding vector
    word_vec = nlp(word_name.lower())

    # If words don't exist in dataset
    # cut them using uppercase letter (SoapBottle -> Soap Bottle)
    if word_vec.vector_norm == 0:
        word = re.sub(r"(?<=\w)([A-Z])", r" \1", word_name)
        word_vec = nlp(word.lower())

        # If no embedding found try to cut word to find embedding (SoapBottle -> [Soap, Bottle])
        if word_vec.vector_norm == 0:
            word_split = re.findall('[A-Z][^A-Z]*', word)
            for word in word_split:
                word_vec = nlp(word.lower())
                if word_vec.has_vector:
                    break
            if word_vec.vector_norm == 0:
                print('ERROR: %s not found' % word_name)
                return None
    norm_word_vec = word_vec.vector / word_vec.vector_norm  # Normalize vector size
    return norm_word_vec, word_vec.text


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Create param.json experiment file')
    parser.add_argument('--train_range', nargs=2, default=TRAIN_SPLIT,
                        help='train scene range Ex : 1 11')

    parser.add_argument('--method', type=str, default="ana",
                        help='Method to use Ex : ana')
    parser.add_argument('--reward', type=str, default="soft_goal",
                        help='Method to use Ex : soft_goal')

    parser.add_argument('--eval_unknown', action="store_true")

    parser.add_argument('--train_room', type=str, default='kitchen_bedroom')

    parser.add_argument('--testing', action='store_true')

    args = vars(parser.parse_args())


    if not args['eval_unknown']:
        SCENES_TRAINING = [LIVINGROOM_ID, BATHROOM_ID,KITCHEN_ID,BEDROOM_ID]
        SCENES_EVAL = [LIVINGROOM_ID, BATHROOM_ID,KITCHEN_ID,BEDROOM_ID]
        args["train_range"] = TRAIN_SPLIT
        if args['testing']:
            eval_range = TEST_SPLIT
        else:
            eval_range = VAL_SPLIT
    else:
        if args['train_room'] == 'kitchen_bedroom':
            SCENES_TRAINING = [KITCHEN_ID, BEDROOM_ID]
            SCENES_EVAL = [LIVINGROOM_ID, BATHROOM_ID]
        elif args['train_room'] == 'livingroom_bathroom':
            SCENES_TRAINING = [LIVINGROOM_ID, BATHROOM_ID]
            SCENES_EVAL = [KITCHEN_ID, BEDROOM_ID]
        args["train_range"] = TRAIN_SPLIT_UNK

        if args['testing']:
            eval_range = TEST_SPLIT_UNK
        else:
            eval_range = VAL_SPLIT_UNK


    str_range = list(args["train_range"])
    for i, s in enumerate(str_range):
        str_range[i] = int(s)
    args["train_range"] = str_range

    data = {}

    scene_tasks = { KITCHEN_ID: KITCHEN_OBJECT_CLASS_LIST_TRAIN,
                    LIVINGROOM_ID: LIVING_ROOM_OBJECT_CLASS_LIST_TRAIN,
                    BEDROOM_ID: BEDROOM_OBJECT_CLASS_LIST_TRAIN,
                    BATHROOM_ID: BATHROOM_OBJECT_CLASS_LIST_TRAIN}

    training = {}
    set_obj = None
    for idx_scene, scene in enumerate(SCENES_TRAINING):
        for t in range(*args['train_range']):
            name = "FloorPlan" + str(scene + t)
            f = h5py.File("../visual-navigation-agent-pytorch/data/"+name+".h5", 'r')
            # Use h5py object available
            obj_available = json.loads(f.attrs["task_present"])

            obj_available = np.array(list(set.intersection(
                set(obj_available), set(scene_tasks[scene]))))
            obj_available = np.array(obj_available)
            obj_available_mask = [False for i in obj_available]
            obj_available_mask = np.array(obj_available_mask)

            object_visibility_tmp = [json.loads(j) for j in
                                     f['object_visibility']]

            object_visibility = set()
            for obj_visible in object_visibility_tmp:
                for objectId in obj_visible:
                    obj = objectId.split('|')
                    object_visibility.add(obj[0])
            object_visibility = list(object_visibility)

            for obj_visible in object_visibility:
                for obj_idx, curr_obj in enumerate(obj_available):
                    if obj_visible == curr_obj:
                        obj_available_mask[obj_idx] = True
                        break

            training[name] = [{"object": obj}
                              for obj in obj_available[obj_available_mask == True]]

    if args['eval_unknown']:
        scene_tasks = { KITCHEN_ID: KITCHEN_OBJECT_CLASS_LIST_EVAL,
                    LIVINGROOM_ID: LIVING_ROOM_OBJECT_CLASS_LIST_EVAL,
                    BEDROOM_ID: BEDROOM_OBJECT_CLASS_LIST_EVAL,
                    BATHROOM_ID: BATHROOM_OBJECT_CLASS_LIST_EVAL}


    evaluation = {}

    evaluation_set = dict()
    for idx_scene, scene in enumerate(SCENES_EVAL):
        evaluation_set[scene] = list()
        for t in eval_range:
            name = "FloorPlan" + str(scene + t)
            evaluation[name] = [
                {"object": obj} for obj in scene_tasks[scene]]
    data["task_list"] = {}
    data["task_list"]["train"] = training
    data["task_list"]["eval"] = evaluation
    data["total_step"] = 25000000
    data["h5_file_path"] = "../visual-navigation-agent-pytorch/data/{scene}.h5"
    data["saving_period"] = 1000000
    data["max_t"] = 5
    data["action_size"] = 9

    train_param = {}
    train_param["cuda"] = True
    train_param["num_thread"] = 6
    train_param["gamma"] = 0.7
    train_param["seed"] = 1993
    train_param["reward"] = args["reward"]
    train_param["mask_size"] = 16

    data["train_param"] = train_param
    data["eval_param"] = {}
    data["eval_param"]["num_episode"] = 250
    data["method"] = args["method"]

    with open('param.json', 'w') as outfile:
        outfile.write(json.dumps(data, indent=4))
