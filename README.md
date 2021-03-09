# ANA: Attention-driven Navigation Agent
This code implements the Attention-driven Navigation Agent (ANA) for target-driven visual navigation. It is used for experimenting with the AI2-Thor framework under different settings.

### Reference
If you use our code, please cite our paper:
```
TBD
```

### Disclaimer
We build our code on top of the official implementation of the [Spatial Context model](https://github.com/norips/visual-navigation-agent-pytorch). Please refer to the link for further README information.

### Setup
1. We used Pytorch 1.4.0 and Python 3.7 in our experiments. The requirements can be installed via pip:
```
pip install -r requirements.txt
python -m spacy download en_core_web_lg
```
2. Download the pre-trained weights for [resnet](https://github.com/norips/visual-navigation-agent-pytorch/blob/master/agent/resnet/resnet50_places365.pth.tar) and [yolo](https://github.com/norips/visual-navigation-agent-pytorch/blob/master/yolo_dataset/backup/yolov3_ai2thor_best.weights), and put them in the following directories, respectively:
```
./agent/resnet50
./yolo_dataset/backup
```
3. Create the pre-computed features for AI2-Thor:
```
python create_dataset.py
```
If you are working on a headless server, you may find [this guide](https://medium.com/@etendue2013/how-to-run-ai2-thor-simulation-fast-with-google-cloud-platform-gcp-c9fcde213a4a) useful.

### Training
To train our model, you need to first generate a Json file using `create_experiment.py`, which specifies the experiment settings for different evaluation criteria. For evaluation in unseen environments with known semantics, simply call:
```
python create_experiment.py
```
The code will use environments from all room types for training and evaluation. For evaluation in unseen environments with unknown semantics, append an additional argument `eval_unknown` and specify the room types for training with `train_room` argument (`kitchen_bedroom` or `livingroom_bathroom`):
```
python create_experiment.py --eval_unknown --train_room kitchen_bedroom
```
By default, the evaluation is carried out on the validation splits. To generate results on the test splits, append the argument `testing`:
```
python create_experiment.py --testing
```

After obtaining the Json file `param.json`, place it in the directory (e.g., `EXP_DIR`) where you want to save the checkpoints and log files, and start training:
```
python train.py --exp EXP_DIR/param.json
```

The training log can be accessed with Tensorboard:
```
tensorboard --logdir=EXP_DIR/log/TIME/
```

### Evaluation
Since the evaluation settings have already been specified in the Json file, evaluating our method is straightforward:
```
python eval.py --checkpoint_path EXP_DIR/checkpoints --exp EXP_DIR/param.json --eval_type val_known
```
The code will generate the validation results for all checkpoints saved in the directory. After selecting the best checkpoint, copy the corresponding weights to another directory and compute the test results:
```
python eval.py --checkpoint_path EXP_DIR/checkpoint_test --exp EXP_DIR/param.json --eval_type test_known
```

We also provide code for visualizing the attention computed by our method. To generate the corresponding videos, comment/uncomment the following lines:
```
line 9/10, eval.py
line 795/796, agent/network.py
line 89/90 and line 115/116, agent/method/similarity_grid.py
```
and run the evaluation:
```
python eval.py --checkpoint_path EXP_DIR/checkpoints --exp EXP_DIR/param.json --eval_type val_known --show
```

### Pre-trained Model
TBD
