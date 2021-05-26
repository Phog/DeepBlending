# Deep Blending for Free-Viewpoint Image-Based Rendering
[Peter Hedman](http://www.phogzone.com), 
[Julien Philip](https://www-sop.inria.fr/members/Julien.Philip), 
[True Price](https://www.cs.unc.edu/~jtprice), 
[Jan-Michael Frahm](http://frahm.web.unc.edu),
[George Drettakis](https://www-sop.inria.fr/members/George.Drettakis), and
[Gabriel Brostow](http://www0.cs.ucl.ac.uk/staff/G.Brostow). *SIGGRAPH Asia 2018*.

http://visual.cs.ucl.ac.uk/pubs/deepblending

<a href="http://www.youtube.com/watch?feature=player_embedded&v=F1g7PAZ9cI4
" target="_blank"><img src="http://img.youtube.com/vi/F1g7PAZ9cI4/0.jpg" 
alt="Teaser video" /></a>


This repository contains the training and test code for our blending network. 

## New! Source code for rendering and geometry refinement.

You can find a full source code release [here](https://gitlab.inria.fr/sibr/projects/inside_out_deep_blending).

## Usage

Only tested on [Ubuntu 16.04](http://releases.ubuntu.com/16.04/) and [NVIDIA](http://www.nvidia.com) GPUs. 

### Prerequisites

Download and install [Python 3](https://www.python.org/download/releases/3.0/), [Pip](https://pip.pypa.io/en/stable/installing/), and [Tensorflow-GPU](https://www.tensorflow.org/install/gpu).

Required Python packages: *Numpy*, *Scipy*, and *PIL*.

Download the [training data](https://repo-sam.inria.fr/fungraph/deep-blending/data/DeepBlendingTrainingData.zip), and the [test data](https://repo-sam.inria.fr/fungraph/deep-blending/data/DeepBlendingTestDataAndResults.zip). Unzip these datasets.

### Training

For training, we need to set a few more things up.

1) Download pretrained VGG weights
```
cd [DEEP_BLENDING_CODE_FOLDER]
wget http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz
tar -xvf vgg_16_2016_08_28.tar.gz
rm vgg_16_2016_08_28.tar.gz
```

2) Create an input txt file listing all the training images.
```
cd [DEEP_BLENDING_CODE_FOLDER]
DATASET_DIR=[PATH_TO_UNZIPPED_TRAINING_DATA]
for scene in `ls $DATASET_DIR | grep -v txt` ; 
do
python3 make_data_txt.py $DATASET_DIR/$scene > ${scene}_training.txt ; 
cat ${scene}_training.txt >> training_all_ordered.txt ; 
rm ${scene}_training.txt ;
done ;
shuf training_all_ordered.txt > $DATASET_DIR/training.txt ;
rm training_all_ordered.txt ;
```

3) Create an input txt file listing all the validation images.
```
cd [DEEP_BLENDING_CODE_FOLDER]
DATASET_DIR=[PATH_TO_UNZIPPED_TRAINING_DATA]
for scene in `ls $DATASET_DIR | grep -v txt` ; 
do
python3 make_data_txt.py $DATASET_DIR/$scene validation > ${scene}_validation.txt ;
cat ${scene}_validation.txt >> validation_all_ordered.txt ;
rm ${scene}_validation.txt ;
done ;
shuf validation_all_ordered.txt > $DATASET_DIR/validation.txt ;
rm validation_all_ordered.txt ;
```

Now we're ready to train the network. This command trains the network with the parameters used in the paper:
```
cd [DEEP_BLENDING_CODE_FOLDER]
python3 train.py [PATH_TO_UNZIPPED_TRAINING_DATA] NETWORK_OUTPUT_FOLDER --training_file=[PATH_TO_UNZIPPED_TRAINING_DATA]/training.txt --validation_file=[PATH_TO_UNZIPPED_TRAINING_DATA]/validation.txt
```

However, you can also use the following command line parameters to train a different version of the network:

`--loss_function`,
 determines the mage loss to be used for training (*L1*, *VGG*, or *VGG_AND_L1*). Defaults to *VGG_AND_L1*.

`--direct_regression`,
directly regresses the output image instead of predicting blend weights. Off by default.

`--no_temporal_loss`,
trains the network without a temporal loss. Off by default.

`--no_textured_mesh`,
disable the input layer from the textured mesh. Off by default.

`--num_input_mosaics`,
number of input mosaic layers to use. Defaults to 4.

`--temporal_alpha`,
relative strength of the temporal loss. Defaults to 0.33.

`--debug`,
debug mode for training, shows more intermediate outputs in tensorboard. Off by default.

`--num_batches`,
training duration (in terms of number of minibatches). Defaults to 256000.

`--batch_size`,
batch size to be used for training. Defaults to 8.

`--crop`,
crop size for data augmentation. Defaults to 256.


### Testing

Run the following command:
```
cd [DEEP_BLENDING_CODE_FOLDER]
python3 test.py [PATH_TO_TEST_SCENE] [OUTPUT_DIRECTORY] --model_path=[NETWORK_OUTPUT_FOLDER]
```

**IMPORTANT** If you trained a network using custom command-line parameters, make sure that they match when you run the network in test mode!

For testing on new scenes, you need to create a txt file which lists all the inputs:
```
cd [DEEP_BLENDING_CODE_FOLDER]
TEST_SCENE_DIR=[PATH_TO_TEST_SCENE]
python3 make_data_txt.py $TEST_SCENE_DIR/testdump test > $TEST_SCENE_DIR/test.txt
```
