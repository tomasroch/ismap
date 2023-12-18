# Transparent Boobs detector
## Basic Usage:
- Add images to censor to {Project}/images/test folder and run main_app.py
- (Test images used in article are in the images/test folder in link below)


## Link to dataset and test images
https://drive.google.com/drive/folders/1nqjSDXZfGZ_LOSYVF3lybsRmY-VOciTR?usp=sharing


### train.py - to train model on your own
- Edit config.yaml path to the project and paths to the labeled train and test images. \
Then run train.py (delete 'mps' flag if you are not using apple silicon chip)
- Don't forget to add {Project}/images/train and {Project}/images/val folders with your dataset images
- Don't forget to add {Project}/labels/train and {Project}/labels/val folders with your dataset labels
