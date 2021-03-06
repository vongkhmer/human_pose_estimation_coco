# human_pose_estimation_coco
This an implementation of a simple human pose's estimation model that is trained on MS COCO dataset.
The architecture of this model is based on a paper by Bin Xiao et al.
Paper: https://arxiv.org/abs/1804.06208

I implement this model using Pytorch and train in on MS COCO dataset.
This implementation is a replicating experiment and for my learning purpose. 
If you use this code, please also credit the original authors.

Required python packages: numpy, PIL, torch, scikit-learn, tqdm, pickle.

To train a new model:
- cd libs
- python3 train.py

To predict keypoints of an image using a trained model:

-cd libs <br>
-python3 predict.py [MODEL_STATE_DICT_NAME] [IMAGE_NAME] [OUTPUT_IMAGE_NAME]

for example: "python3 predict.py pose_model_with_val_best_val_loss test6.jpeg result6.png" <br>
The model's state dict must be in the <b>trained_models/</b> directory. <br>
The test image must be in the <b>test/ </b> directory. <br>
The result image will be save in the <b>test/ </b> directory <br>

Some results that was tested on my machine: <br>
<img src="https://user-images.githubusercontent.com/10069391/119505466-f3a81880-bda7-11eb-8291-a29b17605ccb.png" alt="test1" height="300">
<img src="https://user-images.githubusercontent.com/10069391/119505480-f86ccc80-bda7-11eb-901a-03d60582c10b.png" alt="test4" height="300">
<img src="https://user-images.githubusercontent.com/10069391/119505516-01f63480-bda8-11eb-93d8-86aa7fcb9b32.png" alt="test5" height="300">


Enjoy and happy learning!
