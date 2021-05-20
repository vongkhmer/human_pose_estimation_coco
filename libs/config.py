class Config:
  image_shape = (224,224)
  bbx_multiplier = 1.2
  num_keypoints = None
  heatmap_shape = None
  sigma = 3
  heatmap_radius = 9 
  ignore_keypoints = ["left_eye", "right_eye", "left_ear", "right_ear"]


keypoints_index = [ "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle" ]
keypoints_dict = dict()
for i, kp in enumerate(keypoints_index):
  keypoints_dict[kp] = i

def remove_ignored_keypoints(keyp):
  ret = []
  for i in range(len(keyp) // 3):
    if keypoints_index[i] in Config.ignore_keypoints:
      continue
    ret.extend(keyp[3*i:3*i+3])
  return ret
