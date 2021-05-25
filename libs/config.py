class Config:
  # crop_shape = (256,256)
  image_shape = (224,224)
  bbx_multiplier = 1.2
  num_keypoints = 13 
  heatmap_shape = (56, 56)
  heatmap_radius = 6
  sigma = 2 
  ignore_keypoints = ["left_eye", "right_eye", "left_ear", "right_ear"]
  used_keypoints = []
  sym_pairs = []
  connected_line = []

keypoints_index = [ "nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle" ]
keypoints_dict = dict()

for i, kp in enumerate(keypoints_index):
  keypoints_dict[kp] = i
for k in keypoints_index:
  if k not in Config.ignore_keypoints:
    Config.used_keypoints.append(k)

for i, point in enumerate(Config.used_keypoints):
  if "left" in point:
    j = Config.used_keypoints.index("right" + point[4:])
    Config.sym_pairs.append((i, j))

lines = [("left_shoulder", "right_shoulder"), ("left_shoulder", "left_elbow"), ("right_shoulder", "right_elbow"), ("left_elbow", "left_wrist"), ("right_elbow", "right_wrist"), ("left_shoulder", "left_hip"), ("right_shoulder", "right_hip"), ("left_hip", "right_hip"), ("left_hip", "left_knee"), ("left_knee", "left_ankle"), ("right_hip", "right_knee"), ("right_knee", "right_ankle")]

for (x, y) in lines:
  try :
    i = Config.used_keypoints.index(x)
  except:
    i =  -1
  try:
    j = Config.used_keypoints.index(y)
  except:
    j = -1
  if i >= 0 and j >= 0:
    Config.connected_line.append((i, j))

def remove_ignored_keypoints(keyp):
  ret = []
  for i in range(len(keyp) // 3):
    if keypoints_index[i] in Config.ignore_keypoints:
      continue
    ret.extend(keyp[3*i:3*i+3])
  return ret