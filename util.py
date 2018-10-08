import copy
import json
import numpy as np
import os
import re
from render_scenes_json import RenderScenes
from scipy.ndimage import imread
from scipy.misc import imsave
import sys
import time


HEIGHT = 400
WIDTH = 700

def get_global_image(img):
    '''Returns a large background image with the img superimposed in the center'''
    assert img.shape[2] == 3
    scene_type = get_scene_type(img)
    global_img = add_image_to_background(img, scene_type)
    return global_img

def center_image_at_pose(global_img, pose):
    (x, y, global_rotation, flip) = pose
    i = int(y * (0 < y < HEIGHT))
    j = int(x * (0 < x < WIDTH))
    if flip:
        return global_img[i:i+2*HEIGHT, j+2*WIDTH:j:-1]
    else:
        return global_img[i:i+2*HEIGHT, j:j+2*WIDTH]

def get_scene_type(img):
    '''
    Empirically, all frames match the background in at least >20% of the pixels.
    The most overlap between a frame and a scene of the opposite category is <3%
    '''
    park_background = imread('afs/csail.mit.edu/u/b/bce/TheoryOfMind/abstract_scenes_v002/site_pngs/Park/BG1.png', mode='RGB')
    living_background = imread('afs/csail.mit.edu/u/b/bce/TheoryOfMind/abstract_scenes_v002/site_pngs/Living/BG1.png', mode='RGB')

    park_match = np.mean(img == park_background)
    living_match = np.mean(img == living_background)

    if park_match > living_match:
        scene_type = 'Park'
    else:
        scene_type = 'Living'

    if max(park_match, living_match) / min(park_match, living_match) < 5:
        print 'Warning: uncertain scene type'
        print '\t park=%.3f\tliving=%.3f' % (park_match, living_match)

    return scene_type

def enlarge_background(background):
    big_background = np.zeros((3*HEIGHT, 3*WIDTH, 3), dtype=np.uint8)
    big_background[HEIGHT:2*HEIGHT, WIDTH:2*WIDTH, :] = background
    
    # Fill in the edges
    big_background[:HEIGHT, WIDTH:2*WIDTH, :] = background[0, :, :]
    big_background[2*HEIGHT:, WIDTH:2*WIDTH, :] = background[-1, :, :]
    big_background[HEIGHT:2*HEIGHT, :WIDTH, :] = background[:, 0, :][:, None, :]
    big_background[HEIGHT:2*HEIGHT, 2*WIDTH:, :] = background[:, -1, :][:, None, :]

    # Fill in the corners
    big_background[:HEIGHT, :WIDTH, :] = background[0, 0, :]
    big_background[:HEIGHT, 2*WIDTH:, :] = background[0, -1, :]
    big_background[2*HEIGHT:, :WIDTH, :] = background[-1, 0, :]
    big_background[2*HEIGHT:, 2*WIDTH:, :] = background[-1, -1, :]
    return big_background

def add_image_to_background(img, scene_type):
    if scene_type == 'Park':
        background = imread('big_park_background.png', mode='RGB')
    else:
        assert scene_type == 'Living'
        background = imread('big_living_background.png', mode='RGB')
    background[HEIGHT:2*HEIGHT, WIDTH:2*WIDTH, :] = img
    return background

def save_big_backgrouns():
    park_background = imread('/afs/csail.mit.edu/u/b/bce/TheoryOfMind/abstract_scenes_v002/site_pngs/Park/BG1.png', mode='RGB')
    big_park_background = enlarge_background(park_background)
    imsave('big_park_background.png', big_park_background)

    living_background = imread('/afs/csail.mit.edu/u/b/bce/TheoryOfMind/abstract_scenes_v002/site_pngs/Living/BG1.png', mode='RGB')
    big_living_background = enlarge_background(living_background)
    imsave('big_living_background.png', big_living_background)


def get_characters_in_frame(frame):
    '''
    Finds the characters who are present in some frame. Returns a list of
    object *instances*.
    ''' 
    character_vec = []
    for obj in frame['scene']['availableObject']:
        for instance in obj['instance']:
            if instance['present']:
                name = instance['name']
                if is_doll(name):
                    character_vec.append(instance)
    return character_vec

def is_doll(name):
    '''Checks whether an object's name is a doll name'''
    match = re.search('^Doll\d{2}?', name)
    return (match is not None)

def get_part_size(obj, part_name, renderer):
    '''
    Currently only supports body parts
    '''
    filename = renderer.paperdoll_part_img_filename_expr(obj, part_name)
    img = imread(filename)
    (h, w, _) = img.shape
    return (h, w)

def get_part_pose(obj, part_name, z_decay, img_pad_num, renderer):
    '''Computes the pose of an object's body part given an instance of the character.
    '''
    part_index = obj['partIdxList'][part_name]
    part = obj['body'][part_index]

    X1 = [-part['childX'], 
          -part['childY']]
    X = [obj['deformableX'][part_index], 
         obj['deformableY'][part_index]]
    rotation = obj['deformableGlobalRot'][part_index]
    flip = obj['flip']

    scale = obj['globalScale'] * (z_decay ** obj['z'])

    Tinvtuple = renderer.get_render_transform(X1, X, 
                                          rotation, flip, scale)

    T = np.matrix(list(Tinvtuple) + [0.0, 0.0, 1.0]).reshape((3, 3)).I

    (h, w) = get_part_size(obj, part_name, renderer)
    X = np.matrix([[0, w],
                   [0, h],
                   [1, 1]])


    coords = (T * X)
    x = np.mean(coords[0, :2])
    y = np.mean(coords[1, :2])

    global_rotation = obj['deformableGlobalRot'][part_index]

    return (x, y, global_rotation, flip)

def get_renderer(frame, img_filename, json_filename):
    '''
    img_filename and json_filename must be absolute paths
    '''
    assert os.path.isabs(img_filename)
    assert os.path.isabs(json_filename)
    frame['imgName'] = img_filename
    with open(json_filename, 'wb') as f:
        json.dump(frame, f)
    
    RENDER_ARGS = {
        '--config_dir': 'afs/csail.mit.edu/u/b/bce/TheoryOfMind/abstract_scenes_v002/site_data/',
        '--format': 'png',
        '--overwrite': True,
        '--site_pngs_dir': 'afs/csail.mit.edu/u/b/bce/TheoryOfMind/abstract_scenes_v002/site_pngs/',
        '<jsondata>': json_filename,
        '<outdir>': 'afs/csail.mit.edu/u/b/bce/public_html/TheoryOfMind/convSVM/',
        'render': True
    }
    renderer = RenderScenes(RENDER_ARGS)
    renderer.run()
    return renderer

def flip_character_in_frame(frame, character_name):
    flipped_frame = copy.deepcopy(frame)
    # Find the character object
    count = 0
    for obj in flipped_frame['scene']['availableObject']:
        for instance in obj['instance']:
            if instance['present'] and instance['name'] == character_name:
                instance['present'] = 0
                count += 1
    assert count == 1

    return flipped_frame

def balance(X, y, random_seed=0):
    np.random.seed(random_seed)
    neg_indices = np.where(y == 0)[0]
    pos_indices = np.where(y == 1)[0]
    num_neg = len(neg_indices)
    num_pos = len(pos_indices)
    
    assert type(num_neg) == int
    assert type(num_pos) == int
    assert num_pos + num_neg == len(y)
    if num_pos > num_neg:
        extra_indices = np.random.choice(neg_indices, num_pos - num_neg)
    else:
        extra_indices = np.random.choice(pos_indices, num_neg - num_pos)

    y = np.hstack([y, y[extra_indices]])
    if type(X) in [list, tuple]:
        X_balanced = []
        for item in X:
            if len(item.shape) == 1:
                X_balanced.append(np.hstack([item, item[extra_indices]]))
            else:
                X_balanced.append(np.vstack([item, item[extra_indices]]))
    else:
        X_balanced = np.vstack([X, X[extra_indices]])

    assert 0.49 < y.mean() < 0.51
    return (X_balanced, y)

def accuracy_score(y_true, y_hat):
    assert y_true.size == y_hat.size
    assert np.sum(y_true == 0) + np.sum(y_true == 1) == y_true.size
    assert np.sum(y_hat == 0) + np.sum(y_hat == 1) == y_hat.size
    return np.mean(y_true.flatten() == y_hat.flatten()) 
