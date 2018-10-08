########################################################################
# Imports

import matplotlib
matplotlib.use('Agg')

import argparse
import copy
import glob
import h5py
import json
import matplotlib.pyplot as plt
from multiprocessing import Pool
import numpy as np
import os
from render_scenes_json import RenderScenes
from scipy.misc import imsave, imresize
from scipy.ndimage import imread
import shutil
from sklearn.metrics import confusion_matrix, average_precision_score
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
import sys
from tqdm import tqdm
import util

#sys.path.append("/home/ngocthach/caffe/python")

########################################################################
# Global Variables

DATA_PATH = 'data'                              # Where to store temporary files
PUBLIC_PATH = 'public'                          # Where to show graphs and diagnostics
EGOCENTRIC_IMAGE_FOLDER = 'egocentric_images'   # Where egocentric images are stored
TMP_PATH = 'tmp'                                # Temporary directly. Deleted after running

########################################################################
# Command flags

COMPUTE_CHARACTER_FRAMES            = False
COMPUTE_RESNET_FEATURES          = False
COMPUTE_IMAGE_CNN_FEATURES          = False
CHECK_IMAGE_FEATURES                = False
COMPUTE_XYTF                        = False
COMPUTE_NAME_EXPRESSION             = False
COMPUTE_PRESENT                     = False
MAKE_QUESTION_DICT                  = False
MAKE_SPLITS                         = False
CONCATENATE_FEATURES                = False
MAKE_SPLIT_MASK                     = False
TRAIN_MODEL                         = True
MISTAKEN_TASK                       = True
WHO_TASK                            = True
WHEN_TASK                           = True

########################################################################
## Parse command line arguments
# Which features to use?
parser = argparse.ArgumentParser()
parser.add_argument('-USE_CAFFENET_FEATURE', action='store_true')
parser.add_argument('-USE_ANSWER', action='store_true')
parser.add_argument('-USE_XYF', action='store_true')
parser.add_argument('-USE_T', action='store_true')
parser.add_argument('-USE_NAME', action='store_true')
parser.add_argument('-USE_EXPRESSION', action='store_true')
parser.add_argument('-USE_PRESENT', action='store_true')
parser.add_argument('--LOOKAHEAD', type=int, default=0)
parser.add_argument('--LOOKBEHIND', type=int, default=0)
parser.add_argument('--RANDOM_SEED', type=int, default=0)
parser.add_argument('--C', type=float, default=1.0)
parser.add_argument('--LR', type=float, default=1e-5)
parser.add_argument('--MODEL', type=str, default='keras')

args = parser.parse_args()
print args

feature_args = [__file__, args.USE_CAFFENET_FEATURE, args.USE_ANSWER, args.USE_XYF, args.USE_T, args.USE_NAME, args.USE_EXPRESSION, args.USE_PRESENT, args.LOOKAHEAD, args.LOOKBEHIND]

experiment_args = [__file__, args.USE_CAFFENET_FEATURE, args.USE_ANSWER, args.USE_XYF, args.USE_T, args.USE_NAME, args.USE_EXPRESSION, args.USE_PRESENT, args.LOOKAHEAD, args.LOOKBEHIND, args.RANDOM_SEED, args.C, args.LR, args.MODEL]

feature_hash = abs(hash(str(feature_args)))
experiment_hash = abs(hash(str(experiment_args)))
#feature_hash = 2408
print 'Feature hash:', feature_hash
print 'Experiment hash:', experiment_hash


################################################################################
## Compute egocentric images

def compute_character_frames(frame):
    ''' Recenters and image at the character's head

    sceneId_frameIndex_characterName.png
    sceneId_frameIndex_characterName_flipped.png
    sceneId_frameIndex_original.png
    sceneId_frameIndex_centered.png

    '''
    character_vec = util.get_characters_in_frame(frame)
    scene_type = frame['scene']['sceneType']
    scene_id = frame['assignmentId']
    frame_index = frame['hitIdx']

    # Render original scene
    img_filename = os.path.join(TMP_PATH, '%s_%d_all.png' % (scene_id, frame_index))
    json_filename = os.path.join(TMP_PATH, '%s_%d_all.json' % (scene_id, frame_index))
    renderer = util.get_renderer(frame, img_filename, json_filename)
    scene_config = renderer.scene_config_data[scene_type]
    z_decay = scene_config['zSizeDecay']
    img_pad_num = scene_config['imgPadNum']

    img = imread(img_filename, mode='RGB')
    global_img = util.get_global_image(img)
    for character in character_vec:
        character_name = character['name']
        img_filename = os.path.join(EGOCENTRIC_IMAGE_FOLDER, '%s_%d_%s.png' % (scene_id, frame_index, character_name))
        pose = util.get_part_pose(character, 'Head', z_decay, img_pad_num, renderer)
        character_img = util.center_image_at_pose(global_img, pose)
        imsave(img_filename, character_img)


if COMPUTE_CHARACTER_FRAMES:
    # Running on 20 cores, takes about 35 min
    print 'Computing image features'

    print 'Loading json file containing scenes'
    with open(os.path.join(DATA_PATH, 'scenes.json')) as f:
        frame_vec = json.loads(f.read())

    print 'Done loading json file'
    pool = Pool(20)
    pool.map(compute_character_frames, frame_vec)

################################################################################
## Extract caffenet features for egocentric images

if COMPUTE_RESNET_FEATURES:
    from keras.applications.resnet50 import ResNet50
    from keras.applications import resnet50
    from keras.models import Model
    from keras.layers import Dense
    from keras.preprocessing import image

    FEATURE_RESNET50_PATH = 'resnet50_features_original_size'
    frame_dict_filename = os.path.join(DATA_PATH, 'frame_dict_resnet50_original.json')

    image_features_filename = os.path.join(DATA_PATH, 'egocentric_image_resnet50_features')
    main_image_features_filename = os.path.join(DATA_PATH, 'egocentric_image_resnet50_original_features')

    img_filename_vec = sorted(glob.glob(os.path.join(EGOCENTRIC_IMAGE_FOLDER,'*.png')))

    frame_dict = {os.path.basename(filename).split('.')[0] : index for (index, filename) in enumerate(img_filename_vec)}
    feature_length = 25*22*1024
    b_model = ResNet50(weights='imagenet', input_shape=(800, 1400, 3), include_top=False)
    model = Model(inputs=b_model.input, outputs=b_model.layers[-2].output)

    main_file = h5py.File(main_image_features_filename + '.hdf5', 'w')
    image_feature_matrix = main_file.create_dataset(image_features_filename, (len(img_filename_vec), feature_length))

    for loop in range(len(img_filename_vec)//3000):
        start_index = loop*3000
        end_index = start_index + 3000

        if end_index > len(img_filename_vec):
            end_index = len(img_filename_vec)

        image_features_filename_temp = image_features_filename + str(loop) + 'hdf5'
        f = h5py.File(image_features_filename_temp, 'r')
        #image_feature_matrix = f.create_dataset(image_features_filename, (3000, feature_length))
        #image_feature_matrix = np.zeros((len(img_filename_vec),feature_length))
        img_feature = f[image_features_filename]

        print("Start: End: ", start_index, ':', end_index)
        print("Filename: ", image_features_filename_temp)

        image_feature_matrix[start_index:end_index, :] = img_feature

        f.close()

        """
        used_indices=[]
        for (index, img_filename) in tqdm(enumerate(img_filename_vec[start_index:end_index]), total=len(img_filename_vec[start_index:end_index])):
            temp_img_filename = "".join(img_filename.split('/')[1:])
            #feature_filename = os.path.join(FEATURE_RESNET50_PATH, "".join(temp_img_filename.split('.')[:-1]) + '.npy')

            img = image.load_img(img_filename, target_size = (800,1400))
            x = image.img_to_array(img)
            x = np.expand_dims(x,axis=0)
            x = resnet50.preprocess_input(x)
            img_feature = model.predict(x)
            img_feature = img_feature.reshape((25,44,2048))
            print(img_feature.shape)
            assert img_feature.shape == (25, 44, 2048)
            img_feature = img_feature.reshape((25, 22, 2, 1024, 2)).max(axis=4).max(axis=2).flatten()
            #image_feature_matrix[index] = img_feature
            if os.path.exists(feature_filename):

                try:
                    #img_feature = np.load(feature_filename)
                except IOError:
                    img = image.load_img(img_filename, target_size = (224,224))

                    x = image.img_to_array(img)
                    x = np.expand_dims(x, axis = 0)
                    x = resnet50.preprocess_input(x)
                    img_feature = model.predict(x)

                    #np.save(feature_filename, img_feature)
            else:
                img = image.load_img(img_filename, target_size = (224,224))


                x = image.img_to_array(img)
                x = np.expand_dims(x, axis = 0)
                x = resnet50.preprocess_input(x)
                img_feature = model.predict(x)

                #np.save(feature_filename, img_feature)
            used_indices.append(index)
            image_feature_matrix[index] = img_feature

        #assert used_indices == range(len(img_filename_vec))
        #assert np.all(np.mean(image_feature_matrix != 0, axis=1) > 0)
        print("Feature shape: ", image_feature_matrix.shape)
        print("Frame dict: ", frame_dict)

        print("Save file")
        f.close()
        """
    print("Close file")
    main_file.close()
        #np.save(image_features_filename, image_feature_matrix)

if COMPUTE_IMAGE_CNN_FEATURES:

    print 'Extracting image cnn features'
    import caffe
    caffe.set_mode_gpu()
    caffe.set_device(1)
 
    layer = 'pool5'

    frame_dict_filename = os.path.join(DATA_PATH, 'frame_dict_cnn.json')

    image_features_filename = os.path.join(DATA_PATH, 'egocentric_image_features.npy')

    model_filename = os.path.join(DATA_PATH, 'deploy.prototxt')
    weights_filename = os.path.join(DATA_PATH, 'bvlc_reference_caffenet.caffemodel')

    img_filename_vec = sorted(glob.glob(os.path.join(EGOCENTRIC_IMAGE_FOLDER, '*.png')))
    frame_dict = {os.path.basename(filename).split('.')[0] : index for (index, filename) in enumerate(img_filename_vec)}


    with open(frame_dict_filename, 'wb') as f:
        json.dump(frame_dict, f)

    net = caffe.Net(model_filename, weights_filename, caffe.TEST)

    BATCH_SIZE = 50
    net.blobs['data'].reshape(BATCH_SIZE, 3, 800, 1400)
    net.forward(end=layer)
    MEAN_BGR = np.array([ 104.00698793,  116.66876762,  122.67891434])

    dimension = 256 * 12 * 21
    image_feature_matrix = np.zeros((len(img_filename_vec), dimension))
   
    used_indices = []
    net.blobs['data'].data[...] = 0
    for (index, img_filename) in tqdm(enumerate(img_filename_vec), total=len(img_filename_vec)):
        img = imread(img_filename, mode='RGB')
        assert img.shape == (800, 1400, 3)
        img = img[:, :, ::-1] # RGB -> BGR
        img = img.astype(np.float32)
        img -= MEAN_BGR[None, None, :]
        img = img.transpose((2, 0, 1)) # (800, 1400, 3) -> (3, 800, 1400)
        net.blobs['data'].data[index % BATCH_SIZE] = img
        if index % 50 == 49:
            net.forward(end=layer)
            for batch_index in range(BATCH_SIZE):
                img_index = index - BATCH_SIZE + 1 + batch_index
                used_indices.append(img_index)
                pool5 = net.blobs[layer].data[batch_index]
                assert pool5.shape == (256, 24, 43)
                pool5 = pool5[:, :, :-1]
                pool5 = pool5.reshape((256, 12, 2, 21, 2)).max(axis=4).max(axis=2).flatten()
                image_feature_matrix[img_index] = pool5.flatten()
            net.blobs['data'].data[...] = 0

    net.forward(end=layer)
    final_batch_size = len(img_filename_vec) % BATCH_SIZE
    for batch_index in range(final_batch_size):
        img_index = len(img_filename_vec) - final_batch_size + batch_index
        used_indices.append(img_index)

        pool5 = net.blobs[layer].data[batch_index]
        assert pool5.shape == (256, 24, 43)
        pool5 = pool5[:, :, :-1]
        pool5 = pool5.reshape((256, 12, 2, 21, 2)).max(axis=4).max(axis=2).flatten()
        image_feature_matrix[img_index] = pool5.flatten()

    # Make sure that we didn't miss any images
    assert used_indices == range(len(img_filename_vec))
    assert np.all(np.mean(image_feature_matrix != 0, axis=1) > 0)

    np.save(image_features_filename, image_feature_matrix)

################################################################################
## Visualize the caffenet features by using them for retrieval

if CHECK_IMAGE_FEATURES:
    print 'Checking image features'
    frame_dict_filename = os.path.join(DATA_PATH, 'frame_dict_cnn.json')
    image_features_filename = os.path.join(DATA_PATH, 'egocentric_image_features.npy')
    X = np.load(image_features_filename)

    with open(frame_dict_filename) as f:
        frame_dict = json.load(f)

    inverse_frame_dict = {feature_index: scene_frame_char for (scene_frame_char, feature_index) in frame_dict.items()}

    index = 0
    diff = np.sum((X - X[[index], :])**2, axis=1)
    indices = np.argsort(diff)
    print indices[:10]

    for i in range(0, 100, 10):
        feature_index = indices[i]
        scene_frame_char = inverse_frame_dict[feature_index]
        print scene_frame_char
        img_filename = os.path.join(EGOCENTRIC_IMAGE_FOLDER, '%s.png' % scene_frame_char)
        new_img_filename = os.path.join(PUBLIC_PATH, 'retrieve_1000_%05d.png' % i)
        shutil.copy(img_filename, new_img_filename)
 
################################################################################
## Extract (x,y,t,f) features for baseline experiments
# Running on 20 cores, takes 8 minutes

def get_pose(frame):
    key_vec = []
    pose_vec = []

    character_vec = util.get_characters_in_frame(frame)
    scene_type = frame['scene']['sceneType']
    scene_id = frame['assignmentId']
    frame_index = frame['hitIdx']
    frame_key = '%s_%d' % (scene_id, frame_index)

    # Render original scene
    img_filename = os.path.join(TMP_PATH, '%s_%d_all.png' % (scene_id, frame_index))
    json_filename = os.path.join(TMP_PATH, '%s_%d_all.json' % (scene_id, frame_index))
    renderer = util.get_renderer(frame, img_filename, json_filename)
    for character in character_vec:
        character_name = character['name']
        if character_name in ['all', 'none']:
            continue
        assert 'Doll' in character_name

        scene_config = renderer.scene_config_data[scene_type]
        z_decay =  scene_config['zSizeDecay']
        img_pad_num = scene_config['imgPadNum']
        pose = util.get_part_pose(character, 'Head', z_decay, img_pad_num, renderer)
        key = '%s_%d_%s' % (scene_id, frame_index, character_name)

        key_vec.append(key)
        pose_vec.append(pose)

    assert len(key_vec) == len(pose_vec)
    return (key_vec, pose_vec)

########################################################################
## Make XYTF dictionary

if COMPUTE_XYTF:
    print 'Computing xytf features'
    scenes_filename = os.path.join(DATA_PATH, 'scenes.json')
    with open(scenes_filename) as f:
        scene_vec = json.load(f)

    pool = Pool(20)
    pair_vec = pool.map(get_pose, scene_vec)

    xytf_dict = dict()
    for (key_vec, pose_vec) in pair_vec:
        for (key, pose) in zip(key_vec, pose_vec):
            assert key not in xytf_dict
            (x, y, _, f) = pose
            t = int(key.split('_')[1])
            xytf_dict[key] = {'x': x, 'y': y, 't': t, 'f': f}

    xytf_filename = os.path.join(DATA_PATH, 'xytf_dict.json')
    with open(xytf_filename, 'wb') as f:
        json.dump(xytf_dict, f)


########################################################################
## Make name expression dictionary

if COMPUTE_NAME_EXPRESSION:
    print 'Computing name expression dictionary'

    print 'Loading scene dictionary'
    with open(os.path.join(DATA_PATH, 'scenes.json')) as f:
        frame_vec = json.loads(f.read())
    print 'Done loading scene dictionary'

    name_lookup = []
    d = dict()
    for frame in tqdm(frame_vec):
        scene_id = frame['assignmentId']
        frame_index = frame['hitIdx']
        character_vec = util.get_characters_in_frame(frame)
        for character in character_vec:
            character_name = character['name']
            scene_frame_character = '%s_%d_%s' % (scene_id, frame_index, character_name)

            if character_name not in name_lookup:
                name_lookup.append(character_name)
            name_id = name_lookup.index(character_name)
            expression_id = character['expressionID']
            d[scene_frame_character] = {'name': name_id, 'expression': expression_id}
    num_names = max([name_expression['name'] for name_expression in d.values()]) + 1
    num_expressions = max([name_expression['expression'] for name_expression in d.values()]) + 1
    for (scene_frame_character, name_expression) in d.items():
        vec = [0] * num_names
        vec[name_expression['name']] = 1
        d[scene_frame_character]['name_one_hot'] = vec

        vec = [0] * num_expressions
        vec[name_expression['expression']] = 1
        d[scene_frame_character]['expression_one_hot'] = vec

    with open(os.path.join(DATA_PATH, 'name_expression_dict.json'), 'w') as f:
        json.dump(d, f)

 

if COMPUTE_PRESENT:
    print 'Computing precense dictionary'

    print 'Loading scene dictionary'
    with open(os.path.join('scenes.json')) as f:
        frame_vec = json.loads(f.read())
    print 'Done loading scene dictionary'

    s = set()
    for frame in tqdm(frame_vec):
        scene_id = frame['assignmentId']
        frame_index = frame['hitIdx']
        character_vec = util.get_characters_in_frame(frame)
        for character in character_vec:
            character_name = character['name']
            scene_frame_character = '%s_%d_%s' % (scene_id, frame_index, character_name)
            s.add(scene_frame_character)


    present_dict = dict()
    for frame in tqdm(frame_vec):
        scene_id = frame['assignmentId']
        frame_index = frame['hitIdx']
        character_vec = util.get_characters_in_frame(frame)
        for character in character_vec:
            character_name = character['name']
            scene_frame_character = '%s_%d_%s' % (scene_id, frame_index, character_name)
            present_vec = []
            inbounds_vec = []
            for other_frame_index in range(frame_index - 3, frame_index + 4):
                other_scene_frame_character = '%s_%d_%s' % (scene_id, other_frame_index, character_name)
                present_vec.append(int(other_scene_frame_character in s))
                inbounds_vec.append(int(0 <= other_frame_index and other_frame_index < 8))
            present_dict[scene_frame_character] = present_vec + inbounds_vec
  

    with open(os.path.join(DATA_PATH, 'present_dict.json'), 'w') as f:
        json.dump(present_dict, f)


########################################################################
## Make question dictionary

if MAKE_QUESTION_DICT:
    print 'Making question dictionary'

    questions_filename = os.path.join(DATA_PATH, 'question.json')

    with open(questions_filename) as f:
        question_json_vec = json.loads(f.read())

    question_dict = dict()
    question_vec = []
    count = 0
    for question_json in question_json_vec:
        questions = question_json['Answer.hitResult']['questions']
        answers_this_scene = question_json['Answer.hitResult']['answers']
        scene_id = question_json['Answer.hitResult']['sceneId']
        assert len(questions) == 8
        for (frame_index, q) in enumerate(questions):
            question_vec.append(q)
            key = '%s_%d' % (scene_id, frame_index)
            answers_this_frame = {character_name: answer_vec[frame_index] for (character_name, answer_vec) in answers_this_scene.items()}
            if key in question_dict:
                prev_value = question_dict[key]
                assert sorted(answers_this_scene.keys()) == sorted(prev_value['characters'])
                new_value = {
                    'indices': prev_value['indices'] + [count],
                    'questions': prev_value['questions'] + [q],
                    'answers': prev_value['answers'] + [answers_this_frame],
                    'characters': answers_this_scene.keys()
                }
            else:
                new_value = {
                    'indices': [count],
                    'questions': [q],
                    'answers': [answers_this_frame],
                    'characters': answers_this_scene.keys()
                }

            question_dict[key] = new_value
            count += 1

    assert count == len(question_vec)

    print 'Total number of questions:', len(question_vec)
   
    # Save the mapping from frames to question indices
    # Because frames can have multiple questions, there may be multiple indices per frame
    question_dict_filename = os.path.join(DATA_PATH, 'question_dict.json')
    with open(question_dict_filename, 'wb') as f:
        json.dump(question_dict, f)

  
########################################################################
# Train/Test splits
# Create a dictionary which maps scene_ids to 'train' or 'test'

if MAKE_SPLITS:
    print 'Making train/test splits'
    with open(os.path.join(DATA_PATH, 'scenes.json')) as f:
        scene_vec = json.loads(f.read())
    scene_id_set = set()
    for scene in scene_vec:
        scene_id = scene['assignmentId']
        scene_id_set.add(scene_id)

    scene_id_list = list(scene_id_set)
    np.random.seed(args.RANDOM_SEED)
    np.random.shuffle(scene_id_list)
    train_fraction = 0.8
    val_fraction = 0.1
    test_fraction = 0.1
    assert train_fraction + val_fraction + test_fraction == 1
    train_val_cutoff = int(train_fraction * len(scene_id_list))
    val_test_cutoff = int((train_fraction + val_fraction) * len(scene_id_list))
    split_dict = dict()
    for (index, scene_id) in enumerate(scene_id_list):
        if index < train_val_cutoff:
            split_dict[scene_id] = 'train'
        elif train_val_cutoff <= index < val_test_cutoff:
            split_dict[scene_id] = 'val'
        else:
            split_dict[scene_id] = 'test'

    print 'Number of scenes for train set: ', sum([split == 'train' for split in split_dict.values()])
    print 'Number of scenes for test set: ', sum([split == 'test' for split in split_dict.values()])
    print 'Number of scenes for validation set: ', sum([split == 'val' for split in split_dict.values()])
    print 'Number of scenes total: ', len(split_dict.values())

    with open(os.path.join(DATA_PATH, 'split_dict_%02d.json' % args.RANDOM_SEED), 'wb') as f:
        json.dump(split_dict, f)


########################################################################
# CONCATENATE FEATURES

if CONCATENATE_FEATURES:
    print 'Concatenating features'
    feature_hash = 200000000000
    experiment_hash = 200000000000

    ANY_IMAGE_FEATURE = args.USE_CAFFENET_FEATURE
    POOL = False
    if args.USE_CAFFENET_FEATURE:
        frame_dict_filename = os.path.join(DATA_PATH, 'frame_dict_resnet50_original.json')
        print("Load egocentric_image_features.npy file")
        image_features_filename = os.path.join(DATA_PATH, 'egocentric_image_resnet50_features')
        main_image_features_filename = os.path.join(DATA_PATH, 'egocentric_image_resnet50_original_features')
        main_file = h5py.File(main_image_features_filename + '.hdf5', 'r')
        image_feature_matrix = main_file[image_features_filename][:]

        #image_feature_matrix = np.load(image_features_filename)
        #image_feature_matrix = np.load(image_features_filename)

        print("Load frame_dict_cnn.json file")
        with open(frame_dict_filename) as f:
            frame_dict = json.load(f)

        pool5_size = (256, 12, 21)

    else:
        print 'WARNING: no image features used.\n'
        frame_dict_filename = os.path.join(DATA_PATH, 'frame_dict_cnn.json')
        with open(frame_dict_filename) as f:
            frame_dict = json.load(f)


    print("Load other feature")
    question_dict_filename = os.path.join(DATA_PATH, 'question_dict.json')
    xytf_filename = os.path.join(DATA_PATH, 'xytf_dict.json')
    name_expression_filename = os.path.join(DATA_PATH, 'name_expression_dict.json')
    present_filename = os.path.join(DATA_PATH, 'present_dict.json')

    with open(question_dict_filename) as f:
        question_dict = json.load(f)

    with open(xytf_filename) as f:
        xytf_dict = json.load(f)

    with open(name_expression_filename) as f:
        name_expression_dict = json.load(f)

    with open(present_filename) as f:
        present_dict = json.load(f)

    # --------------- NOTICE:  Just use for CNN Features  -------------------
    # find size of features
    num_samples = 0
    for scene_frame_character in tqdm(frame_dict.keys()):
        if 'flipped' in scene_frame_character:
            continue
        (scene_id, frame_index, character_name) = scene_frame_character.split('_')
        frame_index = int(frame_index)
        frame_key = '%s_%d' % (scene_id, frame_index)
        question_vec = question_dict[frame_key]['questions']
        answer_dict_vec = question_dict[frame_key]['answers']
        if character_name not in ['all', 'none', 'centered']:
            assert 'Doll' in character_name, 'Character name: %s' % character_name

            answer_vec = []
            for answer_dict in answer_dict_vec:
                answer = int(answer_dict[character_name] == answer_dict['omniscient'])
                answer_vec.append(answer)
            if answer_vec[0] != answer_vec[1]:
                continue

            num_samples += 1

    #X = np.empty((num_samples, pool5_size[0]*pool5_size[1]*pool5_size[2]))
    # --------------- ---- -------------------

    ### open file h5py ####
    feature_length = 25*22*1024
    f = h5py.File('X_%d.hdf5' % feature_hash, 'w')
    X = f.create_dataset('X_%d' % feature_hash, (num_samples, feature_length*7), dtype='f')

    #X = []
    y = []
    t = []

    scene_id_vec = []
    image_filename_vec = []
    concat_indices = dict()
    index = 0
    for scene_frame_character in tqdm(frame_dict.keys()):
        if 'flipped' in scene_frame_character:
            continue
        (scene_id, frame_index, character_name) = scene_frame_character.split('_')
        frame_index = int(frame_index)
        frame_key = '%s_%d' % (scene_id, frame_index)
        question_vec = question_dict[frame_key]['questions']
        answer_dict_vec = question_dict[frame_key]['answers']

        if character_name not in ['all', 'none', 'centered']:
            assert 'Doll' in character_name, 'Character name: %s' % character_name

            if ANY_IMAGE_FEATURE:
                image_feature_vec = []
                for other_frame_index in range(frame_index - args.LOOKBEHIND, frame_index + args.LOOKAHEAD + 1):
                    key = '%s_%d_%s' % (scene_id, other_frame_index, character_name)
                    print("Key: ", key)

                    if key in frame_dict.keys():
                        image_feature_index = frame_dict[key]
                        pool5 = image_feature_matrix[image_feature_index]
                    else: # Character is not present
                        pool5 = np.zeros(feature_length)

                    image_feature_vec.append(pool5)

                image_feature = np.hstack(image_feature_vec)

            answer_vec = []
            for answer_dict in answer_dict_vec:
                answer = int(answer_dict[character_name] == answer_dict['omniscient'])
                answer_vec.append(answer)
            if answer_vec[0] != answer_vec[1]:
                continue

            # Only used for visualization
            question = question_vec[0]
            image_filename_vec.append('%s_%02d.png' % (scene_id, frame_index))

            answer_dict = answer_dict_vec[0]
            assert answer_dict[character_name] in ['yes', 'no']
            answer = int(answer_dict[character_name] == answer_dict['omniscient'])
            assert answer == answer_vec[0]

            xytf = xytf_dict[scene_frame_character]

            feature = []
            if ANY_IMAGE_FEATURE:	# CNN feature
                feature.append(image_feature)
            if args.USE_ANSWER:	# Answer ???
                feature.append([float(answer)])
            if args.USE_XYF: # Pose feature
                feature.append([xytf['x'] / 700.0, xytf['y'] / 400.0])
            if args.USE_T:	# time feature # vector 7 elements
                feature.append([xytf['t'] / 8.0])
            if args.USE_NAME:	# character ID feature
                feature.append(name_expression_dict[scene_frame_character]['name_one_hot'])
            if args.USE_EXPRESSION:	# Facial Expression feature
                feature.append(name_expression_dict[scene_frame_character]['expression_one_hot'])
            if args.USE_PRESENT:	# Present Feature # vector 14 elements
                feature.append(present_dict[scene_frame_character])


            #index = len(X)
            #assert len(X) == len(y)
            #assert scene_frame_character not in concat_indices
            #concat_indices[scene_frame_character] = index

            assert index == len(y)
            assert scene_frame_character not in concat_indices
            concat_indices[scene_frame_character] = index

            #X[index] = feature
            X[index] = np.hstack(feature)
            index += 1
            #X.append(np.hstack(feature))
            y.append(answer)
            t.append(xytf['t'])
            scene_id_vec.append(scene_id)

    #X = np.array(X)
    y = np.array(y)
    t = np.array(t)
    assert X.shape[0] == len(y) == len(t)



    print 'X.shape:', X.shape
    print 'y.shape:', y.shape
    print 'Number of rows:', len(X)
    print 'Number of unique rows:', len(set([hash(row.tostring()) for row in X]))

    #np.save('X_%d.npy' % feature_hash, X)
    f.close()
    np.save('y_%d.npy' % feature_hash, y)
    np.save('t_%d.npy' % feature_hash, t)

    with open('concat_indices_%d.json' % feature_hash, 'wb') as f:
        json.dump(concat_indices, f)
 
    with open('scene_id_vec_%d.json' % feature_hash, 'wb') as f:
        json.dump(scene_id_vec, f)

    with open('image_filename_vec_%d.json' % feature_hash, 'wb') as f:
        json.dump(image_filename_vec, f)


########################################################################
## Make split mask
# Uses the split_dict and list of scene_ids from the generated dataset to create a vector with entries 'train', 'val', test'
# Note: we cannot do this without seeing the concatenated features because we use a different frame_dict for differen features

if MAKE_SPLIT_MASK:
    split_dict_filename = os.path.join(DATA_PATH, 'split_dict_%02d.json' % args.RANDOM_SEED)
    scene_id_vec_filename = 'scene_id_vec_%d.json' % feature_hash

    with open(split_dict_filename) as f:
        split_dict = json.load(f)

    with open(scene_id_vec_filename) as f:
        scene_id_vec = json.load(f)

    split_mask = []
    for scene_id in scene_id_vec:
        assert split_dict[scene_id] in ['train', 'val', 'test']
        split_mask.append(split_dict[scene_id])

    split_mask = np.array(split_mask)
    np.save('split_mask_%d.npy' % experiment_hash, split_mask)

######################################################################
### CODE THACH ###
### LOAD BATCH WHEN INPUT LARGE ###
def load_batch(indices, experiment_hash, batch_index, batch_size=32):
    """

    :param indices:
    :param batch_size:
    :return:
    """
    # check if exist file batch
    filename = os.path.join('batch_files', '%d_%d'%(feature_hash, batch_index) + '.npy')
    if os.path.exists(filename):
        x = np.load(filename)
    else:
        x = []
        # loop over indices:
        for index in indices:
            x.append(np.load(os.path.join('resnet50_concatenate_features', str(feature_hash) + '_' + str(index) + '.npy')))
        x = np.array(x).reshape((len(indices), 7*7*512*7))
        x = x.clip(0, 1)
        np.save(os.path.join('batch_files', '%d_%d'%(feature_hash, batch_index) + '.npy'), x)
    return x

########################################################################
# Train the SVM
print("Loading file")
#feature_hash = 2408
experiment_hash_list = [4754952150018512305, 8647626824687750802, 8949258729311709173, 1182471099183327604]
#experiment_hash_list = [8647626824687750802, 8949258729311709173, 1182471099183327604]
#experiment_hash_list = [1182471099183327604]
#X = np.load('X_%d.npy' % feature_hash)
f = h5py.File('X_%d.hdf5' % feature_hash, 'r')
X = np.array(f['X_%d' % feature_hash])
#print 'Get original feature'
#X = X[:, :-15]

args.RANDOM_SEED = 0
for experiment_hash in experiment_hash_list[1:2]:
    print("#########################################################")
    print("Experiment hash: ", experiment_hash)
    args.RANDOM_SEED += 1
    if TRAIN_MODEL:
        print 'TRAINING MODEL'
        print 'Loading data'

        #X = np.load('X_%d.npy' % feature_hash)
        #f = h5py.File('X_%d1909.hdf5' % feature_hash, 'r')
        #print 'Loading X...'
        #X = np.array(f['X_%d1909' % feature_hash])
        #print 'Load done...'
        #print 'Get original feature'
        #X = X[:, :-15]
        #print("Clip")
        #X = X.clip(0,1)
        #print("Ceate file")
        #f2 = h5py.File('X_%d1909.hdf5' % feature_hash, 'w')
        #print("create dataset")
        #X2 = f2.create_dataset('X_%d1909' % feature_hash, data=X)
        #f2.close()
        #print("Done")
        #X_nor_cnn = X[:,0:-15]/np.sum(X[:,0:-15],axis = 1)[:,None]
        #X_nor_present = X[:,-14:]/np.sum(X[:,-14:],axis=1)[:,None]

        #X[:,0:-15]=X_nor_cnn
        #X[:,-14:]=X_nor_present
        #X[:, 0:-15] = (X[:,0:-15]-np.mean(X[:,0:-15]))/(np.max(X[:,0:-15]) - np.min(X[:, 0:-15]))/(X.shape[1]-15)
        #X[:, -14:] = (X[:,-14:] - np.mean(X[:,-14:]))/(np.max(X[:,-14:]) - np.min(X[:, -14]))/14
        #X[:, -15] = (X[:,-15] - np.mean(X[:, -15]))/(np.max(X[:, -15]) - np.min(X[:, -15]))

        ### Standardization feature ####
        from sklearn import preprocessing
        #X = preprocessing.scale(X)
        #X = preprocessing.normalize(X)

        # CNN feature
        #X[:, 0:-15] = preprocessing.scale(X[:,0:-15])
        #X[:, 0:-15] = preprocessing.normalize(X[:, 0:-15], norm='l2')
        # Time feature
        #X[:, -15] = preprocessing.scale(X[:,-15])
        #X[:, -15] = preprocessing.normalize(X[:, -15], norm='l2')
        # Present feature
        #X[:, -14:] = preprocessing.scale(X[:,-14:])
        #X[:, -14:] = preprocessing.normalize(X[:, -14:], norm='l2')

        #X = preprocessing.normalize(X)
        print('X_shape: ', X.shape)
        print 'Number of unique rows X:', len(set([hash(row.tostring()) for row in X]))
        y = np.load('y_%d.npy' % feature_hash)
        t = np.load('t_%d.npy' % feature_hash)
        #assert X.shape[0] == y.shape[0] == t.shape[0]
        split_mask = np.load('split_mask_%d.npy' % experiment_hash)
        print("Load Done...")
        tempLength = 7
        #feature_length = X.shape[1]//7
        #X = X[:, feature_length:-feature_length]
        #print('X_shape: ', X.shape)

        X_train = X[split_mask == 'train']
        X_val = X[split_mask == 'val']
        X_test = X[split_mask == 'test']

        y_train = y[split_mask == 'train']
        y_val = y[split_mask == 'val']
        y_test = y[split_mask == 'test']

        t_train = t[split_mask == 'train']
        t_val = t[split_mask == 'val']
        t_test = t[split_mask == 'test']

        X_train = X_train.clip(0, 1)
        X_val = X_val.clip(0, 1)
        X_test = X_test.clip(0, 1)


        """
        ####### Because of large input -> train by batch ###
        ## Read X
        num_samples = 11955
        X = np.array(list(range(num_samples)))
        X_train = X[split_mask == 'train']
        X_val = X[split_mask == 'val']
        X_test = X[split_mask == 'test']

        y_train = y[split_mask == 'train']
        y_val = y[split_mask == 'val']
        y_test = y[split_mask == 'test']

        t_train = t[split_mask == 'train']
        t_val = t[split_mask == 'val']
        t_test = t[split_mask == 'test']
        """

        ############################

        assert X_train.shape[0] == y_train.shape[0] == t_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0] == t_val.shape[0]
        assert X_test.shape[0] == y_test.shape[0] == y_test.shape[0]

        ((X_train, t_train), y_train) = util.balance((X_train, t_train), y_train)
        ((X_val, t_val), y_val) = util.balance((X_val, t_val), y_val)
        ((X_test, t_test), y_test) = util.balance((X_test, t_test), y_test)

        assert X_train.shape[0] == y_train.shape[0] == t_train.shape[0]
        assert X_val.shape[0] == y_val.shape[0] == t_val.shape[0]
        assert X_test.shape[0] == y_test.shape[0] == y_test.shape[0]

        if args.MODEL == 'keras':

            from keras.models import Sequential
            from keras.layers import Dense
            from keras.optimizers import Adam
            from keras.regularizers import l2
            from keras.callbacks import EarlyStopping
            from keras.layers import Bidirectional
            from keras.layers import LSTM
            import math
            print 'Constructing model'


            model = Sequential()
            #model.add(Dense(output_dim=7,
            #                input_shape=(X.shape[1],),
            #               W_regularizer=l2(args.C), activation='relu'))
            #model.add(Dense(output_dim=3,
            #               W_regularizer=l2(args.C), activation='relu'))
            model.add(Dense(output_dim=1,
                            input_shape=(X.shape[1],),
                            W_regularizer=l2(args.C), activation='sigmoid'))
            print model.summary()

            #np.random.seed(args.RANDOM_SEED)
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.LR), metrics=['accuracy'])
            
            print 'Training model'
            history = model.fit(X_train, y_train, nb_epoch=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[EarlyStopping(patience=3)])
            model.save('model.h5')

            # list all data in history
            print(history.history.keys())
            # summarize history for accuracy
            plt.plot(history.history['acc'])
            plt.plot(history.history['val_acc'])
            plt.title('model accuracy')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(['train', 'test'], loc='upper left')
            plt.show()
            # summarize history for loss
            plt.plot(history.history['loss'])
            plt.plot(history.history['val_loss'])
            plt.title('model loss')
            plt.ylabel('loss')
            plt.xlabel('epoch')
            plt.legend(['train', 'val'], loc='upper left')
            plt.show()

            # save history ##
            import pickle
            with open('training_history_%d' % experiment_hash, mode='wb') as training_history:
                pickle.dump(history.history, training_history, protocol=pickle.HIGHEST_PROTOCOL)

            """
            ####### Because of large input -> train by batch ###
            ## Use train_on_batch instead of fit ##
            batch_size = 32
            num_batches = math.ceil(len(X_train)/32)
            print("X_train: ", len(X_train))
            print("#batches: ", num_batches)
            for epoch in tqdm(range(100)):
                for batch in range(int(num_batches)):

                    if batch < num_batches - 1:
                        batch_index = X_train[batch*32:batch*batch_size+batch_size]
                        y = y_train[batch*32:batch*batch_size+batch_size]
                    else:
                        batch_index = X_train[batch*batch_size:]
                        y = y_train[batch*32:batch*batch_size+batch_size]
                    x = load_batch(indices=batch_index, experiment_hash=experiment_hash, batch_index=batch, batch_size=batch_size)
                    model.train_on_batch(x, y)

            #model.save('model.h5')
            print("Test model on validation set")

            #### Test on batch ###
            ### Test on validation data ###
            num_batches = math.ceil(len(X_val)/32)
            print("X_val: ", len(X_val))
            print("#batches: ", num_batches)
            for epoch in tqdm(range(100)):
                for batch in range(int(num_batches)):

                    if batch < num_batches - 1:
                        batch_index = X_val[batch*32:batch*batch_size+batch_size]
                        y = y_val[batch*32:batch*batch_size+batch_size]
                    else:
                        batch_index = X_val[batch*batch_size:]
                        y = y_val[batch*32:batch*batch_size+batch_size]
                    x = load_batch(indices=batch_index, experiment_hash=experiment_hash, batch_index=batch, batch_size=batch_size)
                    model.test_on_batch(x, y)

            print("Predict all dataset")
            ### Predict on batch ###
            ## Predict all examples and save to y_hat and y_prob ##
            y_prob, y_hat = [], []
            num_batches = math.ceil(len(X)/32)
            print("X: ", len(X))
            print("#batches: ", num_batches)
            for epoch in tqdm(range(100)):
                for batch in range(int(num_batches)):

                    if batch < num_batches - 1:
                        batch_index = X[batch*32:batch*batch_size+batch_size]
                        #y = y[batch*32:batch*batch_size+batch_size]
                    else:
                        batch_index = X[batch*batch_size:]
                        #y = y[batch*32:batch*batch_size+batch_size]
                    x = load_batch(indices=batch_index, experiment_hash=experiment_hash, batch_index=batch, batch_size=batch_size)
                    y_prob_batch = model.predict_proba(X)
                    y_hat_batch = model.predict_classes(X)
                    y_prob = y_prob + y_prob_batch
                    y_hat  = y_hat + y_hat_batch

            np.save('y_prob_%d.npy' % experiment_hash, y_prob)
            np.save('y_hat_%d.npy' % experiment_hash, y_hat)


            ##################################################



            print("Reshape X Train...")
            tempLength = args.LOOKAHEAD + args.LOOKBEHIND + 1
            X_train = X_train.reshape((X_train.shape[0], tempLength, X_train.shape[1]//tempLength))
            X_val = X_val.reshape((X_val.shape[0], tempLength, X_val.shape[1]//tempLength))
            X_test = X_test.reshape((X_test.shape[0], tempLength, X_test.shape[1]//tempLength))

            model = Sequential()
            model.add(LSTM(1, return_sequences=False, input_shape=(X_train.shape[1], X_train.shape[2])))
            model.add(Dense(1, W_regularizer=l2(args.C), activation='sigmoid'))
            model.summary()
            #np.random.seed(args.RANDOM_SEED)
            model.compile(loss='binary_crossentropy', optimizer=Adam(lr=args.LR), metrics=['accuracy'])
            print 'Training model'
            history = model.fit(X_train, y_train, nb_epoch=100, batch_size=32, validation_data=(X_val, y_val), callbacks=[EarlyStopping(patience=3)])
            model.save('model.h5')

            import pickle
            with open('training_history_%d' % experiment_hash, mode='wb') as training_history:
                pickle.dump(history.history, training_history, protocol=pickle.HIGHEST_PROTOCOL)
            """
        elif args.MODEL == 'svm':
            model = SVC(C=args.C, kernel='rbf')
            model.fit(X_train, y_train)
        elif args.MODEL == 'logistic_regression':
            model = LogisticRegression()
            model.fit(X_train, y_train)
        else:
            assert False, 'Unknown model: %d' % args.MODEL


        print 'Prediction X'
        #X = np.array(f['X_%d' % feature_hash])

        if args.MODEL == 'keras':
            model.save('model_%d.h5' % experiment_hash)
            #model.load_weights(os.path.join('results/alexnet_7_1___2', 'model.h5'))
            #X = X.reshape((X.shape[0], tempLength, X.shape[1]//tempLength))
            y_prob = model.predict_proba(X)
            y_hat = model.predict_classes(X)
        elif args.MODEL == 'svm':
            y_prob = 1.0 / (1.0 + np.exp(-model.decision_function(X)))
            y_hat = model.predict(X)
        elif args.MODEL == 'logistic_regression':
            y_hat = model.predict(X)
            y_prob = model.predict_proba(X)[:, 1].flatten()
        else:
            assert False, 'Unknown model: %d' % args.MODEL

        """
        print("Saving filename")
        count = 0
        while True:
            temp_str = '_%d' % count
            if os.path.exists('y_prob_%d%s.npy' % (experiment_hash, temp_str)):
                np.save('y_prob_%d%s.npy' % (experiment_hash, temp_str), y_prob)
                np.save('y_hat_%d%s.npy' % (experiment_hash, temp_str), y_hat)
                break
            else:
                count += 1
        """
        print("Save")
        np.save('y_prob_%d.npy' % experiment_hash, y_prob)
        np.save('y_hat_%d.npy' % experiment_hash, y_hat)

    ####################################################################################

    # Mistaken task:
    # 1. Given a character and a frame, is the character mistaken in this frame?
    # 2. Given a character and a scene, is the character mistaken in any frame in the scene?
    if MISTAKEN_TASK:
        print 'Mistaken task'
        y = np.load('y_%d.npy' % feature_hash)
        y_prob = np.load('y_prob_%d.npy' % experiment_hash)
        t = np.load('t_%d.npy' % feature_hash)
        y_hat = np.load('y_hat_%d.npy' % experiment_hash)
        split_mask = np.load('split_mask_%d.npy' % experiment_hash)

        y_prob_test = y_prob[split_mask == 'test']
        y_train = y[split_mask == 'train']
        y_val = y[split_mask == 'val']
        y_test = y[split_mask == 'test']

        y_train_hat = y_hat[split_mask == 'train']
        y_val_hat = y_hat[split_mask == 'val']
        y_test_hat = y_hat[split_mask == 'test']

        t_train = t[split_mask == 'train']
        t_val = t[split_mask == 'val']
        t_test = t[split_mask == 'test']

        print 'Mistaken Task - single frame'

        print 'Training Accuracy: %.3f%%' % (100.0 * util.accuracy_score(y_train, y_train_hat))
        print 'Confusion matrix:'
        print confusion_matrix(y_train, y_train_hat)
        print 'Accuracy vs time:', [100.0 * util.accuracy_score(y_train[t_train == i], y_train_hat[t_train == i]) for i in range(8)]
        print '\n\n'
        print 'Validation Accuracy: %.3f%%' % (100.0 * util.accuracy_score(y_val, y_val_hat))
        print 'Confusion matrix:'
        print confusion_matrix(y_val, y_val_hat)
        print 'Accuracy vs time:', [100.0 * util.accuracy_score(y_val[t_val == i], y_val_hat[t_val == i]) for i in range(8)]
        print '\n\n'

        print 'Test Accuracy: %.3f%%' % (100.0 * util.accuracy_score(y_test, y_test_hat))
        print 'Confusion matrix:'
        print confusion_matrix(y_test, y_test_hat)
        print 'Accuracy vs time:', [100.0 * util.accuracy_score(y_test[t_test == i], y_test_hat[t_test == i]) for i in range(8)]
        print '\n\n'
        ((y_test_hat, y_prob_test), y_test) = util.balance((y_test_hat, y_prob_test), y_test)
        print 'Mistaken[frame]: Accuracy: %.3f%%' % (100.0 * util.accuracy_score(y_test, y_test_hat))
        print 'Mistaken[frame]: AP: %.3f' % average_precision_score(y_test, y_prob_test)
        print 'Mistaken[frame]: Chance: %.3f%%' % (100.0 * np.mean(y_test == 1))
        print 'Mistaken[frame]: Chance AP: %.3f' % average_precision_score(y_test, np.mean(y_test) * np.ones_like(y_test))
        print

        # Mistaken[scene] - given a character and a scene, is the character mistaken in any frame in the scene?

        question_dict_filename = os.path.join(DATA_PATH, 'question_dict.json')

        split_dict_filename = os.path.join(DATA_PATH, 'split_dict_%02d.json' % args.RANDOM_SEED)


        with open(split_dict_filename) as f:
            split_dict = json.load(f)

        with open(question_dict_filename) as f:
            question_dict = json.load(f)

        with open('concat_indices_%d.json' % feature_hash, 'rb') as f:
            concat_indices = json.load(f)


        val_min_predict = []
        val_avg_predict = []
        val_y = []
        test_min_predict = []
        test_avg_predict = []
        test_y = []

        for frame_key in tqdm(question_dict):
            (scene_id, frame_index) = frame_key.split('_')
            frame_index = int(frame_index)

            # We only want to loop through each scene once
            if frame_index != 0:
                continue

            # We only want to evaluate on characters who were mistaken in every frame or correct in every frame
            # In the loop below, the answers for characters who are not in the frame may be incorrect.
            # This is OK, because in the second loop, we only look at frames whether the model made predictions (i.e. the character was present)

            confused_dict = dict()

            for frame_index in range(8):
                frame_key = '%s_%d' % (scene_id, frame_index)
                answer_dict_vec = question_dict[frame_key]['answers']
                character_vec = question_dict[frame_key]['characters']
                for character_name in character_vec:
                    key = '%s_%s_%s' % (scene_id, frame_index, character_name)
                    if 'Doll' in character_name and key in concat_indices:
                        answer_vec = []
                        for answer_dict in answer_dict_vec:
                            answer = int(answer_dict[character_name] == answer_dict['omniscient'] )
                            answer_vec.append(answer)

                        if answer_vec[0] == answer_vec[1]:
                            confused_dict[character_name] = [answer] + confused_dict.get(character_name, [])


            for (character_name, answer_vec) in confused_dict.items():
                correct = min(answer_vec)

                predict = []
                for frame_index in range(8):
                    key = '%s_%s_%s' % (scene_id, frame_index, character_name)
                    # A character may appear in only a subset of the 8 frames. We
                    # predict whether he is mistaken based on predictions from those
                    # frames.
                    if key in concat_indices:
                        index = concat_indices[key]
                        predict.append(y_prob[index])

                assert len(predict) > 0
                min_predict = min(predict)
                avg_predict = np.mean(predict)

                if split_dict[scene_id] == 'val':
                    val_min_predict.append(min_predict)
                    val_avg_predict.append(avg_predict)
                    val_y.append(correct)

                if split_dict[scene_id] == 'test':
                    test_min_predict.append(min_predict)
                    test_avg_predict.append(avg_predict)
                    test_y.append(correct)

        test_y = np.array(test_y)
        test_min_predict = np.array(test_min_predict)
        test_avg_predict = np.array(test_avg_predict)

        ((test_min_predict, test_avg_predict), test_y) = util.balance((test_min_predict, test_avg_predict), test_y)


        print 'Mistaken[scene]: Accuracy (min pool): %.3f%%' % (100.0 * util.accuracy_score(test_y, test_min_predict > 0.5))
        print 'Mistaken[scene]: AP (min pool): %.3f' % average_precision_score(test_y, test_min_predict)
        print 'Mistaken[scene]: Accuracy (avg pool): %.3f%%' % (100.0 * util.accuracy_score(test_y, test_avg_predict > 0.5))
        print 'Mistaken[scene]: AP (avg pool): %.3f' % average_precision_score(test_y, test_avg_predict)
        print 'Mistaken[scene]: Chance: %.3f%%' % (100.0 * max(np.mean(test_y), 1 - np.mean(test_y)))
        print 'Mistaken[scene]: Chance AP: %.3f' % average_precision_score(test_y, np.mean(test_y) * np.ones_like(test_y))
        print



    ####################################################################################


    if WHO_TASK:
        question_dict_filename = os.path.join(DATA_PATH, 'question_dict.json')

        split_dict_filename = os.path.join(DATA_PATH, 'split_dict_%02d.json' % args.RANDOM_SEED)

        y_prob = np.load('y_prob_%d.npy' % experiment_hash)

        with open(split_dict_filename) as f:
            split_dict = json.load(f)

        with open(question_dict_filename) as f:
            question_dict = json.load(f)

        with open('concat_indices_%d.json' % feature_hash, 'rb') as f:
            concat_indices = json.load(f)

        # Who task for a single frame:
        # Given a frame with a mistaken character and a non-mistaken character, predict which character is mistaken
        # Note: If a frame contains more than two characters, we will compute this for all pairs of (mistaken, non-mistaken) characters in this frame

        correct = 0
        incorrect = 0
        predict_vec = []
        y_vec = []
        for frame_key in tqdm(question_dict):
            (scene_id, frame_index) = frame_key.split('_')
            if split_dict[scene_id] == 'test':
                answer_dict_vec = question_dict[frame_key]['answers']
                answer_vec = []
                confused_dict = dict()
                for answer_dict in answer_dict_vec:
                    for character_name in answer_dict:
                        if 'Doll' in character_name:
                            answer = int(answer_dict[character_name] == answer_dict['omniscient'])
                            confused_dict[character_name] = [answer] + confused_dict.get(character_name, [])
                consistent_confused_dict = dict()
                for (character_name, answers) in confused_dict.items():
                    if answers[0] == answers[1]:
                        consistent_confused_dict[character_name] = answers[0]
                for (character_name_1, correct_1) in consistent_confused_dict.items():
                    for (character_name_2, correct_2) in consistent_confused_dict.items():
                        if correct_1 == 0 and correct_2 == 1:
                            assert character_name_1 != character_name_2
                            key_1 = '%s_%s_%s' % (scene_id, frame_index, character_name_1)
                            key_2 = '%s_%s_%s' % (scene_id, frame_index, character_name_2)
                            if key_1 in concat_indices and key_2 in concat_indices:
                                index_1 = concat_indices[key_1]
                                index_2 = concat_indices[key_2]
                                if y_prob[index_1] < y_prob[index_2]:
                                    correct += 1
                                elif y_prob[index_1] == y_prob[index_2]:
                                    correct += 0.5
                                    incorrect += 0.5
                                else:
                                    incorrect += 1

        print 'Who[frame]: Accuracy: %.3f%%' % (100.0 * float(correct) / (correct + incorrect))

        # Who task for a scene:
        # Given a scene with a mistaken character and a non-mistaken character, predict which character is mistaken
        # Note: If a scene contains more than two characters, we will compute this for all pairs of (mistaken, non-mistaken) characters in this scene

        # We can aggregate predictions across frames by taking the mean or min
        min_correct = 0
        min_incorrect = 0
        mean_correct = 0
        mean_incorrect = 0
        for frame_key in tqdm(question_dict):
            (scene_id, frame_index) = frame_key.split('_')
            frame_index = int(frame_index)

            # We only want to loop through each scene once
            if frame_index != 0:
                continue

            if split_dict[scene_id] != 'test':
                continue

            # Create confused dict, which says which characters are consistently confused
            confused_dict = dict()
            for frame_index in range(8):
                frame_key = '%s_%d' % (scene_id, frame_index)
                answer_dict_vec = question_dict[frame_key]['answers']
                character_vec = question_dict[frame_key]['characters']
                for character_name in character_vec:
                    key = '%s_%s_%s' % (scene_id, frame_index, character_name)
                    answer_vec = []
                    if 'Doll' in character_name and key in concat_indices:
                        for answer_dict in answer_dict_vec:
                            answer = int(answer_dict[character_name] == answer_dict['omniscient'])
                            answer_vec.append(answer)
                        if answer_vec[0] == answer_vec[1]:
                            confused_dict[character_name] = [answer] + confused_dict.get(character_name, [])

            for (character_name_1, answers_1) in confused_dict.items():
                for (character_name_2, answers_2) in confused_dict.items():
                    correct_1 = min(answers_1)
                    correct_2  = min(answers_2)

                    if correct_1 == 0 and correct_2 == 1:
                        predict_1 = []
                        predict_2 = []
                        for frame_index in range(8):
                            key_1 = '%s_%s_%s' % (scene_id, frame_index, character_name_1)
                            key_2 = '%s_%s_%s' % (scene_id, frame_index, character_name_2)
                            if key_1 in concat_indices:
                                index_1 = concat_indices[key_1]
                                predict_1.append(y_prob[index_1])

                            if key_2 in concat_indices:
                                index_2 = concat_indices[key_2]
                                predict_2.append(y_prob[index_2])

                        assert len(predict_1) > 0 and len(predict_2) > 0
                        if np.min(predict_1) < np.min(predict_2):
                            min_correct += 1.0
                        elif np.min(predict_1) == np.min(predict_2):
                            min_correct += 0.5
                            min_incorrect += 0.5
                        else:
                            min_incorrect += 1.0

                        if np.mean(predict_1) < np.mean(predict_2):
                            mean_correct += 1.0
                        elif np.mean(predict_1) == np.mean(predict_2):
                            mean_correct += 0.5
                            mean_incorrect += 0.5
                        else:
                            mean_incorrect += 1.0

        print 'Who[scene]: Accuracy (min pool): %.3f%%' % (100.0 * float(min_correct) / (min_correct + min_incorrect))
        print 'Who[scene]: Accuracy (avg pool): %.3f%%' % (100.0 * float(mean_correct) / (mean_correct + mean_incorrect))


    if WHEN_TASK:
        # Given a frame, is any character mistaken in the frame?
        question_dict_filename = os.path.join(DATA_PATH, 'question_dict.json')

        split_dict_filename = os.path.join(DATA_PATH, 'split_dict_%02d.json' % args.RANDOM_SEED)
        y_prob = np.load('y_prob_%d.npy' % experiment_hash)

        with open(split_dict_filename) as f:
            split_dict = json.load(f)

        with open(question_dict_filename) as f:
            question_dict = json.load(f)

        with open('concat_indices_%d.json' % feature_hash, 'rb') as f:
            concat_indices = json.load(f)

        min_predict_vec = []
        avg_predict_vec = []
        correct_vec = []
        for frame_key in question_dict:
            (scene_id, frame_index) = frame_key.split('_')
            frame_index = int(frame_index)

            if split_dict[scene_id] != 'test':
                continue

            # Create confused dict, which says which characters are consistently confused
            confused_dict = dict()

            answer_dict_vec = question_dict[frame_key]['answers']
            for answer_dict in answer_dict_vec:
                for character_name in answer_dict:
                    key = '%s_%s_%s' % (scene_id, frame_index, character_name)
                    if 'Doll' in character_name and key in concat_indices:
                        answer = int(answer_dict[character_name] == answer_dict['omniscient'])
                        confused_dict[character_name] = [answer] + confused_dict.get(character_name, [])

            consistent_frame = True
            for (character_name, answers) in confused_dict.items():
                if len(set(answers)) > 1:
                    consistent_frame = False
                    break

            if not consistent_frame or len(confused_dict.keys()) == 0:
                continue

            # answer tells us whether every character is correct.
            # if the frame contains a consistently mistaken character (across the two annotations), answer = 0
            answer = min([ans for ans_vec in confused_dict.values() for ans in ans_vec])
            predict = []

            for character_name in confused_dict.keys():
                key = '%s_%s_%s' % (scene_id, frame_index, character_name)
                if key in concat_indices:
                    index = concat_indices[key]
                    predict.append(y_prob[index])

            min_predict = min(predict)
            avg_predict = np.mean(predict)


            min_predict_vec.append(min_predict)
            avg_predict_vec.append(avg_predict)
            correct_vec.append(answer)


        min_predict_vec = np.array(min_predict_vec)
        avg_predict_vec = np.array(avg_predict_vec)
        correct_vec = np.array(correct_vec)

        ((min_predict_vec, avg_predict_vec), correct_vec) = util.balance((min_predict_vec, avg_predict_vec), correct_vec)

        print 'When: Accuracy (min pool): %.3f%%' % (100.0 * util.accuracy_score(correct_vec, min_predict_vec > 0.5))
        print 'When: AP (min pool): %.3f' % average_precision_score(correct_vec, min_predict_vec)
        print 'When: Accuracy (avg pool): %.3f%%' % (100.0 * util.accuracy_score(correct_vec, avg_predict_vec > 0.5))
        print 'When: AP (avg pool): %.3f' % average_precision_score(correct_vec, avg_predict_vec)
        print 'When: Chance: %.3f%%' % (100.0 * np.mean(correct_vec))
        print 'When: Chance AP: %.3f' % average_precision_score(correct_vec, np.mean(correct_vec) * np.ones_like(correct_vec))


    ########################################################################
    # Cleanup

print 'Cleaning up'