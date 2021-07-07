"""DeepQLearning Agent that plays Atari game Breakout.

Usage:
  agent train [--mode=<mode>]
  agent test
  agent -h | --help
  agent --version

Options:
  -h --help         Show this screen.
  --version         Show version.
  --mode=<mode>

"""

import numpy as np
import tensorflow.compat.v1 as tf
import gym
import skimage
import random
import time
import os
from docopt import docopt
from skimage import color, transform, exposure
from keras import layers
from keras.models import Model
from keras.optimizers import RMSprop
from keras import backend as K
from keras.callbacks import TensorBoard
from keras.models import load_model
from keras.models import clone_model
from collections import deque
from datetime import datetime

INPUT_SHAPE = (84,84,4)
ACTIONS = 3
LEARNING_RATE = 0.00025
STEP = 50000
BATCH_SIZE = 32
GAMMA = 0.99
REPLAY_MEMORY = 350000
INIT_EPSILON = 1
FINAL_EPSILON = 0.1
EPSILON_STEP_NUM = 1000000
TRAIN_DIR = "model"
NUM_EPISODES = 100000
NUM_TEST_EPISODES = 1
NO_ACTION_STEPS = 30
REFRESH_TARGET_MODEL = 10000
SAVE_AFTER_N_EPISODES = 50

def preprocess(frame):
    frame = skimage.color.rgb2gray(frame)
    frame = skimage.transform.resize(frame,(84,84))
    frame = skimage.exposure.rescale_intensity(frame,out_range=(0,255))
    frame = frame.astype(np.uint8)
    return frame

def huber_loss(a, b):
    error = a - b
    quadratic_term = error*error / 2
    linear_term = abs(error) - 1/2
    use_linear_term = (abs(error) > 1.0)
    # Keras won't let us multiply floats by booleans, so we explicitly cast the booleans to floats
    use_linear_term = K.cast(use_linear_term, 'float32')
    return use_linear_term * linear_term + (1-use_linear_term) * quadratic_term

def buildmodel():

    frames_input = layers.Input(INPUT_SHAPE, name = 'frames')

    actions_input = layers.Input((ACTIONS,), name = 'mask')

    normalized = layers.Lambda(lambda x: x / 255.0)(frames_input)

    conv1 = layers.Conv2D(16, kernel_size = (8,8), strides = (4,4), activation = 'relu')(normalized)
    conv2 = layers.Conv2D(32, kernel_size = (4,4), strides = (2,2), activation = 'relu')(conv1)
    #conv3 = layers.Conv2D(64, kernel_size = (3,3), strides = (1,1), activation = 'relu')(conv2)
    conv_flattened = layers.Flatten()(conv2)
    dense1 = layers.Dense(256, activation = 'relu')(conv_flattened)
    dense2 = layers.Dense(ACTIONS)(dense1)

    filtered_output = layers.Multiply()([dense2, actions_input])

    model = Model(inputs = [frames_input, actions_input], outputs = filtered_output)
    optimizer = RMSprop(learning_rate = LEARNING_RATE, rho = 0.95, epsilon = 0.01)
    # compare with mse
    model.compile(optimizer, loss = 'mse')

    return model

def get_action(epsilon, step, model, state):
    if random.random() <= epsilon or step <= STEP:
       return random.randrange(ACTIONS)
    else:
        q_value = model.predict([state, np.ones(ACTIONS).reshape(1, ACTIONS)])
        return np.argmax(q_value[0])

def store_memory(memory, state, action, reward, next_state, finished):
    memory.append((state, action, reward, next_state, finished))

def get_one_hot(actions):
    return np.eye(ACTIONS)[np.array(actions).reshape(-1)]

def train_memory_batch(memory, model_target, model):
    minibatch = random.sample(memory, BATCH_SIZE)
    state = np.zeros((BATCH_SIZE, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
    next_state = np.zeros((BATCH_SIZE, INPUT_SHAPE[0], INPUT_SHAPE[1], INPUT_SHAPE[2]))
    target = np.zeros((BATCH_SIZE,))
    action, reward, finished = [], [], []

    for idx, val in enumerate(minibatch):
        state[idx] = val[0]
        next_state[idx] = val[3]
        action.append(val[1])
        reward.append(val[2])
        finished.append(val[4])

    actions_mask = np.ones((BATCH_SIZE, ACTIONS))
    next_Q_values = model_target.predict([next_state, actions_mask])

    for i in range(BATCH_SIZE):
        if finished[i]:
            target[i] = -1
        else:
            target[i] = reward[i] + GAMMA * np.max(next_Q_values[i])  

    action_one_hot = get_one_hot(action)
    target_one_hot = action_one_hot * target[:, None]

    #tb_callback = TensorBoard(log_dir = log_dir, histogram_freq = 0, write_graph = True, write_images = False)

    h = model.fit([state, action_one_hot], target_one_hot, epochs=1, batch_size=BATCH_SIZE, verbose = 0)
    
    return h.history['loss'][0]

def train(args):

    env = gym.make('BreakoutDeterministic-v4')

    memory = deque(maxlen = REPLAY_MEMORY)
    episode_number = 0
    epsilon = INIT_EPSILON
    epsilon_decay = (INIT_EPSILON - FINAL_EPSILON)/EPSILON_STEP_NUM
    global_step = 0

    if args['--mode'] is not None and args['--mode'].lower() == 'resume':
        if 'google.colab' in str(get_ipython()):
        # symlinks don't work on google colab
            file = open(f"{TRAIN_DIR}/current/model.h5", "r")
            content = file.read()
            file.close()
            model = load_model(f"{TRAIN_DIR}/current/{content}")
            epsilon = FINAL_EPSILON
        else:
            model = load_model(f"{TRAIN_DIR}/current/model.h5")
            epsilon = FINAL_EPSILON
    else:
        model = buildmodel()

    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")

    log_dir = f"{TRAIN_DIR}/{now}"
    
    with tf.Graph().as_default():
        file_writer = tf.summary.FileWriter(log_dir,tf.get_default_graph())

    model_target = clone_model(model)
    model_target.set_weights(model.get_weights())

    while episode_number < NUM_EPISODES:

        is_done = False
        finished = False
        step, score, start_life = 0, 0, 5
        loss = 0
        frame = env.reset()

        for _ in range(random.randint(1, NO_ACTION_STEPS)):
            frame, _, _, _ = env.step(1)

        state = preprocess(frame)
        history = np.stack((state, state, state, state), axis = 2)
        history = np.reshape([history], (1,84,84,4))

        while not is_done:
            # if args['--mode'] is not None and args['--mode'].lower() == 'render':
            #     env.render()
            #     time.sleep(0.01)

            action = get_action(epsilon, global_step, model, history)
            #real_action = action + 1

            if epsilon > FINAL_EPSILON and global_step > STEP:
                epsilon -= epsilon_decay

            frame, reward, is_done, info = env.step(action)

            next_state = preprocess(frame)
            next_state = np.reshape([next_state], (1,84,84,1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            if start_life > info['ale.lives']:
                finished = True
                start_life = info['ale.lives']

            store_memory(memory, history, action, reward, next_history, finished)

            if global_step > STEP:
                loss = loss + train_memory_batch(memory, model_target, model)

                if global_step % REFRESH_TARGET_MODEL == 0:
                    model_target.set_weights(model.get_weights())

            score += reward

            history = next_history

            global_step += 1
            step += 1

            if is_done:
                if global_step <= STEP:
                    state = "observe"
                elif STEP < global_step <= STEP + EPSILON_STEP_NUM:
                    state = "explore"
                else:
                    state = "train"
                print(f'state: {state}, episode: {episode_number}, score: {score}, global_step: {global_step}, avg loss: {loss/float(step)}, step: {step}, memory length: {len(memory)}')

                if episode_number % SAVE_AFTER_N_EPISODES == 0 or (episode_number+1) == NUM_EPISODES:
                    file_name = "model.h5"
                    model_path = os.path.join(log_dir, file_name)
                    model.save(model_path)
                    current_model_path = f"{TRAIN_DIR}/current/model.h5"
                    os.remove(current_model_path)
                    os.symlink(f"../{now}/model.h5", current_model_path)

                loss_summary = tf.Summary(value=[tf.Summary.Value(tag="loss", simple_value=loss/float(step))])
                file_writer.add_summary(loss_summary, global_step=episode_number)

                score_summary = tf.Summary(value=[tf.Summary.Value(tag="score", simple_value=score)])
                file_writer.add_summary(score_summary, global_step=episode_number)

                episode_number+=1

    file_writer.close()

def test():
    env = gym.make('BreakoutDeterministic-v4')

    episode_number = 0
    epsilon = 0.001
    global_step = STEP + 1
    model = load_model(f"{TRAIN_DIR}/current/model.h5")

    while episode_number < NUM_EPISODES:

        is_done = False
        finished = False
        score, start_life = 0, 5
        frame = env.reset()

        frame, _, _, _ = env.step(1)
        state = preprocess(frame)
        history = np.stack((state, state, state, state), axis = 2)
        history = np.reshape([history], (1,84,84,4))

        while not is_done:
            env.render()
            time.sleep(0.01)

            action = get_action(epsilon, global_step, model, history)
            real_action = action + 1
            #q_value = model.predict([history, np.ones(ACTIONS).reshape(1, ACTIONS)])
            #action = np.argmax(q_value[0]) 

            frame, reward, is_done, info = env.step(real_action)

            #env.step(1)

            next_state = preprocess(frame)
            next_state = np.reshape([next_state], (1,84,84,1))
            next_history = np.append(next_state, history[:, :, :, :3], axis=3)

            if start_life > info['ale.lives']:
                finished = True
                start_life = info['ale.lives']

            reward = np.clip(reward, -1, 1)

            score += reward

            if finished:
                finished = False
                #history = next_history
            else:
                history = next_history

            global_step += 1

            if is_done:
                episode_number += 1
                print(f'episode: {episode_number}, score: {score}')

def main():
    args = docopt(__doc__, version='DQN Agent 0.9.0')
    
    if args['test']:
        test()
    
    if args['train']:
        train(args)
    

if __name__ == "__main__":
    main()
