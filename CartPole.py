
import gym, os, random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from collections import deque
import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Reshape, Dropout, Input
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.activations import relu, linear, elu
from tensorflow.keras.models import Model, Sequential
import progressbar
from random import shuffle

# Global Variables

GAMMA = 0.8
ALPHA = 1
epsilon = 1
EPSILON_DECAY = 0.99
LEARNING_RATE = 0.2
MIN_EPSILON = 0.01
REPLAY_MEMORY_SIZE = 10000

class Agent:
    def __init__(self, optimizer, env):
        self.env = env
        self.optimizer = optimizer
        self.q_model = self.model()
        self.target_model = self.model()
        self.align_weights()
        self.experience_reply = deque(maxlen=REPLAY_MEMORY_SIZE)

    def action(self, state):
        if np.random.rand() <= epsilon:
            return self.env.action_space.sample()
        else:
            return np.argmax(self.q_model.predict(state)[0])

    def model(self):
        global epsilon
        if os.path.isfile('CartPole.h5'):
            print("loading the model...")
            model = tf.keras.models.load_model('CartPole.h5')
        else:
            print("Creating the model....")
            model = Sequential()
            model.add(Dense(24, input_shape=(self.env.observation_space.shape[0],), activation='relu'))
            model.add(Dense(24, activation='relu'))
            model.add(Dense(self.env.action_space.n, activation='linear'))
            model.compile(loss="mse", optimizer=self.optimizer, metrics=[])

        return model

    def align_weights(self):
        self.target_model.set_weights(self.q_model.get_weights())

    def store(self, state, action, reward, next_state, done):
        if len(self.experience_reply) > REPLAY_MEMORY_SIZE:
            if np.random.rand() < 0.5:
                shuffle(self.experience_reply)
                self.experience_reply.popleft()
        self.experience_reply.append((state, action, reward, next_state, done))

    def train(self, batchsize):
        minibatch = random.sample(self.experience_reply, batchsize)

        for state, action, reward, next_state, done in minibatch:
            q_values = self.target_model.predict(state)
            if not done:
                q_values[0][action] = ALPHA * (reward + GAMMA * (np.max(self.target_model.predict(next_state))))  # [0]
            else:
                q_values[0][action] = reward
            self.q_model.fit(state, q_values, epochs=1, verbose=0)

def main():
    global epsilon
    env = gym.make('CartPole-v1')
    env.reset()
    count = 0

    print(f"States: {env.observation_space.shape[0]}")
    print(f"Actions: {env.action_space.n}")

    opt = Adam(learning_rate=0.001)
    # opt = SGD(learning_rate=0.01)
    agent = Agent(optimizer=opt, env=env)

    timestep_per_episode = 1000
    num_episodes = 700
    batch_size = 64

    agent.q_model.summary()

    for episode in range(num_episodes):
        episode_reward = 0
        done = False
        state = env.reset()
        state = np.reshape(state, [-1, env.observation_space.shape[0]])
        bar = progressbar.ProgressBar(maxval=int(timestep_per_episode / 10),
                                      widgets=[progressbar.Bar('=', '[', ']'), '', progressbar.Percentage()])

        for timestep in range(timestep_per_episode):
            action = agent.action(state)
            next_state, reward, done, info = env.step(action)
            # reward = reward if done else -reward
            next_state = np.reshape(next_state, [-1, env.observation_space.shape[0]])  # [1, observation_space]

            if done:
                agent.align_weights()
                count += 1
                
                if timestep != timestep_per_episode:
                    reward = -5
                if timestep == timestep_per_episode:
                    reward = 5

                if epsilon >= MIN_EPSILON:
                    epsilon *= EPSILON_DECAY
                    epsilon = max(epsilon, MIN_EPSILON)

                agent.store(state, action, reward, next_state, done)
                break
            agent.store(state, action, reward, next_state, done)

            state = next_state
            if len(agent.experience_reply) > batch_size:
                agent.train(batch_size)

            if not timestep % 10:
                bar.update(timestep / 10 + 1)



        bar.finish()

        if not (episode + 1) % 5:
            print("**************************")
            print(f"Episode: {episode + 1}, epsilon: {epsilon}")
            env.render()
            print("**************************")
    agent.target_model.save("CartPole.h5")
    print(count)


if __name__ == '__main__':
    main()
