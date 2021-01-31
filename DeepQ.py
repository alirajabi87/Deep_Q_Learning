import gym, os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
import random
from collections import deque
import progressbar

import tensorflow as tf
from tensorflow.keras.layers import Dense, Embedding, Reshape
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.optimizers import Adam

env = gym.make("Taxi-v3")

env.render()

print(f"Number of states: {env.observation_space.n}")
print(f"Number of actions: {env.action_space.n}")


class DQN:
    def __init__(self, env, optimizer):
        self.state_Size = env.observation_space.n
        self.action_size = env.action_space.n
        self.optimizer = optimizer

        self.experience_reply = deque(maxlen=2000)

        # Discount and Exploration rate
        self.GAMMA = 0.8
        self.epsilon = 1
        self.EPSILON_DECAY = 0.997

        # Build Network
        self.q_model = self.buildModel()
        self.target_model = self.buildModel()
        self.align_targetModel()

    def store(self, state, action, reward, next_state, done):
        self.experience_reply.append((state, action, reward, next_state, done))

    def buildModel(self):
        global epsilon
        if os.path.isfile('model.h5'):
            print("loading the model ...")
            model = tf.keras.models.load_model('model.h5')
            epsilon = 0.75
        else:
            epsilon = 1
            print("Creating the model ...")
            model = Sequential()
            model.add(Embedding(self.state_Size, 10, input_length=1))
            model.add(Reshape((10,)))
            model.add(Dense(50, activation='relu'))
            model.add(Dense(50, activation='relu'))
            model.add(Dense(self.action_size, activation='linear'))

            model.compile(optimizer=self.optimizer, loss="mse")
        return model

    def align_targetModel(self):
        self.target_model.set_weights(self.q_model.get_weights())

    def action(self, state, epsilon):
        if np.random.rand() <= epsilon:
            return env.action_space.sample()

        q_values = self.q_model.predict(state)
        return np.argmax(q_values[0])  # Possible Error

    def train(self, batchSize):
        miniBatch = random.sample(self.experience_reply, batchSize)

        for state, action, reward, next_state, done in miniBatch:
            target = self.target_model.predict(state)

            if done:
                target[0][action] = reward
            else:
                t = self.target_model.predict(next_state)
                target[0][action] = reward + self.GAMMA * np.amax(t)

            self.q_model.fit(state, target, epochs=1, verbose=0)


opt = Adam(learning_rate=0.01)
# optmizer = tf.keras.optimizers.SGD(learning_rate=0.01)

Agent = DQN(env, optimizer=opt)

batch_size = 32
num_episodes = 20 #100
timesteps_per_episode = 1000

Agent.q_model.summary()
epsilon = Agent.epsilon
EPSILON_DECAY = Agent.EPSILON_DECAY
MIN_EPSILON = 0.01
rewards = []
counter = 0
count = []
for episode in range(0, num_episodes):
    episode_reward = 0
    done = False

    state = env.reset()
    state = np.reshape(state, [1, 1])

    bar = progressbar.ProgressBar(maxval=int(timesteps_per_episode / 10),
                                  widgets=[progressbar.Bar('=', '[', ']'), '', progressbar.Percentage()])
    bar.start()

    for timestep in range(timesteps_per_episode):

        action = Agent.action(state, epsilon)

        next_state, reward, done, info = env.step(action)

        next_state = np.reshape(next_state, [1, 1])

        Agent.store(state, action, reward, next_state, done)

        state = next_state
        episode_reward += reward

        if done:
            Agent.align_targetModel()
            counter += 1
            if epsilon > MIN_EPSILON:
                epsilon *= EPSILON_DECAY
                epsilon = max(MIN_EPSILON, epsilon)
            break

        if len(Agent.experience_reply) > batch_size:
            Agent.train(batch_size)

        if not timestep % 10:
            bar.update(int(timestep / 10)+1)

    rewards.append(episode_reward)
    count.append(counter)
    bar.finish()

    if not (episode + 1) % 5:
        print("*************************")
        print(f"Episode: {episode + 1}, Epsilon: {epsilon}")
        env.render()
        print("*************************")

Agent.target_model.save('model.h5')

import matplotlib.pyplot as plt

plt.plot(range(num_episodes), rewards)
plt.show()

plt.plot(range(num_episodes), count)
plt.show()
