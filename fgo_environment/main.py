# extra imports to set GPU options
import tensorflow as tf
from keras import backend as k
 
###################################
# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.3
 
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))
k.tensorflow_backend._get_available_gpus()
from colosseum import Battle
from netlearner import DQNLearner

import time


def main():
    num_learning_rounds = 30000
    game = Battle(num_learning_rounds = num_learning_rounds) #Deep Q Network Learner
    #game = Game(num_learning_rounds, Learner()) #Q learner
    number_of_test_rounds = 1000
    for k in range(0,num_learning_rounds + number_of_test_rounds):
        game.fight_battle()


    #df = game.p.get_optimal_strategy()
    #print(df)
    #df.to_csv('optimal_policy.csv')

if __name__ == "__main__":
    start_time = time.time()
    main()
    print("--- %s seconds ---" % (time.time() - start_time))