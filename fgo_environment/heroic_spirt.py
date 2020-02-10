# extra imports to set GPU options
import tensorflow as tf
from keras import backend as k
 
###################################
# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.2
 
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))


import random
import copy
import keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop,Adam,SGD
from keras.layers import LeakyReLU
from keras.layers import Dropout

import numpy as np
import pandas as pd
#from colosseum import team_chaldela
import itertools
from random import choice
from keras.models import load_model
from keras.layers import Dense, Dropout, Embedding, LSTM, GlobalMaxPooling1D, SpatialDropout1D

from fgo_environment.utils import use_predicted_probability, convert_card_list
from fgo_environment.netlearner import DQNLearner
<<<<<<< Updated upstream

def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo/expo_sum
=======
from fgo_environment.utils import use_predicted_probability, convert_card_list

FEATURE_LEN_FOR_NETWORKS = 2+9#1+9#
>>>>>>> Stashed changes
'''
CHALDEA.hero1._model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/run6_good_run_8K/jalter_iteration_8000.h5')
CHALDEA.hero2._model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/run6_good_run_8K/Ishtar_iteration_8000.h5')
CHALDEA.hero3._model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/run6_good_run_8K/artoria_pendragon_iteration_8000.h5')


'''
class Chaldea():
    def __init__(self,
                 hero_list = None
                 ):
        super().__init__()
        self._learning = True
        self._epsilon = 1.0
        if hero_list is None:
            self.hero1 = JAlter(health=10,NP=50,spot='hero1',_epsilon=self._epsilon)
            self.hero2 = Ishtar(health=10,NP=50,spot='hero2',_epsilon=self._epsilon)
            self.hero3 = ArtoriaSaber(health=10,NP=50,spot='hero3',_epsilon=self._epsilon)
        else:
            self.hero1 = hero_list[0]
            self.hero2 = hero_list[1]
            self.hero3 = hero_list[2]

        self.hero_dict = {'hero1': self.hero1,
                          'hero2': self.hero2,
                          'hero3': self.hero3}

        self.deck = self.hero1.deck + self.hero2.deck + self.hero3.deck
        self.hit_points = self.hero1.health + self.hero2.health + self.hero3.health

        self.current_stars = 0
        self.critical_strength = 0

        self.card_picker = DQNLearner()


def hero_network(feature_vec_len=FEATURE_LEN_FOR_NETWORKS,action_dict=4,initial_lr=0.01):
        model = Sequential()
        model.add(Dense(128, init='glorot_normal', input_dim=feature_vec_len))#activation = 'relu', input_dim=self.feature_vector_len))
        #model.add(Dropout(0.4))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(64, init='glorot_normal', activation = 'relu'))
        #model.add(Dropout(0.4))
        model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(action_dict, init='glorot_normal',activation='softmax'))
        opt = SGD(lr=initial_lr, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        '''
        
        #model.add(Embedding(feature_vec_len, 10, input_length=feature_vec_len))
        #model.add(LSTM(10, dropout = 0.3, recurrent_dropout = 0.3))
        #model.add(Dense(5, activation = 'relu'))
        #model.add(Dropout(0.3))
        model.add(Dense(10, init='glorot_normal', input_dim=feature_vec_len))#activation = 'relu', input_dim=self.feature_vector_len))
        #model.add(Dropout(0.4))
        #model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(8, init='glorot_normal', activation = 'relu'))
        #model.add(Dropout(0.4))
        #model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(action_dict, init='glorot_normal',activation='softmax'))
        opt = SGD(lr=initial_lr, momentum=0.9)
        model.compile(loss='categorical_crossentropy', optimizer=opt)
        '''
        return model

class HeroicSpirit():
    def __init__(self,
                 name,
                 health=1000,
                 NP=0,
                 NP_damage=20,
                 spot='hero1',
                 deck=None,
<<<<<<< Updated upstream
                 _epsilon=.3,
=======
                 _epsilon=.05,#1.0,
>>>>>>> Stashed changes
                 _discount=.1
                 ):
        super().__init__()
        self.name = name
        self.health = health
        self.NP_charge = NP
        self.NP_damage = NP_damage
        self.spot = spot
        self._epsilon = _epsilon
        self._discount = .1
        self.feature_vector_len = FEATURE_LEN_FOR_NETWORKS
        self._learning = True

        if deck is None:
            deck = [(self.spot,'quick'),(self.spot,'arts'),(self.spot,'arts'),(self.spot,'buster'),(self.spot,'buster')]
        else:
            _deck_holder = []
            for card in deck:
                _deck_holder.append((self.spot,card))
            deck = _deck_holder

        self.deck = deck
        self.damage_mod_dict = {'buster': 1,
                                'arts': 1,
                                'quick': 1}
        self.active_buffs_dict = {}

        self.action_list = [('pass','pass'),('pass','pass'),('pass','pass'),
                            ('sk1',self.skill_1()), ('sk2',self.skill_2()),
                            ('sk3',self.skill_3())]

        self.NP_gain_buff = 0

        self.used_skill_1 = 0
        self.used_skill_2 = 0
        self.used_skill_3 = 0

        self.action_dict = {0: ('pass', 'pass'),
                            1: ('sk1', self.skill_1()),
                            2: ('sk2', self.skill_2()),
                            3: ('sk3', self.skill_3())}
        self.initial_lr = 0.00001
        self.class_weight = {0: 1.0,
<<<<<<< Updated upstream
                1: 5.0,
                2: 5.0,
                3: 5.0}
        model = Sequential()

        model.add(Dense(64, init='glorot_normal', input_dim=self.feature_vector_len,kernel_regularizer=keras.regularizers.l2(l=0.02)))#activation = 'relu', input_dim=self.feature_vector_len))
        model.add(Dropout(0.4))
        model.add(LeakyReLU(alpha=0.03))
        model.add(Dense(32, init='glorot_normal',kernel_regularizer=keras.regularizers.l2(l=0.02)))#$, activation = 'relu'))
        model.add(Dropout(0.4))
        model.add(LeakyReLU(alpha=0.03))
        #model.add(Dense(64, init='glorot_normal',kernel_regularizer=keras.regularizers.l2(l=0.02)))#, activation = 'relu'))
        #model.add(Dropout(0.5))
        #model.add(LeakyReLU(alpha=0.03))
        #model.add(Dense(64, init='glorot_normal',kernel_regularizer=keras.regularizers.l2(l=0.2)))#, activation = 'relu'))
        #model.add(LeakyReLU(alpha=0.01))

        model.add(Dense(len(self.action_dict), init='glorot_normal',activation='linear'))
        opt = SGD(lr=self.initial_lr, momentum=0.9, clipnorm=2.0)
        #opt = Adam(learning_rate=self.initial_lr)
        model.compile(loss='binary_crossentropy', optimizer=opt)
=======
                1: 8.0,
                2: 8.0,
                3: 8.0}

    
>>>>>>> Stashed changes

        #self._model = hero_network(self.feature_vec_len,self.action_dict,self.initial_lr)

    def skill_1(self):
        #print('skill_1')
        output_dict = {'name':'jeanne_sk1',
                       'target': [self.spot],
                       'hp_boost': 1,
                       'np_boost': .01,
                       'critical_boost': .01,
                       'critical_star_boost': .00,
                       'dmg_boost':('all', .50),
                       'duration':3}

        return output_dict

    def skill_2(self):
        #print('skill_2')
        output_dict = {'name':'jeanne_sk2',
                       'target': ['hero1','hero2','hero3'],
                       'hp_boost': 1,
                       'np_boost': .01,
                       'critical_boost': .01,
                       'critical_star_boost': .01,
                       'dmg_boost':('all', .10),
                       'duration':3}
        return output_dict

    def skill_3(self):
        #print('skill_3')
        output_dict = {'name':'jeanne_sk3',
                       'target': [self.spot],
                       'hp_boost': 1,
                       'np_boost': .01,
                       'critical_boost': .01,
                       'critical_star_boost': .01,
                       'dmg_boost':('all', .30),
                       'duration':1}
        return output_dict

    def invalid_steps(self, preds):
        sk1_used = self.used_skill_1
        sk2_used = self.used_skill_2
        sk3_used = self.used_skill_3
<<<<<<< Updated upstream

        new_pred_min = np.min(preds[0])-.5 #with softmax everything is just probabilities
=======
        super_sk3_used = 0
        new_pred_min = 0 #np.min(preds[0])-.5 #with softmax everything is just probabilities
        if len(self.action_dict) >4:
            if sk3_used ==1:
                super_sk3_used = 1# and _index>=3: #need to handle the case where self is not buffed?
                #preds[0][_index] = new_pred_min
>>>>>>> Stashed changes
        for _index in range(len(self.action_dict)):
            if 'sk1' in self.action_dict[_index] and sk1_used == 1:
                preds[0][_index] = new_pred_min
            if 'sk2' in self.action_dict[_index] and sk2_used == 1:
                preds[0][_index] = new_pred_min
            if 'sk3' in self.action_dict[_index] and sk3_used == 1:
                preds[0][_index] = new_pred_min

            if super_sk3_used == 1 and _index>=3: #need to handle the case where self is not buffed?
                preds[0][_index] = new_pred_min

    def get_random_action(self,random_pass = 2):
        action_dict_copy = copy.deepcopy(self.action_dict )
        if self.used_skill_1 == 1:
            for action_key in list(action_dict_copy.keys()):
                if 'sk1' in  action_dict_copy[action_key][0]:
                    del action_dict_copy[action_key]
        if self.used_skill_2 == 1:
            for action_key in list(action_dict_copy.keys()):
                if 'sk1' in  action_dict_copy[action_key][0]:
                        del action_dict_copy[action_key]
            del action_dict_copy[2]
        if self.used_skill_3 == 1:
<<<<<<< Updated upstream
            del action_dict_copy[3]
=======
            for action_key in list(action_dict_copy.keys()):
                if 'sk3' in  action_dict_copy[action_key][0]:
                        del action_dict_copy[action_key]
                elif 'sk3_a' in  action_dict_copy[action_key][0]:
                        del action_dict_copy[action_key]
                elif 'sk3_b' in  action_dict_copy[action_key][0]:
                        del action_dict_copy[action_key]
                elif 'sk3_c' in  action_dict_copy[action_key][0]:
                        del action_dict_copy[action_key]
        
        #if random_pass==0:
        #    if len(action_dict_copy) >1:
        #        del action_dict_copy[0]


>>>>>>> Stashed changes

        random_pass_list = []
        if random_pass >0:
            for i in range(int(random_pass*len(self.action_dict))):
                random_pass_list.append((0, ('pass', 'pass')))

        return random.choice(list(action_dict_copy.items())+random_pass_list)

    def get_action(self, state):
        # Take in game state and get a prediction
        game_state_array = np.reshape(np.asarray(state), (1, self.feature_vector_len))

        preds = self._model.predict(game_state_array,batch_size=1)
        #print(preds)
        self.invalid_steps(preds)
        #print(preds)
        predicted_class = np.argmax(preds)

        # Exploration vs Exploitation!
        # if the randomly generated value is less than the epsilon value
        # we go down the Exploitation route... (I might have defined this backwards from normal...)
        if np.random.uniform(0, 1) < self._epsilon:
            action = use_predicted_probability(self.action_dict, predicted_class)
        # When the random number is greater than the epsilon value we randomly select an action
        # and we see how it goes!
        else:
<<<<<<< Updated upstream
            if state[0] ==1/3:
                random_pass_number = 1
            if state[0] ==2/3:
                random_pass_number = .5
            if state[0] ==3/3:
=======
            
            if state[0] ==1:
                random_pass_number = 2
            if state[0] ==2:
                random_pass_number = 1
            if state[0] ==3:
>>>>>>> Stashed changes
                random_pass_number = 0
            random_action_key_value = self.get_random_action(random_pass=random_pass_number)
            predicted_class = random_action_key_value[0]
            action = random_action_key_value[1]
<<<<<<< Updated upstream
=======
            '''
            if state[0] ==1/3:
                #print('passing_ round1')
                predicted_class = 0
                action = ('pass', 'pass')
            '''
>>>>>>> Stashed changes
            #action = use_predicted_probability(self.action_dict,predicted_classs)

        # store some stuff for later
        self._last_state = game_state_array
        self._last_action = predicted_class
        self._last_target = preds

        return action,predicted_class, preds 

    def update(self, state, preds,predicted_class, reward):
        '''
        reward:
                reward genearted from the game envionment
            state:
                game state
            new:
                discounted model outputs. This gets combined with with the game environment rewards
        '''
        if self._learning:
            # In this version I call predict again... could just use self._last_target... *shrug
            game_state_array = np.reshape(np.asarray(state), (1, self.feature_vector_len))
            #preds = self._model.predict([outfit_state_array], batch_size=1)
            #self.invalid_steps(preds)
            maxQ = np.amax(preds)
            #new = self._discount * maxQ #discount is applied bc it is a future action??? idk.. is standard?

<<<<<<< Updated upstream
            combined_reward = reward + maxQ

            preds[0][predicted_class] = combined_reward
            softmaxed_adjusted_preds = softmax(preds)
            #print(preds,combined_reward,softmaxed_adjusted_preds)

            # at every update we are doing are training on a single batch of size 1
            self._model.fit(game_state_array, softmaxed_adjusted_preds, batch_size=1, epochs=1, verbose=0,class_weight=self.class_weight)
    
=======
            #combined_reward = reward + maxQ
            #normalized = (combined_reward-min(combined_reward))/(max(combined_reward)-min(combined_reward))
            #if reward >=0:
            #    group_reward_mod = -reward
            #else:
            #    group_reward_mod = abs(reward)

            #preds[0][predicted_class] = combined_reward

            #nonzero_idxs = np.where(preds[0] > 1)[0]
            #preds[0][nonzero_idxs] = 1

            #nonzero_idxs = np.where(preds_list[i][0] < 0)[0]
            #preds[0][nonzero_idxs] = 0
            '''
            print(preds)
            for i in range(len(preds[0])):
                normalized_i = (preds[0][i]-np.amin(preds))/(np.amax(preds)-np.amin(preds))
                preds[0][i] = normalized_i
            print('normalized?:', preds)
            '''
            #softmaxed_adjusted_preds = softmax(preds)
            #print(preds,combined_reward,softmaxed_adjusted_preds)

            # at every update we are doing are training on a single batch of size 1
            self._model.fit(game_state_array, preds_target, batch_size=512, epochs=50, verbose=0)#,class_weight=self.class_weight)
            '''
            index = 0 
            for i in game_state_array:
                print(i)
                print(preds_target[index])
                index+=1
                i = np.reshape(np.asarray(i), (1, self.feature_vector_len))
                print(self._model.predict(i,batch_size=1))
                print('')
            '''
>>>>>>> Stashed changes
    def save_rl_model(self,model_name):

        self._model.save(str(model_name)+'.h5')

class JAlter(HeroicSpirit):

    def __init__(self,
                 name='jalter',
                 health=10,
                 NP=0,
                 NP_damage=40,
                 spot='hero1',
                 deck=None,
                 _epsilon=.3):
        super().__init__(name, health, NP, NP_damage, spot)
<<<<<<< Updated upstream

        self._model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/run6_good_run_8K/jalter_iteration_9000.h5')

=======
        self._epsilon = _epsilon
        self._model = hero_network()
        #self._model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/run_pure_policy/jalter_iteration_200000.h5')
        #self._model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/it works/jalter_iteration_5000.h5')
>>>>>>> Stashed changes
    def skill_1(self): #should be a crit boost but unsure how to do that atm??
        #print('skill_1')
        output_dict = {'name': 'sk1_'+self.spot+'_'+self.name+'_self_modification_ex',
                       'target': [self.spot],
                       'hp_boost': 0,
                       'np_boost': .00,
                       'critical_boost': .50,
                       'critical_star_boost': .0,
                       'dmg_boost': ('all', .00),
                       'duration': 3}

        return output_dict

    def skill_2(self): 
        #print('skill_2')
        output_dict = {'name': 'sk2_'+self.spot+'_'+self.name+'_dragon_witch',
                       'target': ['hero1','hero2','hero3'],
                       'hp_boost': 0,
                       'np_boost': .00,
                       'critical_boost': .00,
                       'critical_star_boost': .0,
                       'dmg_boost': ('all', .20),
                       'duration': 3}
        return output_dict

    def skill_3(self):
        #print('skill_3')
        output_dict = {'name': 'sk3_'+self.spot+'_'+self.name+'_ephermeral_dream',
                       'target': [self.spot],
                       'hp_boost': 0,
                       'np_boost': .0,
                       'critical_boost': .0,
                       'critical_star_boost': .0,
                       'dmg_boost': ('buster', .50),
                       'duration': 1}
        return output_dict


class Ishtar(HeroicSpirit):

    def __init__(self,
                 name='Ishtar',
                 health=10,
                 NP=0,
                 NP_damage=40,
                 spot='hero1',
                 deck=None,
                 _epsilon=.3):
<<<<<<< Updated upstream
        super().__init__(name, health, NP, NP_damage,spot)

        self.model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/run6_good_run_8K/jalter_iteration_9000.h5')
=======
        super().__init__(name, health, NP, NP_damage, spot)
        self._epsilon = _epsilon
        self._model = hero_network()
        #self._model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/run_pure_policy/Ishtar_iteration_190000.h5')
        #self._model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/it works/Ishtar_iteration_5000.h5')
>>>>>>> Stashed changes

    def skill_1(self): #should be a crit boost but unsure how to do that atm??
        #print('skill_1')
        output_dict = {'name': 'sk1_'+self.spot+'_'+self.name+'_manifestation_of_beauty',
                       'target': ['hero1','hero2','hero3'],
                       'hp_boost': 0,
                       'np_boost': .00,
                       'critical_boost': .00,
                       'critical_star_boost': .0,
                       'dmg_boost': ('all', .2),
                       'duration': 3}

        return output_dict

    def skill_2(self):
        #print('skill_2')
        output_dict = {'name': 'sk2_'+self.spot+'_'+self.name+'_gleaming_brilliant_crown',
                       'target': [self.spot],
                       'hp_boost': 0,
                       'np_boost': 50,
                       'critical_boost': .00,
                       'critical_star_boost': .0,
                       'dmg_boost': ('all', .0),
                       'duration': 1}
        return output_dict

    def skill_3(self):
        #print('skill_3')
        output_dict = {'name': 'sk3_'+self.spot+'_'+self.name+'_mana_gem_burst',
                       'target': [self.spot],
                       'hp_boost': 0,
                       'np_boost': .00,
                       'critical_boost': .00,
                       'critical_star_boost': .0,
                       'dmg_boost': ('all', .50),
                       'duration': 2}
        return output_dict


class ArtoriaSaber(HeroicSpirit):

    def __init__(self,
                 name='artoria_pendragon',
                 health=10,
                 NP=0,
                 NP_damage=40,
                 spot='hero1',
                 deck=None,
                 _epsilon=.3):
<<<<<<< Updated upstream
        super().__init__(name, health, NP, NP_damage,spot)
        self.model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/run6_good_run_8K/artoria_pendragon_iteration_9000.h5')
=======
        super().__init__(name, health, NP, NP_damage, spot)
        self._epsilon = _epsilon
        self._model = hero_network()
        #self._model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/run_pure_policy/artoria_pendragon_iteration_190000.h5')
    
        #self._model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/it works/artoria_pendragon_iteration_5000.h5')
>>>>>>> Stashed changes
    def skill_1(self): #should be a crit boost but unsure how to do that atm??
        #print('skill_1')
        output_dict = {'name': 'sk1_'+self.spot+'_'+self.name+'_charisma_B',
                       'target': ['hero1','hero2','hero3'],
                       'hp_boost': 0,
                       'np_boost': .00,
                       'critical_boost': .00,
                       'critical_star_boost': 0,
                       'dmg_boost': ('all', .18),
                       'duration': 3}

        return output_dict

    def skill_2(self): 
        #print('skill_2')
        output_dict = {'name': 'sk2_'+self.spot+'_'+self.name+'_mana_burst_b',
                       'target': [self.spot],
                       'hp_boost': 0,
                       'np_boost': .00,
                       'critical_boost': .00,
                       'critical_star_boost': .0,
                       'dmg_boost': ('buster', .50),
                       'duration': 1}
        return output_dict

    def skill_3(self): #adds crit stars
        #print('skill_3')
        output_dict = {'name': 'sk3_'+self.spot+'_'+self.name+'_instinct_a',
                       'target': [self.spot],
                       'hp_boost': 0,
                       'np_boost': .00,
                       'critical_boost': .00,
                       'critical_star_boost': 10,
                       'dmg_boost': ('all', 0),
                       'duration': 1}
        return output_dict


class Merlin(HeroicSpirit):

    def __init__(self,
                 name='Merlin',
                 health=10,
                 NP=0,
                 NP_damage=0,
                 spot='hero1',
                 deck=None,
                 _epsilon=.3):
        super().__init__(name, health, NP, NP_damage, spot)
        self._epsilon = _epsilon

        self._model = hero_network(action_dict=6)
        self.damage_mod_dict = {'buster': .8,
                                'arts': .8,
                                'quick': .8}
        self.action_dict = {0: ('pass', 'pass'),
                            1: ('sk1', self.skill_1()),
                            2: ('sk2', self.skill_2()),
                            3: ('sk3_a', self.skill_3_A()),
                            4: ('sk3_b', self.skill_3_B()),
                            5: ('sk3_c', self.skill_3_C())}

        #self._model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/run_pure_policy/artoria_pendragon_iteration_190000.h5')
    
        #self._model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/it works/artoria_pendragon_iteration_5000.h5')
    def skill_1(self): #should be a crit boost but unsure how to do that atm??
        #print('skill_1')
        output_dict = {'name': 'sk1_'+self.spot+'_'+self.name+'_dreamlike_charisma',
                       'target': ['hero1','hero2','hero3'],
                       'hp_boost': 0,
                       'np_boost': 20,
                       'critical_boost': .00,
                       'critical_star_boost': 0,
                       'dmg_boost': ('all', .2),
                       'duration': 3}

        return output_dict

    def skill_2(self): 
        #print('skill_2') #check for this skill name?
        output_dict = {'name': 'sk2_'+self.spot+'_'+self.name+'_illlusion_A',
                       'target': ['hero1','hero2','hero3'],
                       'hp_boost': 0,
                       'np_boost': .00,
                       'critical_boost': .00,
                       'critical_star_boost': .0,
                       'dmg_boost': ('buster', .0),
                       'duration': 1}
        return output_dict

    def skill_3_A(self): #adds crit stars
        #print('skill_3')
        output_dict = {'name': 'sk3_'+self.spot+'_'+self.name+'_hero_creation_EX',
                       'target': ['hero1'],
                       'hp_boost': 0,
                       'np_boost': .00,
                       'critical_boost': 1.00,
                       'critical_star_boost': 0,
                       'dmg_boost': ('buster', .8),
                       'duration': 3}
        return output_dict
    def skill_3_B(self): #adds crit stars
        #print('skill_3')
        output_dict = {'name': 'sk3_'+self.spot+'_'+self.name+'_hero_creation_EX',
                       'target': ['hero2'],
                       'hp_boost': 0,
                       'np_boost': .00,
                       'critical_boost': 1.00,
                       'critical_star_boost': 0,
                       'dmg_boost': ('buster', .8),
                       'duration': 3}
        return output_dict
    def skill_3_C(self): #adds crit stars
        #print('skill_3')
        output_dict = {'name': 'sk3_'+self.spot+'_'+self.name+'_hero_creation_EX',
                       'target': ['hero3'],
                       'hp_boost': 0,
                       'np_boost': .00,
                       'critical_boost': 1.00,
                       'critical_star_boost': 0,
                       'dmg_boost': ('buster', .8),
                       'duration': 3}
        return output_dict

    def hero_creation_holder(self): #adds crit stars
        #print('skill_3')
        output_dict = {'name': 'sk3_holder'+self.spot+'_'+self.name+'_hero_creation_holder_EX',
                       'target': [self.spot],
                       'hp_boost': 0,
                       'np_boost': .00,
                       'critical_boost': 0.00,
                       'critical_star_boost': 0,
                       'dmg_boost': ('all', 0.0),
                       'duration': 3}
        return output_dict

    def garden_of_avalon(self):
        output_dict = {'name': 'NP_'+self.spot+'_'+self.name+'_garden_of_avalon',
                       'target': ['hero1','hero2','hero3'],
                       'hp_boost': 1, #is going to be 3x this amount
                       'np_boost': 5,
                       'critical_boost': 0.00,
                       'critical_star_boost': 0,
                       'dmg_boost': ('all', .0),
                       'duration': 5}
        return output_dict


class NeroCaster(HeroicSpirit):

    def __init__(self,
                 name='NeroCaster',
                 health=10,
                 NP=0,
                 NP_damage=40,
                 spot='hero1',
                 deck=None,
                 _epsilon=.3):
        super().__init__(name, health, NP, NP_damage, spot)

        self._epsilon = _epsilon
        self._model = hero_network()
        #self._model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/run_pure_policy/jalter_iteration_200000.h5')
        #self._model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/it works/jalter_iteration_5000.h5')
    def skill_1(self): #should be a crit boost but unsure how to do that atm??
        #print('skill_1')
        output_dict = {'name': 'sk1_'+self.spot+'_'+self.name+'_Rampaging_Privilege_EX',
                       'target': [self.spot],
                       'hp_boost': 0,
                       'np_boost': 50,
                       'critical_boost': 0.0,
                       'critical_star_boost': .0,
                       'dmg_boost': ('all', .00),
                       'duration': 1}

        return output_dict

    def skill_2(self): 
        #print('skill_2')
        output_dict = {'name': 'sk2_'+self.spot+'_'+self.name+'_seven_crowns',
                       'target': [self.spot],
                       'hp_boost': 0,
                       'np_boost': .00,
                       'critical_boost': .00,
                       'critical_star_boost': .0,
                       'dmg_boost': ('all', .25),
                       'duration': 3}
        return output_dict

    def skill_3(self):
        #print('skill_3')
        output_dict = {'name': 'sk3_'+self.spot+'_'+self.name+'_undying_magus_A',
                       'target': [self.spot],
                       'hp_boost': 0,
                       'np_boost': .0,
                       'critical_boost': .0,
                       'critical_star_boost': .0,
                       'dmg_boost': ('buster', .30),
                       'duration': 3}
        return output_dict

