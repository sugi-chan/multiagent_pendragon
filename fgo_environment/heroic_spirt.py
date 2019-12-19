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

from fgo_environment.utils import use_predicted_probability, convert_card_list
from fgo_environment.netlearner import DQNLearner
from fgo_environment.utils import use_predicted_probability, convert_card_list
def softmax(X):
    expo = np.exp(X)
    expo_sum = np.sum(np.exp(X))
    return expo/expo_sum
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


class HeroicSpirit():
    def __init__(self,
                 name,
                 health=1000,
                 NP=0,
                 NP_damage=20,
                 spot='hero1',
                 deck=None,
                 _epsilon=1.0,
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
        self.feature_vector_len = 2+9+9#1+9+9#
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
        self.initial_lr = 0.001
        self.class_weight = {0: 1.0,
                1: 2.0,
                2: 2.0,
                3: 2.0}

        model = Sequential()

        model.add(Dense(1024, init='glorot_normal', input_dim=self.feature_vector_len))#activation = 'relu', input_dim=self.feature_vector_len))
        #model.add(Dropout(0.4))
        #model.add(LeakyReLU(alpha=0.05))
        model.add(Dense(512, init='glorot_normal', activation = 'relu'))
        #model.add(Dropout(0.4))
        #model.add(LeakyReLU(alpha=0.05))



        model.add(Dense(len(self.action_dict), init='glorot_normal',activation='sigmoid'))
        opt = SGD(lr=self.initial_lr, momentum=0.9)#, clipnorm=2.0)
        model.compile(loss='binary_crossentropy', optimizer=opt)

        #model.add(Dense(256, init='glorot_normal', input_dim=self.feature_vector_len,kernel_regularizer=keras.regularizers.l2(l=0.02)))#activation = 'relu', input_dim=self.feature_vector_len))
        #model.add(Dropout(0.4))
        #model.add(LeakyReLU(alpha=0.03))
        #model.add(Dense(128, init='glorot_normal',kernel_regularizer=keras.regularizers.l2(l=0.02)))#$, activation = 'relu'))
        #model.add(Dropout(0.4))
        #model.add(LeakyReLU(alpha=0.03))
        #model.add(Dense(32, init='glorot_normal',kernel_regularizer=keras.regularizers.l2(l=0.02)))#, activation = 'relu'))
        #model.add(Dropout(0.4))
        #model.add(LeakyReLU(alpha=0.03))
        #model.add(Dense(64, init='glorot_normal',kernel_regularizer=keras.regularizers.l2(l=0.2)))#, activation = 'relu'))
        #model.add(LeakyReLU(alpha=0.01))

        #model.add(Dense(len(self.action_dict), init='glorot_normal',activation='sigmoid'))
        #opt = SGD(lr=self.initial_lr, momentum=0.9, clipnorm=2.0)
        #opt = Adam(learning_rate=self.initial_lr)
        #model.compile(loss='binary_crossentropy', optimizer=opt)

        self._model = model

        #print(self._model.summary())

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

        new_pred_min = 0 #np.min(preds[0])-.5 #with softmax everything is just probabilities
        for _index in range(len(self.action_dict)):
            if 'sk1' in self.action_dict[_index] and sk1_used == 1:
                preds[0][_index] = new_pred_min
            if 'sk2' in self.action_dict[_index] and sk2_used == 1:
                preds[0][_index] = new_pred_min
            if 'sk3' in self.action_dict[_index] and sk3_used == 1:
                preds[0][_index] = new_pred_min

    def get_random_action(self,random_pass = 2):
        action_dict_copy = copy.deepcopy(self.action_dict )
        if self.used_skill_1 == 1:
            del action_dict_copy[1]
        if self.used_skill_2 == 1:
            del action_dict_copy[2]
        if self.used_skill_3 == 1:
            del action_dict_copy[3]
        if random_pass==0:
            if len(action_dict_copy) >1:
                del action_dict_copy[0]



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
            #print('not_random')
            action = use_predicted_probability(self.action_dict, predicted_class)
        # When the random number is greater than the epsilon value we randomly select an action
        # and we see how it goes!
        else:
            
            if state[0] ==1/3:
                random_pass_number = 2
            if state[0] ==2/3:
                random_pass_number = 1
            if state[0] ==3/3:
                random_pass_number = 0
            
            #random_pass_number = 0
            #print('random')
            random_action_key_value = self.get_random_action(random_pass=random_pass_number)
            predicted_class = random_action_key_value[0]
            action = random_action_key_value[1]
            #if state[0] ==1/3:
            #    print('passing_ round1')
            #    predicted_class = 0
            #    action = ('pass', 'pass')
            #action = use_predicted_probability(self.action_dict,predicted_classs)

        # store some stuff for later
        self._last_state = game_state_array
        self._last_action = predicted_class
        self._last_target = preds

        return action,predicted_class, preds 

    def update(self, state_list, preds_list):
        '''
        reward:
                reward genearted from the game envionment
            state:
                game state
            new:
                discounted model outputs. This gets combined with with the game environment rewards
        '''
        if self._learning:


            game_state_array = np.reshape(np.asarray(state_list), (len(state_list), self.feature_vector_len))
            # In this version I call predict again... could just use self._last_target... *shrug
            #game_state_array = np.reshape(np.asarray(state), (1, self.feature_vector_len))
            preds_target = np.array([l[0].tolist() for l in preds_list])
            #preds = self._model.predict([outfit_state_array], batch_size=1)
            #self.invalid_steps(preds)
            #maxQ = np.amax(preds) # use the vale
            #maxQ = preds[0][predicted_class]
            #new = self._discount * maxQ #discount is applied bc it is a future action??? idk.. is standard?

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
            #print(preds,combined_reward,softmaxed_adjusted_preds)MaMagikagikrparp us used ed flyspla!.sh...!  wuandt?? IS??? NOw rethinking its life....

            # at every update we are doing are training on a single batch of size 1
            self._model.fit(game_state_array, preds_target, batch_size=2048, epochs=1000, verbose=0,class_weight=self.class_weight)
    
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

        self._model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/run9_policy_gradient_100game/jalter_iteration_9000.h5')

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
        super().__init__(name, health, NP, NP_damage,spot)

        self.model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/run9_policy_gradient_100game/Ishtar_iteration_2000.h5')

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
        super().__init__(name, health, NP, NP_damage,spot)
        self.model = load_model('D:/projects/multiagent_pendragon_dev_2/fgo_environment/models/run9_policy_gradient_100game/artoria_pendragon_iteration_3000.h5')
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

