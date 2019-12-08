import random
import copy
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import RMSprop,Adam
import numpy as np
import pandas as pd
#from colosseum import team_chaldela
import itertools
from random import choice
from keras.models import load_model

from utils import use_predicted_probability, convert_card_list
from netlearner import DQNLearner

class Chaldea():
    def __init__(self,
                 hero_list = None
                 ):
        super().__init__()
        self._learning = True
        self._epsilon = .3
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
                 _epsilon=.3,
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
        self.feature_vector_len = 2+9+9
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

        model = Sequential()

        model.add(Dense(128, init='glorot_normal', activation = 'relu', input_dim=self.feature_vector_len))

        model.add(Dense(128, init='glorot_normal', activation = 'relu'))
        model.add(Dense(64, init='glorot_normal', activation = 'relu'))
        model.add(Dense(32, init='glorot_normal', activation = 'relu'))

        model.add(Dense(len(self.action_dict), init='glorot_normal',activation='linear'))
        opt = RMSprop()
        model.compile(loss='mse', optimizer=opt)

        self._model = model

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

        new_pred_min = np.min(preds[0])-1
        for _index in range(len(self.action_dict)):
            if 'sk1' in self.action_dict[_index] and sk1_used == 1:
                preds[0][_index] = new_pred_min
            if 'sk2' in self.action_dict[_index] and sk2_used == 1:
                preds[0][_index] = new_pred_min
            if 'sk3' in self.action_dict[_index] and sk3_used == 1:
                preds[0][_index] = new_pred_min

    def get_random_action(self,random_pass = 1):
        action_dict_copy = copy.deepcopy(self.action_dict )
        if self.used_skill_1 == 1:
            del action_dict_copy[1]
        if self.used_skill_2 == 1:
            del action_dict_copy[2]
        if self.used_skill_3 == 1:
            del action_dict_copy[3]

        random_pass_list = [(0, ('pass', 'pass'))*(int(random_pass * len(self.action_dict)))]
        return random.choice(list(action_dict_copy.items())+random_pass_list)

    def get_action(self, state):
        # Take in game state and get a prediction
        game_state_array = np.reshape(np.asarray(state), (1, self.feature_vector_len))

        preds = self._model.predict(game_state_array, batch_size=1)
        #print(preds)
        self.invalid_steps(preds)
        #print(preds)
        predicted_classs = np.argmax(preds)

        # Exploration vs Exploitation!
        # if the randomly generated value is less than the epsilon value
        # we go down the Exploitation route... (I might have defined this backwards from normal...)
        if np.random.uniform(0, 1) < self._epsilon:
            action = use_predicted_probability(self.action_dict, predicted_classs)
        # When the random number is greater than the epsilon value we randomly select an action
        # and we see how it goes!
        else:
            random_action_key_value = self.get_random_action()
            predicted_classs = random_action_key_value[0]
            action = random_action_key_value[1]
            #action = use_predicted_probability(self.action_dict,predicted_classs)

        # store some stuff for later
        self._last_state = game_state_array
        self._last_action = predicted_classs
        self._last_target = preds

        return action

    def update(self, state, reward):
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
            outfit_state_array = np.reshape(np.asarray(state), (1, self.feature_vector_len))
            preds = self._model.predict([outfit_state_array], batch_size=1)
            self.invalid_steps(preds)
            maxQ = np.amax(preds)
            new = self._discount * maxQ #discount is applied bc it is a future action??? idk.. is standard?

            combined_reward = reward + new

            self._last_target[0][self._last_action] = combined_reward

            # at every update we are doing are training on a single batch of size 1
            self._model.fit(self._last_state, self._last_target, batch_size=1, epochs=1, verbose=0)
    
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

