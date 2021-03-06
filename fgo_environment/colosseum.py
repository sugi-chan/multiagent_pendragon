# extra imports to set GPU options
import tensorflow as tf
from keras import backend as k
import pandas as pd
###################################
# TensorFlow wizardry
config = tf.ConfigProto()
 
# Don't pre-allocate memory; allocate as-needed
config.gpu_options.allow_growth = True
 
# Only allow a total of half the GPU memory to be allocated
config.gpu_options.per_process_gpu_memory_fraction = 0.3
 
# Create a session with the above options specified.
k.tensorflow_backend.set_session(tf.Session(config=config))

import numpy as np

from fgo_environment.heroic_spirt import HeroicSpirit, Chaldea
import copy
from utils import use_predicted_probability, convert_card_list

from keras.models import load_model
#card types as classes
import itertools
from random import choice
from random import shuffle, randint

rl_model_name = 'new_chaldea_card_model_12_7_19'
save_dir = 'models'

'''
first two classes track the player and enemy teams
'''
class team_chaldela:
    #hp is total team hitpoints
    #deck is the total 15 card deck of the team
    
    def __init__(self):
        self.hit_points = hp
        self.card_deck = card_deck
        self.np_charge = 0
        self.current_stars = 0

    def get_action(self, state =None):
        print('hi')

class enemy_servants:
    
    def __init__(self,hp,dmg_max):
        self.hit_points = hp
        self.dmg_max = dmg_max
    def deal_damage(self):
        return randint(1, self.dmg_max ) 

'''
way to shuffle 15 card deck and deal 5 cards.
for now cards are replaced. in FGO there is no replacement
and deck is remade every 3 turns
'''
#def deal_hand(deck):
#    shuffle(deck)
#    
#    return deck[:5] 
def deal_hand(deck):
    shuffle(deck)
    final_deck = deck[:5]
    print(final_deck)
    servant_numbers = [i[0] for i in final_deck] 
    card_deck = [i[1] for i in final_deck] 

    return card_deck, servant_numbers 
'''
Next 3 Classes track the cards and give their damage, NP gain, and stars 
that are created when they are played
'''
class Buster:
    def __init__(self, card_number):
        if card_number == 1:
            self.dmg = 1.5
            self.np_gain = 0
            self.stars = 1
            
        if card_number == 2:
            self.dmg = 1.8
            self.np_gain = 0
            self.stars = 2
            
        if card_number == 3:
            self.dmg = 2.1
            self.np_gain = 0
            self.stars = 3
            
class Arts:
    def __init__(self, card_number):
        if card_number == 1:
            self.dmg = 1.0
            self.np_gain = 4
            self.stars = 0
            
        if card_number == 2:
            self.dmg = 1.2
            self.np_gain = 5
            self.stars = 0
            
        if card_number == 3:
            self.dmg = 1.4
            self.np_gain = 6
            self.stars = 0
            
class Quick:
    def __init__(self, card_number):
        if card_number == 1:
            self.dmg = .8
            self.np_gain = 2
            self.stars = 4
            
        if card_number == 2:
            self.dmg = 0.96
            self.np_gain = 3
            self.stars = 6
            
        if card_number == 3:
            self.dmg = 1.12
            self.np_gain = 4
            self.stars = 8
'''
Next 5 functions are used in the calculate damage function which is the core 
part of the loop where the card choices by the player are translated into
damage, np gain, stars created, critical damage is calculated etc.
'''     
#using the input list which is just a list of strings of card types
# convert them into card objects which have stats and such
def command_card_gen(card_type,current_card_num):
    
    if card_type == 'buster':
        card = Buster(current_card_num)
    elif card_type == 'arts':
        card = Arts(current_card_num)
    elif card_type == 'quick':
        card = Quick(current_card_num)
    return card

#there is a bonus that gets applied based on the first card. 
# so check the first card and then we can apply the modifiers 
# in the rest of the cards
def first_card_mods(first_card):
    
    if first_card == 'buster':
        dmg_modifier = 1.5
        np_mod = 1.0
        star_mod = 1
        
    elif first_card == 'arts':
        dmg_modifier = 1.0
        np_mod = 2.0
        star_mod = 1
        
    elif first_card == 'quick':
        dmg_modifier = 1.0
        np_mod = 1.0
        star_mod = 1.2
    else:
        print('invalid command card')
    return dmg_modifier,np_mod,star_mod

'''
I am a fan of the DnD series called critical role so I couldnt resist
'''
def critical_role(team_obj):
    critical_chance = team_obj.current_stars*2
    critical_flip = randint(1,100)
    if critical_flip <=critical_chance:
        critical_mod = 2.0 + team_obj.critical_strength
        #print('')
    else:
        critical_mod = 1.0
    
    team_obj.current_stars = 0
    team_obj.critical_strength = 0
    return critical_mod
    
# combine the previous functions to calculate the damage output and whatnot 
# of a single chain

'''
core damage calculations, takes in a the card chain and outputs damage
also the team class object and the rewards for the round so appropriate 
adjustments can be made
'''
'''
function to pick the cards out of the dealt hand that the network selects
'''
def pick_cards(current_hand5,card_indexes = [0,1,2]):
    #cards can take on values of 0-4 and should be a list of ints
    #default is first 3 cards
    hand_to_play = []
    for i in card_indexes:
        hand_to_play.append(current_hand5[i])
    
    return hand_to_play

def picked_cards_servants(current_hand5,card_indexes = [0,1,2]):
    #cards can take on values of 0-4 and should be a list of ints
    #default is first 3 cards
    servants_list = []
    for i in card_indexes:
        servants_list.append(current_hand5[i])
    
    return servants_list
'''
used for testing, and 10% of the time cards are randomly selected
'''
def random_pick_cards():
    list1= list(itertools.permutations([0,1,2,3,4],3))
    hand = choice(list1)
    return hand 

'''
Function to check if a team is still alive
'''
def check_if_alive(team_instance):
    #if team's hp is higher than 0 than return true, 
    # false when one player drops to 0 or below
    if team_instance.hit_points >0:
        return 'alive'
    else:
        return 'dead'

# deal hands
def deal_hand(deck):
    shuffle(deck)
    final_deck = deck[:5]
    #print(final_deck)
    servant_numbers = [i[0] for i in final_deck] 
    card_deck = [i[1] for i in final_deck] 

    return card_deck, servant_numbers 

#This function contains some of the rewards. 
# At some point I actually listed as a reward of +2 which is actually
# higher than winning. it worked, but may be a break in accepted protocol
def card_chain_effect(cards_to_play,team_obj,round_reward):
    #print(cards_to_play)
    #takes command card chain (list), checks for chain, applies bonuses
    buster_chain_mod = 1
    if cards_to_play.count('arts') == 2:
        if cards_to_play[0] == 'arts' and cards_to_play[2] == 'arts':
            round_reward+=1
    if cards_to_play.count('buster') == 2:
        if cards_to_play[0] == 'buster' and cards_to_play[2] == 'buster':
            round_reward+=1

    if cards_to_play.count('buster') == 3:
        #print('buster chain!')
        buster_chain_mod = 1.2
        round_reward+=2
        #print('')
    elif cards_to_play.count('arts') == 3:
        
        team_obj.hero1.NP_charge +=20 
        team_obj.hero2.NP_charge +=20 
        team_obj.hero3.NP_charge +=20
        round_reward+=2
    elif cards_to_play.count('quick') == 3:
        team_obj.current_stars +=10 #this is still fine
        round_reward+=2
    else:
        #print('')
        #print('not a chain')
        buster_chain_mod = 1
    return buster_chain_mod, round_reward

def get_hero_damage_mods(team_obj,card_type,hero_number):
    dmg_mod = team_obj.hero_dict[hero_number].damage_mod_dict[card_type]
    
    return dmg_mod


def charge_hero_np(team_obj,hero_number,np_gain):
    if hero_number == 'hero1':
        team_obj.hero1.NP_charge +=np_gain
    elif hero_number == 'hero2':
        team_obj.hero2.NP_charge +=np_gain
    elif hero_number == 'hero3':
        team_obj.hero3.NP_charge +=np_gain
        
def use_NP(team_obj,total_damage):
    if team_obj.hero1.NP_charge >= 100:
        #print(team_obj.hero1.name, 'used np')
        total_damage += team_obj.hero1.NP_damage
        team_obj.hero1.NP_charge = 0

    if team_obj.hero2.NP_charge >= 100:
        #print(team_obj.hero2.name, 'used np')
        total_damage += team_obj.hero2.NP_damage
        team_obj.hero2.NP_charge = 0

    if team_obj.hero3.NP_charge >= 100:
        #print(team_obj.hero3.name, 'used np')
        total_damage += team_obj.hero3.NP_damage
        team_obj.hero3.NP_charge = 0
    return total_damage

def calc_chain_damage(team_obj, card_chain, round_reward,servant_list):
    total_damage = 0
    first_card = card_chain[0]
    #print(first_card)
    dmg_modifier, np_mod, star_mod = first_card_mods(first_card)
    crit_mod = critical_role(team_obj)
    
    #Adding in NP charging from abilities
    charge_hero_np(team_obj,'hero1',team_obj.hero_dict['hero1'].NP_gain_buff)
    charge_hero_np(team_obj,'hero2',team_obj.hero_dict['hero2'].NP_gain_buff)
    charge_hero_np(team_obj,'hero3',team_obj.hero_dict['hero3'].NP_gain_buff)
    # zero out the np_gain_buffs
    team_obj.hero_dict['hero1'].NP_gain_buff = 0
    team_obj.hero_dict['hero2'].NP_gain_buff = 0
    team_obj.hero_dict['hero3'].NP_gain_buff = 0
    
    #using the NP should occur before card chain effects are calculated
    # player_team.hero_dict['hero2'].NP_gain_buff
    total_damage = use_NP(team_obj,total_damage)
    
    
    #arts and quick chain bonuses are executed in the 
    # card chain effect function
    #adding buster effect into the damage calculation    
    buster_chain_effect, round_reward = card_chain_effect(card_chain,team_obj,round_reward)
    
    for i in range(3):
        card = command_card_gen(card_chain[i],i+1)
        #star generation
        card_stars = card.stars*star_mod
        team_obj.current_stars += card_stars
        servant_damage_mod = get_hero_damage_mods(team_obj,card_chain[i],servant_list[i])
        #dmg 
        '''
        ADD IN DAMAGE MOD FROM SERVANT HERE SINCE WE GET TO THE CARD LEVEL
        '''
        card_dmg = card.dmg * dmg_modifier * buster_chain_effect*crit_mod * servant_damage_mod
        total_damage+= card_dmg
        
        #np
        card_np_gain = card.np_gain*np_mod
        charge_hero_np(team_obj,servant_list[i],card_np_gain)
        #team_obj.np_charge += card_np_gain #need a function to add to servant np
        
    if team_obj.hero1.NP_charge >= 100:
        round_reward +=1
    if team_obj.hero2.NP_charge >= 100:
        round_reward +=1
    if team_obj.hero3.NP_charge >= 100:
        round_reward +=1

    return total_damage, round_reward

def apply_dmg_boost(team_obj, action_skill):
    targets = action_skill['target']
    
    if action_skill['dmg_boost'][0] == 'all':
        cards_to_mod = ['buster','arts','quick']
    else:
        cards_to_mod = action_skill['dmg_boost'][0]
    dmg_mod = action_skill['dmg_boost'][1]
    
    for i in targets:
        for key, item in team_obj.hero_dict[i].damage_mod_dict.items():
            #print(key)
            if key in cards_to_mod:
                #print(key)
                team_obj.hero_dict[i].damage_mod_dict[key] = team_obj.hero_dict[i].damage_mod_dict[key]+ team_obj.hero_dict[i].damage_mod_dict[key]*dmg_mod
                
def reset_hero_dmg_mods(team_obj,hero_number):
    for key, item in team_obj.hero_dict[hero_number].damage_mod_dict.items():
        team_obj.hero_dict[hero_number].damage_mod_dict[key] = 1





def apply_buffs(team_obj, action_skill):
    targets = action_skill['target']
    hero1_buffs = copy.deepcopy(team_obj.hero_dict['hero1'].active_buffs_dict)
    hero2_buffs = copy.deepcopy(team_obj.hero_dict['hero2'].active_buffs_dict)
    hero3_buffs = copy.deepcopy(team_obj.hero_dict['hero3'].active_buffs_dict)
    #print(hero1_buffs,hero2_buffs,hero3_buffs)
    
    action_skill_copy = copy.deepcopy((copy.deepcopy(action_skill)))
    for target in targets:
        #print(target)
        if target == 'hero1':
            hero1_buffs[target+'_'+action_skill_copy['name']] = action_skill_copy
            team_obj.hero_dict['hero1'].active_buffs_dict = hero1_buffs
        elif target == 'hero2':
            hero2_buffs[target+'_'+action_skill_copy['name']] = action_skill_copy
            team_obj.hero_dict['hero2'].active_buffs_dict = hero2_buffs
        elif target == 'hero3':
            hero3_buffs[target+'_'+action_skill_copy['name']] = action_skill_copy
            team_obj.hero_dict['hero3'].active_buffs_dict = hero3_buffs

# for every hero add up damage for each type of damage?
# then add that to the heroes damage modifier
def reset_hero_damage_mods(team_obj):
    cards_to_mod = ['buster','arts','quick']
    hero_list = ['hero1','hero2','hero3']
    for card_type in cards_to_mod:
        for hero in hero_list:
            if team_obj.hero_dict[hero].name == 'Merlin': #adding merlin functionality?
                team_obj.hero_dict[hero].damage_mod_dict[card_type] = .8
            else:
                team_obj.hero_dict[hero].damage_mod_dict[card_type] = 1

# this is where I should track all the buffs getting applied
def increment_turns_for_buffs_gather_mods(team_obj):
    hero_list = ['hero1','hero2','hero3']
    hero_damage_mods_dict = {}
    #damage_mod_dict
    # reset new stuff before
    #print( team_obj.hero_dict['hero1'].damage_mod_dict)
    reset_hero_damage_mods(team_obj)
    #print( team_obj.hero_dict['hero1'].damage_mod_dict)
    hero1_dict_copy = copy.deepcopy(team_obj.hero_dict['hero1'].active_buffs_dict)
    hero2_dict_copy = copy.deepcopy(team_obj.hero_dict['hero2'].active_buffs_dict)
    hero3_dict_copy = copy.deepcopy(team_obj.hero_dict['hero3'].active_buffs_dict)
    
    hero_active_buff_copy_dict = {'hero1':hero1_dict_copy,'hero2':hero2_dict_copy,'hero3':hero3_dict_copy}
    
    hero1_dmg_dict_copy = copy.deepcopy(team_obj.hero_dict['hero1'].damage_mod_dict)
    hero2_dmg_dict_copy = copy.deepcopy(team_obj.hero_dict['hero2'].damage_mod_dict)
    hero3_dmg_dict_copy = copy.deepcopy(team_obj.hero_dict['hero3'].damage_mod_dict)
    
    hero_dmg_dict_copy_dict = {'hero1':hero1_dmg_dict_copy,
                               'hero2':hero2_dmg_dict_copy,
                               'hero3':hero3_dmg_dict_copy}
    #print(hero_dmg_dict_copy_dict)
    hero_dmg_hacky_index = 0

    for heros_buffs_key,heros_buffs_value in hero_active_buff_copy_dict.items():
        #print('wut',heros_buffs_value)
        for key, value in heros_buffs_value.items():
            #print(key,value)
            #value = copy.deepcopy(value)
            if value['duration'] > 0:
                
                ## accumulate NP boosts when looking at the things and store it somewhere
                if value['np_boost'] >0:
                     team_obj.hero_dict[heros_buffs_key].NP_gain_buff += value['np_boost']
                # Add in critical stars?
                if value['critical_star_boost'] >0: #this one gets added to then set to 0 during calc damage
                     team_obj.current_stars += value['critical_star_boost']
                        
                if value['critical_boost'] >0: #gets added at hero level... needs to gert zeroed
                    team_obj.critical_strength += value['critical_boost']

                if value['hp_boost'] >0: #
                    
                    team_obj.hit_points += value['hp_boost']

                    if team_obj.hit_points > 30:
                        team_obj.hit_points = 30

                ############
                if value['dmg_boost'][0] == 'all':
                    cards_to_mod = ['buster','arts','quick']
                else:
                    cards_to_mod = [value['dmg_boost'][0]]

                for card_type in cards_to_mod:
                    #for hero_dmg_dict in hero_dmg_dict_copy_list:
                    #print(card_type)
                    hero_dmg_dict_copy_dict[heros_buffs_key][card_type]+=value['dmg_boost'][1]
                    #hero1_dmg_dict_copy[card_type] +=value['dmg_boost'][1]
                    #hero2_dmg_dict_copy[card_type] +=value['dmg_boost'][1]
                    #hero3_dmg_dict_copy[card_type] +=value['dmg_boost'][1]
                        #hero_dmg_dict[card_type] += value['dmg_boost'][1]
                heros_buffs_value[key]['duration'] -= 1

    for i in list(hero1_dict_copy.keys()):
        if hero1_dict_copy[i]['duration'] <=0:
            del hero1_dict_copy[i]
    
    for i in list(hero2_dict_copy.keys()):
        if hero2_dict_copy[i]['duration'] <=0:
            del hero2_dict_copy[i]
    
    for i in list(hero3_dict_copy.keys()):
        if hero3_dict_copy[i]['duration'] <=0:
            del hero3_dict_copy[i]

    team_obj.hero_dict['hero1'].active_buffs_dict = hero1_dict_copy
    team_obj.hero_dict['hero2'].active_buffs_dict = hero2_dict_copy
    team_obj.hero_dict['hero3'].active_buffs_dict = hero3_dict_copy
    
    team_obj.hero_dict['hero1'].damage_mod_dict = hero1_dmg_dict_copy
    team_obj.hero_dict['hero2'].damage_mod_dict = hero2_dmg_dict_copy
    team_obj.hero_dict['hero3'].damage_mod_dict = hero3_dmg_dict_copy      


def get_buff_game_state(team_obj,return_dict = False):
    skill_dict = {'sk1_hero1':0,'sk2_hero1':0,'sk3_hero1':0,
                 'sk1_hero2':0,'sk2_hero2':0,'sk3_hero2':0,
                 'sk1_hero3':0,'sk2_hero3':0,'sk3_hero3':0}
    hero1_dict_copy = copy.deepcopy(team_obj.hero_dict['hero1'].active_buffs_dict)
    hero2_dict_copy = copy.deepcopy(team_obj.hero_dict['hero2'].active_buffs_dict)
    hero3_dict_copy = copy.deepcopy(team_obj.hero_dict['hero3'].active_buffs_dict)

    hero_active_buff_copy_dict = {'hero1':hero1_dict_copy,'hero2':hero2_dict_copy,'hero3':hero3_dict_copy}
    for heros_buffs_key,heros_buffs_value in hero_active_buff_copy_dict.items():
        if not heros_buffs_value: # check if empty
            continue
        else:
            for name, buff in heros_buffs_value.items():
                #print(name)
                skill_name = name.split('_')
                skill_dict[skill_name[1]+'_'+skill_name[2]] = 1

    active_buff_state = [skill_dict['sk1_hero1'],skill_dict['sk2_hero1'],skill_dict['sk3_hero1'],
                         skill_dict['sk1_hero2'],skill_dict['sk2_hero2'],skill_dict['sk3_hero2'],
                         skill_dict['sk1_hero3'],skill_dict['sk2_hero3'],skill_dict['sk3_hero3']]
    if return_dict == True:
        return active_buff_state,skill_dict

    return active_buff_state


def get_skill_use_game_state(team_obj):
    skill_use_list = []
    #print(team_obj.hero_dict['hero1'].used_skill_1)
    skill_use_list.append(team_obj.hero_dict['hero1'].used_skill_1)
    skill_use_list.append(team_obj.hero_dict['hero1'].used_skill_2)
    skill_use_list.append(team_obj.hero_dict['hero1'].used_skill_3)
    
    skill_use_list.append(team_obj.hero_dict['hero2'].used_skill_1)
    skill_use_list.append(team_obj.hero_dict['hero2'].used_skill_2)
    skill_use_list.append(team_obj.hero_dict['hero2'].used_skill_3)
    
    skill_use_list.append(team_obj.hero_dict['hero3'].used_skill_1)
    skill_use_list.append(team_obj.hero_dict['hero3'].used_skill_2)
    skill_use_list.append(team_obj.hero_dict['hero3'].used_skill_3)
    
    return skill_use_list
    
def check_active_buffs_for_rewards(skill_dict):
    hero_list = ['hero1','hero2','hero3']
    skill_list = ['sk1','sk2','sk3']
    
    hero1_reward_bonus = 0
    hero2_reward_bonus = 0
    hero3_reward_bonus = 0
    
    for hero in hero_list:
        
        if hero == 'hero1':
            if skill_dict['sk1_'+hero] == 1:
                hero1_reward_bonus = 1
            elif skill_dict['sk3_'+hero] == 1:
                hero1_reward_bonus = 1
            elif skill_dict['sk3_'+hero] == 1:
                hero1_reward_bonus = 1

        elif hero == 'hero2':
            if skill_dict['sk1_'+hero] == 1:
                hero2_reward_bonus = 1
            elif skill_dict['sk3_'+hero] == 1:
                hero2_reward_bonus = 1
            elif skill_dict['sk3_'+hero] == 1:
                hero2_reward_bonus = 1
                
        elif hero == 'hero3':
            if skill_dict['sk1_'+hero] == 1:
                hero3_reward_bonus = 1
            elif skill_dict['sk3_'+hero] == 1:
                hero3_reward_bonus = 1
            elif skill_dict['sk3_'+hero] == 1:
                hero3_reward_bonus = 1
                
    return hero1_reward_bonus,hero2_reward_bonus,hero3_reward_bonus

<<<<<<< Updated upstream
=======
def process_rewards_get_finalized_preds(preds_list,action_list_actions,reward_list,hero_action_dict):
    def invalid_steps_game_eval(preds,sk1,sk2,sk3,hero_action_dict,super_sk3_used):
        sk1_used = sk1
        sk2_used = sk2
        sk3_used = sk3

        
        new_pred_min = 0 #with softmax everything is just probabilities
        
        for _index in range(len(hero_action_dict)):
            #print(_index)
            #if _index > 3:
                #print(_index, sk3_used,'_hero_creation_holder_EX' in hero_action_dict[_index][1]['name'],hero_action_dict[_index])
            if 'sk1' in hero_action_dict[_index] and sk1_used == 1:
                #print('sk1 action?',hero_action_dict[_index])
                preds[0][_index] = new_pred_min
            if 'sk2' in hero_action_dict[_index] and sk2_used == 1:
                #print(preds[0][_index])
                preds[0][_index] = new_pred_min
            if 'sk3' in hero_action_dict[_index] and sk3_used == 1:
                preds[0][_index] = new_pred_min

            if super_sk3_used == 1 and _index>=3: #need to handle the case where self is not buffed?
                preds[0][_index] = new_pred_min

        
    used_skill_1 = 0
    used_skill_2 = 0
    used_skill_3 = 0
    super_sk3_used = 0
    preds_list2 = copy.deepcopy(preds_list)
    for i in range(len(action_list_actions)):

        reward = reward_list[i]
        action_number = action_list_actions[i]
        #print(reward,action_number)
        #if len(hero_action_dict) >4:
        #    print('')
        #    print(reward,action_number,preds_list2[i])
       
        if reward >=0:
            #print(preds_list2[i][0], reward,np.amax(preds_list2[i][0]),preds_list2[i][0][action_number]+ reward)

            preds_list2[i][0][action_number] = np.amax(preds_list2[i][0])+ reward
        else:
            #print(preds_list2[i][0][action_number], reward,preds_list2[i][0][action_number]+ reward)
            preds_list2[i][0][action_number] = preds_list2[i][0][action_number]+ reward

        #print(preds_list2[i][0], 'after')
        
        

        #modify all the other ones up or down
        #action_list = copy.deepcopy(list(hero_action_dict.keys()))
        #if action_number in action_list: action_list.remove(action_number)
        #for action in action_list:
        #    preds_list2[i][0][action] = preds_list2[i][0][action] + group_reward_mod
        '''
        nonzero_idxs = np.where(preds_list2[i][0] > 1)[0]
        preds_list2[i][0][nonzero_idxs] = 1

        nonzero_idxs = np.where(preds_list2[i][0] < 0)[0]
        preds_list2[i][0][nonzero_idxs] = 0
        '''
        #if len(hero_action_dict) >4:
        #    print('post reward',preds_list2[i])
            
        invalid_steps_game_eval(preds_list2[i],used_skill_1,used_skill_2,used_skill_3,hero_action_dict,super_sk3_used)
        #if len(hero_action_dict) >4:
        #    print(preds_list2[i])
            
        if preds_list2[i][0][0] <= 0:
            preds_list2[i][0][0] = 0
        if preds_list2[i][0][1] <= 0:
            preds_list2[i][0][1] = 0
        if preds_list2[i][0][2] <= 0:
            preds_list2[i][0][2] = 0
        if preds_list2[i][0][3] <= 0:
            preds_list2[i][0][3] = 0

        if len(preds_list2[i][0]) >4:
            if preds_list2[i][0][4] <= 0:
                preds_list2[i][0][4] = 0
            if preds_list2[i][0][5] <= 0:
                preds_list2[i][0][5] = 0

        #if len(hero_action_dict) >4:
        #    print('between 0 and 1',preds_list2[i])
        array_total = np.sum(preds_list2[i][0])

        
        #print(preds_list2[i][0], 'normalized')
        

        if preds_list2[i][0][0] > 0:
            preds_list2[i][0][0] = preds_list2[i][0][0] / array_total
        
        
        if preds_list2[i][0][1] > 0:
            preds_list2[i][0][1] = preds_list2[i][0][1] / array_total
        

        if preds_list2[i][0][2] > 0:
            preds_list2[i][0][2] = preds_list2[i][0][2] / array_total
        

        if preds_list2[i][0][3] > 0:
            preds_list2[i][0][3] = preds_list2[i][0][3] / array_total
        
        if len(preds_list2[i][0]) >4:
            if preds_list2[i][0][4] > 0:
                preds_list2[i][0][4] = preds_list2[i][0][4] / array_total
            if preds_list2[i][0][5] > 0:
                preds_list2[i][0][5] = preds_list2[i][0][5] / array_total

        #if len(hero_action_dict) >4:
        #    print('softmax',preds_list2[i])
        if np.count_nonzero(preds_list2[i][0]) == 0:
            preds_list2[i][0][0] = 1

        #preds_list2[i][0][0] =  = preds_list2[i][0] / array_total # convert back to probabilty distribution

        if action_list_actions[i] == 1:
            #print('sk 1 used')
            used_skill_1 = 1
        if action_list_actions[i] == 2:
            used_skill_2 = 1

        if action_list_actions[i] == 3:
            used_skill_3 = 1

        if len(preds_list2[i][0]) >4:
            if action_list_actions[i] == 3:
                super_sk3_used = 1
            if action_list_actions[i] == 4:
                #print(action_list_actions[i] ,'is four?')
                super_sk3_used = 1
            if action_list_actions[i] == 5:
                super_sk3_used = 1
        #if len(hero_action_dict) >4:
        #    print('final',preds_list2[i])
        #    print('')


    return preds_list2
>>>>>>> Stashed changes

#apply_dmg_boost(player_team,player_team.hero1.skill_1())
'''
Core of the process where the environment is created and bot is able 
to train. tracks wins and losses, function fight_battle represents 1 game
to be played
the reset function makes it so the games are always initialized at the same point
the reporting function is helpful for evaluating the output, I have it set 
so when it reports it will also play through a game which helps to show bot behaviour
as the training process goes on
'''


class Battle:
    
<<<<<<< Updated upstream
    def __init__(self, num_learning_rounds =None, learner = Chaldea(), report_every=100):
=======
    def __init__(self, num_learning_rounds =None, learner = Chaldea(), report_every=10000):
>>>>>>> Stashed changes
        self._num_learning_rounds = num_learning_rounds
        self._report_every = report_every
        self.player = learner
        self.win = 0
        self.loss = 0
        self.game = 1
        self.evaluation = False
        self.og_hitpoints = self.player.hit_points
        self.result_list = []
        self.game_count_list = []
<<<<<<< Updated upstream
=======
        self.epsilon_list = []

        self.win_len_full = [] # This one does not
        self.game_len_list = [] # This one gets reset

        self.lr_list = []
>>>>>>> Stashed changes
        self.lr_holder = self.player.hero1.initial_lr
        
    def fight_battle(self):
        #print(self.game)
        player_team = self.reset_battle()
        #print('####################################')
        #print(player_team.hero2.used_skill_1,player_team.hero2.used_skill_2,player_team.hero2.used_skill_3)
        #print('####################################')
        player_team.card_deck = player_team.deck
        turn_counter = 1
        # storing things per game.
        hero1_game_state_list = []
        hero2_game_state_list = []
        hero3_game_state_list = []

        hero1_preds_list = []
        hero2_preds_list = []
        hero3_preds_list = []

        hero1_predicted_class_list = []
        hero2_predicted_class_list = []
        hero3_predicted_class_list = []

        hero1_action_list = []
        hero2_action_list = []
        hero3_action_list = []

        hero1_rewards_list = []
        hero2_rewards_list = []
        hero3_rewards_list = []
        #print(player_team.hero1._model.summary())
        #print(player_team.hero2._model.summary())
        #print(player_team.hero3._model.summary())
        for game in range(1, 4):
            if self.evaluation == True:
                print('')
                print('#'*50)
                print('ITS SHOW TIME! round: ',game)
                print('#'*50)
                print('reset enemies')
            if game == 1:
                r1 = [30, 35, 40,45] 
                # hard 40
                enemy_team = enemy_servants(hp=40,dmg_max = 3)
            if game == 2:
                r1 = [40,45,50]
                # hard 65
                enemy_team = enemy_servants(hp=43,dmg_max = 4)
                #enemy_team = enemy_servants(hp=65,dmg_max = 4)
            if game == 3:
                r1 = [55, 60, 50,65]
<<<<<<< Updated upstream
                enemy_team = enemy_servants(hp=50,dmg_max = 5)
=======
                # hard 80
                enemy_team = enemy_servants(hp=44,dmg_max = 5)
                #enemy_team = enemy_servants(hp=80,dmg_max = 5)
>>>>>>> Stashed changes
            if check_if_alive(player_team) =='alive':
                while True:
                    round_reward = 0
                    hero1_reward = 0
                    hero2_reward = 0
                    hero3_reward = 0
                    pass_reward = 0
                    enemy_dps = enemy_team.deal_damage()

                    #### PLAYER TURN SECTION
                    current_hand, servant_numbers = deal_hand(player_team.card_deck)
                    #if self.evaluation ==True:
                    #print('hand is: ',current_hand)
                    #send list to the model

                    #it should return an iterable? i think a tuple? of length 3 which correspond
                    # to the card indicies
                    if self.evaluation == True:
                        print('#'*10,'hero1 damage mod dict Before','#'*10)
                        print('hero1:',player_team.hero_dict['hero1'].damage_mod_dict)
                        print('hero2:',player_team.hero_dict['hero2'].damage_mod_dict)
                        print('hero3:',player_team.hero_dict['hero3'].damage_mod_dict)

                    ### GET THOSE ACTIONS! ###
                    # game state
                    # do hero 1 actions and whatever
                    # calculate pass reward
                    '''
                    if turn_counter <= 6:
                        pass_reward += 1
                    if turn_counter < 10 and turn_counter > 6:
                        pass_reward += .5
                    if game == 0:
                        pass_reward += 2 
                    if game == 1:
                        pass_reward += 1.5 
                    if game == 2:
                        pass_reward += -2
                    #if pass_reward < 0:
                    #    pass_reward = 0
                    '''
                    #hero1_state = [game]+[turn_counter]+get_buff_game_state(player_team)+get_skill_use_game_state(player_team)
                    #hero1_state = [game/3]+[turn_counter/15]+get_buff_game_state(player_team)+get_skill_use_game_state(player_team)
                    #hero1_state = [game/3]+get_buff_game_state(player_team)+get_skill_use_game_state(player_team)
                    #hero1_state = [game/3]+[turn_counter/15]#+get_buff_game_state(player_team)+get_skill_use_game_state(player_team)
                    hero1_state = [game]+[turn_counter]+get_skill_use_game_state(player_team)

                    hero1_action, hero1_pred_class, hero1_preds = player_team.hero_dict['hero1'].get_action(hero1_state)
                    #action,predicted_classs, preds
                    hero1_game_state_list.append(hero1_state)
                    hero1_preds_list.append(hero1_preds)
                    hero1_predicted_class_list.append(hero1_pred_class)
                    #print(self.game,game,turn_counter,hero1_action[0],hero1_preds,hero1_state)
                    if self.evaluation == True:
                        print('hero1:',hero1_action,hero1_preds,)
                    if hero1_action[0] != 'pass':
                        '''
                        Going to place merlin here
                        '''
                        apply_buffs(player_team,hero1_action[1])
                        if 'sk1' in hero1_action[0]:
                            player_team.hero_dict['hero1'].used_skill_1 = 1
                        elif 'sk2' in hero1_action[0]:
                            player_team.hero_dict['hero1'].used_skill_2 = 1
                            if player_team.hero_dict['hero1'].name == 'Merlin':
                                enemy_dps = 0 # set damage to 0 for merlin's second skill
                        elif 'sk3' in hero1_action[0]:
                            apply_buffs(player_team,player_team.hero1.hero_creation_holder())
                            player_team.hero_dict['hero1'].used_skill_3 = 1

                    if hero1_action[0] == 'pass':
                        hero1_action_list.append('pass')
                        #hero1_reward = pass_reward
                    else:
                        hero1_action_list.append('skill_used')
                    if self.evaluation == True:
                        print('game_state 1: game',game,'turn: ',turn_counter,'buffs',get_buff_game_state(player_team), get_skill_use_game_state(player_team))
                    #hero2_state = [game]+[turn_counter]+get_buff_game_state(player_team)+get_skill_use_game_state(player_team)
                    #hero2_state = [game/3]+[turn_counter/15]+get_buff_game_state(player_team)+get_skill_use_game_state(player_team)
                    #hero2_state = [game/3]+get_buff_game_state(player_team)+get_skill_use_game_state(player_team)
                    #hero2_state = [game/3]+[turn_counter/15]#+get_buff_game_state(player_team)+get_skill_use_game_state(player_team)
                    hero2_state = [game]+[turn_counter]+get_skill_use_game_state(player_team)

                    hero2_action, hero2_pred_class, hero2_preds = player_team.hero_dict['hero2'].get_action(hero2_state)

                    hero2_game_state_list.append(hero2_state)
                    hero2_preds_list.append(hero2_preds)
                    hero2_predicted_class_list.append(hero2_pred_class)

                    if self.evaluation == True:
                        print('hero2:',hero2_action)
                    #print(game,turn_counter,hero2_action[0])
                    if hero2_action[0] != 'pass':
                        #if hero2_action[0] == 'passsss':
                        #print(hero2_action)
                        '''
                        hero2_reward = .5
                        if game == 1:
                            hero2_reward +=1
                        if game == 2:
                            hero2_reward +=1.5
                        '''
                        apply_buffs(player_team,hero2_action[1])
                        if 'sk1' in hero2_action[0]:
                            player_team.hero_dict['hero2'].used_skill_1 = 1
                        elif 'sk2' in hero2_action[0]:
                            player_team.hero_dict['hero2'].used_skill_2 = 1
                        elif 'sk3' in hero2_action[0]:
                            player_team.hero_dict['hero2'].used_skill_3 = 1
                    if hero2_action[0] == 'pass':
                        hero2_action_list.append('pass')
                        #hero1_reward = pass_reward
                    else:
                        hero2_action_list.append('skill_used')
              
                    if self.evaluation == True:
                        print('game_state 2: game',game,'turn: ',turn_counter,'buffs',get_buff_game_state(player_team), get_skill_use_game_state(player_team))
                    #hero3_state = [game]+[turn_counter]+get_buff_game_state(player_team)+get_skill_use_game_state(player_team)
                    ##hero3_state = [game/3]+[turn_counter/15]+get_buff_game_state(player_team)+get_skill_use_game_state(player_team)
                    #hero3_state = [game/3]+get_buff_game_state(player_team)+get_skill_use_game_state(player_team)
                    #hero3_state = [game/3]+[turn_counter/15]#+get_buff_game_state(player_team)+get_skill_use_game_state(player_team)
                    hero3_state = [game]+[turn_counter]+get_skill_use_game_state(player_team)

                    hero3_action, hero3_pred_class, hero3_preds = player_team.hero_dict['hero3'].get_action(hero3_state)

                    hero3_game_state_list.append(hero3_state)
                    hero3_preds_list.append(hero3_preds)
                    hero3_predicted_class_list.append(hero3_pred_class)

                    if self.evaluation == True:
                        print('hero3:',hero3_action)


                    if hero3_action[0] != 'pass':
                        #if hero3_action[0] == 'passss':
                        '''
                        hero3_reward = .5
                        if game == 1:
                            hero3_reward +=1
                        if game == 2:
                            hero3_reward +=1.5

                        '''
                        apply_buffs(player_team,hero3_action[1])
                        if 'sk1' in hero3_action[0]:
                            player_team.hero_dict['hero3'].used_skill_1 = 1
                        elif 'sk2' in hero3_action[0]:
                            player_team.hero_dict['hero3'].used_skill_2 = 1
                        elif 'sk3' in hero3_action[0]:
                            player_team.hero_dict['hero3'].used_skill_3 = 1

                    if hero3_action[0] == 'pass':
                        hero3_action_list.append('pass')
                        #hero1_reward = pass_reward
                    else:
                        hero3_action_list.append('skill_used')
                    if self.evaluation == True:
                        print('game_state 3: game',game,'turn: ',turn_counter,'buffs',get_buff_game_state(player_team), get_skill_use_game_state(player_team))

                    increment_turns_for_buffs_gather_mods(player_team)  
                    if self.evaluation == True:
                        print('#'*10,'hero1 damage mod dict after','#'*10)
                        print('hero1:',player_team.hero_dict['hero1'].damage_mod_dict)
                        print('hero2:',player_team.hero_dict['hero2'].damage_mod_dict)
                        print('hero3:',player_team.hero_dict['hero3'].damage_mod_dict)
                        print('#'*60)
                        print('hero1 active buffs:',player_team.hero_dict['hero1'].active_buffs_dict)
                        print('#'*60)
                        print('hero2 active buffs:',player_team.hero_dict['hero2'].active_buffs_dict)
                        print('#'*60)
                        print('hero3 active buffs:',player_team.hero_dict['hero3'].active_buffs_dict)
                        print('#'*60)
                    #3 card tuple fed into pick_cards function which will generate the hand and start
                    #dmg calculations
                    converted_current_hand = convert_card_list(current_hand)
<<<<<<< Updated upstream
                    card_picker_state = converted_current_hand+get_buff_game_state(player_team)#+get_skill_use_game_state(player_team)

=======
                    #print(type(converted_current_hand),type(get_buff_game_state(player_team)))

                    card_picker_state = list(converted_current_hand[0])#+ get_buff_game_state(player_team)#+get_skill_use_game_state(player_team)
                    
>>>>>>> Stashed changes
                    p1_action = player_team.card_picker.get_action(card_picker_state) #comment out for auto testing

                    #p1_action = random_pick_cards() #testing pipeline
                    if self.evaluation == True:
                        print('game_state card picker: game',game,'turn: ',turn_counter,'buffs',get_buff_game_state(player_team), get_skill_use_game_state(player_team))

                    picked_cards = pick_cards(current_hand,p1_action) #
                    picked_servants = picked_cards_servants(servant_numbers,p1_action)
                    if self.evaluation == True:
                        print('current hand: ',current_hand)
                        print('cards to play: ',picked_cards, 'indexes: ', p1_action, 'servants: ',picked_servants)

                    team_dps, round_reward = calc_chain_damage(player_team,picked_cards, round_reward, picked_servants)  
                    '''
                    MERLIN ULT
                    '''  
                    if player_team.hero1.name == 'Merlin' and player_team.hero1.NP_charge >= 100:
                        apply_buffs(player_team,player_team.hero1.garden_of_avalon())
                        #garden_of_avalon
                    if self.evaluation == True:
                        print('team deals ',team_dps,'team hp',player_team.hit_points, ' current np bars: ', player_team.hero1.NP_charge,player_team.hero2.NP_charge,player_team.hero3.NP_charge, player_team.current_stars)
                        print('enemy deals ',enemy_dps)

                    player_team.hit_points -= enemy_dps
                    enemy_team.hit_points -= team_dps
 
                    #check if player is still alive
                    turn_counter += 1 #always keep incrementing the turn?
                    
                    if check_if_alive(player_team) =='alive' and check_if_alive(enemy_team) == 'dead':
                        winner = 'player'
                        
                        #round_reward+=1
                        ## check active skills, gets bonus for having them active towards end of rounds?
                        #skill_dict = {'sk1_hero1':0,'sk2_hero1':0,'sk3_hero1':0,
                        ## 'sk1_hero2':0,'sk2_hero2':0,'sk3_hero2':0,
                        # 'sk1_hero3':0,'sk2_hero3':0,'sk3_hero3':0}
                        _, active_skill_dict = get_buff_game_state(player_team, return_dict = True)
                        h1_bonus, h2_bonus, h3_bonus = check_active_buffs_for_rewards(active_skill_dict)
                        
<<<<<<< Updated upstream
                        if game == 3: 
                            round_reward+=1
                            self.win +=1
                            hero_rewards = 2
=======
                        #print(self.game,game,turn_counter, 'HERO1 SKILLS USED IN GAME', player_team.hero_dict['hero1'].used_skill_1, player_team.hero_dict['hero1'].used_skill_2, player_team.hero_dict['hero1'].used_skill_3)

                        if game == 3:
                            hero_rewards = 1
                            '''
                            TRYING TO 3 TURN IT
                            '''
                            if len(hero1_game_state_list) <=5:
                                hero_rewards = 1
                            if len(hero1_game_state_list) >=5:
                                hero_rewards = -.25
                            
                            
                            self.win +=1
                            
                            #print(self.game,game,turn_counter,  'HERO1 SKILLS USED IN GAME', player_team.hero_dict['hero1'].used_skill_1, player_team.hero_dict['hero1'].used_skill_2, player_team.hero_dict['hero1'].used_skill_3)
                            #print('')
>>>>>>> Stashed changes
                            if self.evaluation == True:
                                print('game is a win')
                                print('#'*60)
                                print('preds example hero 1')
                                print(hero1_game_state_list,hero1_preds_list,hero1_predicted_class_list)
                                print('#'*60)
                                print('#'*60)
                                print('preds example hero 2')
                                print(hero2_game_state_list,hero2_preds_list,hero2_predicted_class_list)
                                print('#'*60)
                                print('#'*60)
                                print('preds example hero 3')
                                print(hero3_game_state_list,hero3_preds_list,hero3_predicted_class_list)
                                print('#'*60)




                            if self.evaluation == False:
                                '''
                                hero1_reward += h1_bonus
                                hero2_reward += h2_bonus
                                hero3_reward += h3_bonus

                                #hero3_game_state.append(hero3_state)
                                #hero3_preds.append(hero3_preds)
                                #hero3_predicted_class.append(hero3_pred_class)
                                #self, state, preds,predicted_classs, reward
                                #hero1_action_list.append('skill_used')
                                '''
                                #hero1_rewards_list = []
                                #hero2_rewards_list = []
                                #hero3_rewards_list = []
                                for i in range(len(hero1_game_state_list)):
                                    if hero1_game_state_list[i][0] == 1: 
                                        if hero1_action_list[i] == 'pass':
<<<<<<< Updated upstream
                                            hero1_rewards_list.append(hero_rewards+1)
=======
                                            hero1_rewards_list.append(hero_rewards)
>>>>>>> Stashed changes
                                        else:
                                            hero1_rewards_list.append(hero_rewards)

                                        if hero2_action_list[i] == 'pass':
<<<<<<< Updated upstream
                                            hero2_rewards_list.append(hero_rewards+1)
=======
                                            hero2_rewards_list.append(hero_rewards)
>>>>>>> Stashed changes
                                        else:
                                            hero2_rewards_list.append(hero_rewards)

                                        if hero3_action_list[i] == 'pass':
<<<<<<< Updated upstream
                                            hero3_rewards_list.append(hero_rewards+1)
=======
                                            hero3_rewards_list.append(hero_rewards)
>>>>>>> Stashed changes
                                        else:
                                            hero3_rewards_list.append(hero_rewards)

                                    if hero1_game_state_list[i][0] == 2: 
                                        if hero1_action_list[i] == 'pass':
                                            hero1_rewards_list.append(hero_rewards)
                                        else:
<<<<<<< Updated upstream
                                            hero1_rewards_list.append(hero_rewards+1)
=======
                                            hero1_rewards_list.append(hero_rewards)
>>>>>>> Stashed changes

                                        if hero2_action_list[i] == 'pass':
                                            hero2_rewards_list.append(hero_rewards)
                                        else:
<<<<<<< Updated upstream
                                            hero2_rewards_list.append(hero_rewards+1)
=======
                                            hero2_rewards_list.append(hero_rewards)
>>>>>>> Stashed changes

                                        if hero3_action_list[i] == 'pass':
                                            hero3_rewards_list.append(hero_rewards)
                                        else:
<<<<<<< Updated upstream
                                            hero3_rewards_list.append(hero_rewards+1)
                                    if hero1_game_state_list[i][0] == 3/3: 
=======
                                            hero3_rewards_list.append(hero_rewards)
                                    if hero1_game_state_list[i][0] == 3: 
>>>>>>> Stashed changes
                                        if hero1_action_list[i] == 'pass':
                                            hero1_rewards_list.append(hero_rewards)
                                        else:
<<<<<<< Updated upstream
                                            hero1_rewards_list.append(hero_rewards+1.5)
=======
                                            hero1_rewards_list.append(hero_rewards)
>>>>>>> Stashed changes

                                        if hero2_action_list[i] == 'pass':
                                            hero2_rewards_list.append(hero_rewards)
                                        else:
<<<<<<< Updated upstream
                                            hero2_rewards_list.append(hero_rewards+1.5)
=======
                                            hero2_rewards_list.append(hero_rewards)
>>>>>>> Stashed changes

                                        if hero3_action_list[i] == 'pass':
                                            hero3_rewards_list.append(hero_rewards)
                                        else:
<<<<<<< Updated upstream
                                            hero3_rewards_list.append(hero_rewards+1.5)
                                #print(len(hero1_game_state_list))
                                '''
                                for i in range(len(hero1_game_state_list)):
                                    player_team.hero_dict['hero1'].update(hero1_game_state_list[i],hero1_preds_list[i],hero1_predicted_class_list[i],hero1_rewards_list[i])
                                    player_team.hero_dict['hero2'].update(hero2_game_state_list[i],hero2_preds_list[i],hero2_predicted_class_list[i],hero2_rewards_list[i])
                                    player_team.hero_dict['hero3'].update(hero3_game_state_list[i],hero3_preds_list[i],hero3_predicted_class_list[i],hero3_rewards_list[i])
                                '''
=======
                                            hero3_rewards_list.append(hero_rewards)

                                if 'skill_used' not in hero1_action_list:
                                    hero1_action_list_no_skill = []
                                    for i in hero1_action_list:
                                        hero1_action_list_no_skill.append(-1)
                                    hero1_rewards_list = hero1_action_list_no_skill


                                #if hero1_action_list[0] == 'skill_used' and hero1_action_list[1] == 'skill_used' and hero1_action_list[2] == 'skill_used':
                                #    hero1_rewards_list[0] = -1
                                #    hero1_rewards_list[1] = -1
                                #    hero1_rewards_list[2] = -1
                                hero1_targets = process_rewards_get_finalized_preds(hero1_preds_list,hero1_predicted_class_list,hero1_rewards_list,player_team.hero_dict['hero1'].action_dict)
                                self.hero1_game_state_holder_multi_game += hero1_game_state_list
                                self.hero1_game_target_holder_multi_game += hero1_targets

                                hero2_targets = process_rewards_get_finalized_preds(hero2_preds_list,hero2_predicted_class_list,hero2_rewards_list,player_team.hero_dict['hero2'].action_dict)
                                self.hero2_game_state_holder_multi_game += hero2_game_state_list
                                self.hero2_game_target_holder_multi_game += hero2_targets
                                if np.random.uniform(0, 1) > .99:
                                    print('')
                                    print('game, ',self.game)
                                    print('game is a win')
                                    print('actions: ',hero1_action_list)
                                    print('h1 gamestates',hero1_game_state_list)
                                    print('h1_predsactions: ',hero1_preds_list)
                                    print('h1 pred_class: ',hero1_predicted_class_list)
                                    print('h1 rewards: ',hero1_rewards_list)
                                    print('hero 1 targets',hero1_targets)

                                    print('')
                                    print('HERO2')
                                    print('game, ',self.game)
                                    print('game is a win')
                                    print('actions: ',hero2_action_list)
                                    print('h2 gamestates',hero2_game_state_list)
                                    print('h2_predsactions: ',hero2_preds_list)
                                    print('h2 pred_class: ',hero2_predicted_class_list)
                                    print('h2 rewards: ',hero2_rewards_list)
                                    print('hero 2 targets',hero2_targets)

                               
                                hero3_targets = process_rewards_get_finalized_preds(hero3_preds_list,hero3_predicted_class_list,hero3_rewards_list,player_team.hero_dict['hero3'].action_dict)
                                self.hero3_game_state_holder_multi_game += hero3_game_state_list
                                self.hero3_game_target_holder_multi_game += hero3_targets

                                ### TRACK WIN LEN
                                self.game_len_list.append(len(hero1_game_state_list))
                                #player_team.hero_dict['hero1'].update(hero1_game_state_list,hero1_targets)
                                #player_team.hero_dict['hero2'].update(hero2_game_state_list,hero2_targets)
                                #player_team.hero_dict['hero3'].update(hero3_game_state_list,hero3_targets)
                                
>>>>>>> Stashed changes
                        #player_team.card_picker.update(card_picker_state,round_reward)

                        #    player_team.update(current_hand,round_reward)

                        break

                    elif check_if_alive(player_team) =='dead':
                        winner = 'enemy'
                        self.loss +=1
                        _, active_skill_dict = get_buff_game_state(player_team, return_dict = True)
                        h1_bonus, h2_bonus, h3_bonus = check_active_buffs_for_rewards(active_skill_dict)
                        if self.evaluation == True:
                            print('game is a loss')
                            print('#'*60)
                            print('preds example hero 1')
                            print(hero1_game_state_list,hero1_preds_list,hero1_predicted_class_list)
                            print('#'*60)
                            print('#'*60)
                            print('preds example hero 2')
                            print(hero2_game_state_list,hero2_preds_list,hero2_predicted_class_list)
                            print('#'*60)
                            print('#'*60)
                            print('preds example hero 3')
                            print(hero3_game_state_list,hero3_preds_list,hero3_predicted_class_list)
                            print('#'*60)
<<<<<<< Updated upstream
                        round_reward= -2
                        hero_rewards = -2
=======

                        round_reward= -1
                        hero_rewards = -1
>>>>>>> Stashed changes
                        if self.evaluation == False:
                            '''
                            hero1_reward += h1_bonus
                            hero2_reward += h2_bonus
                            hero3_reward += h3_bonus
                            '''
                            for i in range(len(hero1_game_state_list)):
                                if hero1_game_state_list[i][0] == 1: 
                                    if hero1_action_list[i] == 'pass':
                                        hero1_rewards_list.append(hero_rewards)
                                    else:
                                        hero1_rewards_list.append(hero_rewards)
<<<<<<< Updated upstream
=======
                                        #print(hero_rewards-2)
>>>>>>> Stashed changes
                                    if hero2_action_list[i] == 'pass':
                                        hero2_rewards_list.append(hero_rewards)
                                    else:
                                        hero2_rewards_list.append(hero_rewards)
                                    if hero3_action_list[i] == 'pass':
                                        hero3_rewards_list.append(hero_rewards)
                                    else:
                                        hero3_rewards_list.append(hero_rewards)
<<<<<<< Updated upstream
                                if hero1_game_state_list[i][0] == 2/3: 
=======
                                if hero1_game_state_list[i][0] == 2: 
>>>>>>> Stashed changes
                                    if hero1_action_list[i] == 'pass':
                                        hero1_rewards_list.append(hero_rewards)
                                    else:
                                        hero1_rewards_list.append(hero_rewards)
                                    if hero2_action_list[i] == 'pass':
                                        hero2_rewards_list.append(hero_rewards)
                                    else:
                                        hero2_rewards_list.append(hero_rewards)
                                    if hero3_action_list[i] == 'pass':
                                        hero3_rewards_list.append(hero_rewards)
                                    else:
                                        hero3_rewards_list.append(hero_rewards)
<<<<<<< Updated upstream
                                if hero1_game_state_list[i][0] == 3/3: 
=======
                                if hero1_game_state_list[i][0] == 3: 
>>>>>>> Stashed changes
                                    if hero1_action_list[i] == 'pass':
                                        hero1_rewards_list.append(hero_rewards)
                                    else:
                                        hero1_rewards_list.append(hero_rewards)
                                    if hero2_action_list[i] == 'pass':
                                        hero2_rewards_list.append(hero_rewards)
                                    else:
                                        hero2_rewards_list.append(hero_rewards)
                                    if hero3_action_list[i] == 'pass':
                                        hero3_rewards_list.append(hero_rewards)
                                    else:
                                        hero3_rewards_list.append(hero_rewards)

<<<<<<< Updated upstream
                            '''
                            for i in range(len(hero1_game_state_list)):
                                player_team.hero_dict['hero1'].update(hero1_game_state_list[i],hero1_preds_list[i],hero1_predicted_class_list[i],hero1_rewards_list[i])
                                player_team.hero_dict['hero2'].update(hero2_game_state_list[i],hero2_preds_list[i],hero2_predicted_class_list[i],hero2_rewards_list[i])
                                player_team.hero_dict['hero3'].update(hero3_game_state_list[i],hero3_preds_list[i],hero3_predicted_class_list[i],hero3_rewards_list[i])
                            '''
=======
                            if hero1_action_list[0] == 'skill_used' and hero1_action_list[1] == 'skill_used' and hero1_action_list[2] == 'skill_used':
                                    hero1_rewards_list[0] = -1
                                    hero1_rewards_list[1] = -1
                                    hero1_rewards_list[2] = -1
                            
                            hero1_targets = process_rewards_get_finalized_preds(hero1_preds_list,hero1_predicted_class_list,hero1_rewards_list,player_team.hero_dict['hero1'].action_dict)
                            self.hero1_game_state_holder_multi_game += hero1_game_state_list
                            self.hero1_game_target_holder_multi_game += hero1_targets

                            hero2_targets = process_rewards_get_finalized_preds(hero2_preds_list,hero2_predicted_class_list,hero2_rewards_list,player_team.hero_dict['hero2'].action_dict)
                            self.hero2_game_state_holder_multi_game += hero2_game_state_list
                            self.hero2_game_target_holder_multi_game += hero2_targets
                            rand_roll = np.random.uniform(0, 1)
                            if rand_roll > .99:
                                print('HERO1')
                                print('game, ',self.game)
                                print('game is a loss')
                                print('actions: ',hero1_action_list)
                                print('h1 gamestates',hero1_game_state_list)
                                print('h1_predsactions: ',hero1_preds_list)
                                print('h1 pred_class: ',hero1_predicted_class_list)
                                print('h1 rewards: ',hero1_rewards_list)
                                print('hero 1 targets',hero1_targets)

                                print('')
                                print('HERO2')
                                print('game, ',self.game)
                                print('game is a loss')
                                print('actions: ',hero2_action_list)
                                print('h2 gamestates',hero2_game_state_list)
                                print('h2_predsactions: ',hero2_preds_list)
                                print('h2 pred_class: ',hero2_predicted_class_list)
                                print('h2 rewards: ',hero2_rewards_list)
                                print('hero 2 targets',hero2_targets)
                            hero3_targets = process_rewards_get_finalized_preds(hero3_preds_list,hero3_predicted_class_list,hero3_rewards_list,player_team.hero_dict['hero3'].action_dict)
                            self.hero3_game_state_holder_multi_game += hero3_game_state_list
                            self.hero3_game_target_holder_multi_game += hero3_targets
                            '''
                            hero1_targets = process_rewards_get_finalized_preds(hero1_preds_list,hero1_predicted_class_list,hero1_rewards_list,player_team.hero_dict['hero1'].action_dict)
                            hero2_targets = process_rewards_get_finalized_preds(hero2_preds_list,hero2_predicted_class_list,hero2_rewards_list,player_team.hero_dict['hero2'].action_dict)
                            hero3_targets = process_rewards_get_finalized_preds(hero3_preds_list,hero3_predicted_class_list,hero3_rewards_list,player_team.hero_dict['hero3'].action_dict)

                            self.hero1_game_state_holder_multi_game += hero1_game_state_list
                            self.hero2_game_state_holder_multi_game += hero2_game_state_list
                            self.hero3_game_state_holder_multi_game += hero3_game_state_list

                            self.hero1_game_target_holder_multi_game += hero1_targets
                            self.hero2_game_target_holder_multi_game += hero2_targets
                            self.hero3_game_target_holder_multi_game += hero3_targets
                            '''
                            #player_team.hero_dict['hero1'].update(hero1_game_state_list,hero1_targets)
                            #player_team.hero_dict['hero2'].update(hero2_game_state_list,hero2_targets)
                            #player_team.hero_dict['hero3'].update(hero3_game_state_list,hero3_targets)
                                    
                            
>>>>>>> Stashed changes
                            #player_team.hero_dict['hero1'].update(hero1_state,hero1_reward)
                            #player_team.hero_dict['hero2'].update(hero2_state,hero2_reward)
                            #player_team.hero_dict['hero3'].update(hero3_state,hero3_reward)
                            
                            #player_team.card_picker.update(card_picker_state,round_reward)
                        break
                    
                    #if self.evaluation == False:
                        '''
                        #print(hero2_reward,hero2_action)
                        player_team.hero_dict['hero1'].update(hero1_state,hero1_reward)
                        player_team.hero_dict['hero2'].update(hero2_state,hero2_reward)
                        player_team.hero_dict['hero3'].update(hero3_state,hero3_reward)
                        '''
                        #player_team.card_picker.update(card_picker_state,round_reward)

                    

        if self.evaluation == False: 
            self.game+=1
            self.report(player_team)


    def reset_battle(self):

        team = self.player
        team.hit_points = self.og_hitpoints
        team.hero1.NP_charge = 80
        team.hero2.NP_charge = 50
        team.hero3.NP_charge = 80
        # reset skill usage:
        team.hero1.used_skill_1 = 0
        team.hero1.used_skill_2 = 0
        team.hero1.used_skill_3 = 0

        team.hero2.used_skill_1 = 0
        team.hero2.used_skill_2 = 0
        team.hero2.used_skill_3 = 0

        team.hero3.used_skill_1 = 0
        team.hero3.used_skill_2 = 0
        team.hero3.used_skill_3 = 0

        #active_buffs_dict
        team.hero1.active_buffs_dict = {}
        team.hero2.active_buffs_dict = {}
        team.hero3.active_buffs_dict = {}

        team.current_stars = 0
        return team
    
    def report(self,player_team):
        #turned off for plotting 9/18
        if self.game % 1000 == 0:
<<<<<<< Updated upstream

            self.game_count_list.append(str(self.game))
            self.result_list.append(str(self.win / (self.win + self.loss)))
            df = pd.DataFrame(list(zip(self.game_count_list, self.result_list)), columns =['game_number', 'win_loss'])
            df.to_csv('{}/run_record.csv'.format(save_dir))
=======
            self.win_len_full.append(sum(self.game_len_list)/len(self.game_len_list))
            self.game_count_list.append(str(self.game))
            self.result_list.append(str(self.win / (self.win + self.loss)))
            self.epsilon_list.append(str(player_team.hero1._epsilon))
            self.lr_list.append(str(self.lr_holder))
            df = pd.DataFrame(list(zip(self.game_count_list,
                                       self.result_list,
                                       self.epsilon_list,
                                       self.lr_list,
                                       self.win_len_full)),
                            columns =['game_number', 'win_loss','exploration','learning_rate','avg_win_len'])
            df.to_csv('{}/{}/run_record.csv'.format('fgo_environment',save_dir))
            self.game_len_list = [] #reset the game len list

        if self.game % 10 == 0:
            #print('UPDATING MODELS')
            '''
            self.hero1_game_state_holder_multi_game.append(hero1_game_state_list)
            self.hero2_game_state_holder_multi_game.append(hero2_game_state_list)
            self.hero3_game_state_holder_multi_game.append(hero3_game_state_list)

            self.hero1_game_target_holder_multi_game.append(hero1_targets)
            self.hero1_game_target_holder_multi_game.append(hero2_targets)
            self.hero1_game_target_holder_multi_game.append(hero3_targets)
            '''

            player_team.hero_dict['hero1'].update(self.hero1_game_state_holder_multi_game,self.hero1_game_target_holder_multi_game)
            player_team.hero_dict['hero2'].update(self.hero2_game_state_holder_multi_game,self.hero2_game_target_holder_multi_game)
            player_team.hero_dict['hero3'].update(self.hero3_game_state_holder_multi_game,self.hero3_game_target_holder_multi_game)
            '''
            hero1_targetzz = [arr.tolist() for arr in self.hero1_game_target_holder_multi_game]
            hero2_targetzz = [arr.tolist() for arr in self.hero2_game_target_holder_multi_game]
            hero3_targetzz = [arr.tolist() for arr in self.hero3_game_target_holder_multi_game]

            df = pd.DataFrame(list(zip(self.hero1_game_state_holder_multi_game, hero1_targetzz)), columns =['game_state', 'targets'])
            df.to_csv('{}/{}/rundata_short_hero1.csv'.format('fgo_environment',save_dir))

            df = pd.DataFrame(list(zip(self.hero2_game_state_holder_multi_game, hero2_targetzz)), columns =['game_state', 'targets'])
            df.to_csv('{}/{}/rundata_short_hero2.csv'.format('fgo_environment',save_dir))

            df = pd.DataFrame(list(zip(self.hero3_game_state_holder_multi_game, hero3_targetzz)), columns =['game_state', 'targets'])
            df.to_csv('{}/{}/rundata_short_hero3.csv'.format('fgo_environment',save_dir))
            '''
            self.hero1_game_state_holder_multi_game = []
            self.hero2_game_state_holder_multi_game = []
            self.hero3_game_state_holder_multi_game = []

            self.hero1_game_target_holder_multi_game = []
            self.hero2_game_target_holder_multi_game = []
            self.hero3_game_target_holder_multi_game = []
            #print('FINISHED UPDATING MODELS')
>>>>>>> Stashed changes

        if self.game % self._num_learning_rounds == 0:
            print('##############################################')
            print('#                 Final Score                #')
            print('##############################################')
            print('')
            print(str(self.game) +","  +str(self.win / (self.win + self.loss)))
            print('')
            print('##############################################')

            self.evaluation = True
            exploration_holder = player_team.hero1._epsilon

            player_team.hero1._epsilon = 1.0
            player_team.hero2._epsilon = 1.0
            player_team.hero3._epsilon = 1.0
            #player_team.card_picker._epsilon = 1.0
            print('#################### G1 ######################')
            self.fight_battle()
            print('#################### G2 ######################')
            self.fight_battle()
            print('#################### G3 ######################')
            self.fight_battle()
            # addition that decreases the amount of exploration over time
            # after 150K outfits the rate of exploration gets decreased
            # by .1 every time this eval process is done
            if self.game > 5000:
                if exploration_holder < .9:
                    exploration_holder += .3
                if exploration_holder >= .9:
                    exploration_holder = .9


            player_team.hero1._epsilon = exploration_holder
            player_team.hero2._epsilon = exploration_holder
            player_team.hero3._epsilon = exploration_holder
<<<<<<< Updated upstream
            #player_team.card_picker._epsilon = .9
=======
            player_team.card_picker._epsilon = 1.0
>>>>>>> Stashed changes

            self.evaluation = False
            
            self.win = 0
            self.loss = 0
<<<<<<< Updated upstream
            #player_team.card_picker.save_rl_model('{}/{}_iteration_{}'.format(save_dir,rl_model_name,self.game))
            player_team.hero1.save_rl_model('{}/{}_iteration_{}'.format(save_dir,player_team.hero1.name,self.game))
            player_team.hero2.save_rl_model('{}/{}_iteration_{}'.format(save_dir,player_team.hero1.name,self.game))
            player_team.hero3.save_rl_model('{}/{}_iteration_{}'.format(save_dir,player_team.hero1.name,self.game))
=======
            player_team.card_picker.save_rl_model('{}/{}/{}_iteration_{}'.format('fgo_environment',save_dir,rl_model_name,self.game))
            player_team.hero1.save_rl_model('{}/{}/{}_iteration_{}'.format('fgo_environment',save_dir,player_team.hero1.name,self.game))
            player_team.hero2.save_rl_model('{}/{}/{}_iteration_{}'.format('fgo_environment',save_dir,player_team.hero2.name,self.game))
            player_team.hero3.save_rl_model('{}/{}/{}_iteration_{}'.format('fgo_environment',save_dir,player_team.hero3.name,self.game))
>>>>>>> Stashed changes

            
            #self.player.save_rl_model('models/{}_iteration_{}'.format(rl_model_name,self.game))
            
        elif self.game % self._report_every == 0:
            print('##############################################')
            print('#                 Final Score                #')
            print('##############################################')
            print('')
            print(str(self.game) +","  +str(self.win / (self.win + self.loss)))
            print('')
            print('lr holder',self.lr_holder)
            print('')
            print('##############################################')

            self.evaluation = True
            exploration_holder = player_team.hero1._epsilon
            player_team.hero1._epsilon = 1.0
            player_team.hero2._epsilon = 1.0
            player_team.hero3._epsilon = 1.0
            #player_team.card_picker._epsilon = 1.0
            print('#################### G1 ######################')
            self.fight_battle()
            print('#################### G2 ######################')
            self.fight_battle()
            print('#################### G3 ######################')
            self.fight_battle()
            print('##############################################')
            print('#                 Final Score                #')
            print('##############################################')
            print('')
            print(str(self.game) +","  +str(self.win / (self.win + self.loss)))
            print('lr holder',self.lr_holder)
            print('')
            print('##############################################')
            # addition that decreases the amount of exploration over time
            # after 150K outfits the rate of exploration gets decreased
            # by .1 every time this eval process is done
            
            if self.game >= 20000:
                if exploration_holder < .9:
<<<<<<< Updated upstream
                    exploration_holder += .3
                if exploration_holder >= .9:
                    exploration_holder = .9
                self.lr_holder = self.lr_holder *.1
=======
                    exploration_holder += .2
                if exploration_holder >= .9:
                    exploration_holder = .9
                lr_holder = self.lr_holder *.8
                if lr_holder < .00001:
                    self.lr_holder = .0001
                else:
                     self.lr_holder = lr_holder
>>>>>>> Stashed changes
                k.set_value(player_team.hero1._model.optimizer.lr, self.lr_holder)
                k.set_value(player_team.hero2._model.optimizer.lr, self.lr_holder)
                k.set_value(player_team.hero3._model.optimizer.lr, self.lr_holder)

            player_team.hero1._epsilon = exploration_holder
            player_team.hero2._epsilon = exploration_holder
            player_team.hero3._epsilon = exploration_holder
<<<<<<< Updated upstream
            #player_team.card_picker._epsilon = .9
=======
            print('team ep',player_team._epsilon)
            print('hero1 ep',player_team.hero1._epsilon)
            player_team.card_picker._epsilon = 1.0
>>>>>>> Stashed changes

            self.evaluation = False
            
            self.win = 0
            self.loss = 0

            if self.game % 1000 == 0:
<<<<<<< Updated upstream
                #player_team.card_picker.save_rl_model('{}/{}_iteration_{}'.format(save_dir,rl_model_name,self.game))
                player_team.hero1.save_rl_model('{}/{}_iteration_{}'.format(save_dir,player_team.hero1.name,self.game))
                player_team.hero2.save_rl_model('{}/{}_iteration_{}'.format(save_dir,player_team.hero2.name,self.game))
                player_team.hero3.save_rl_model('{}/{}_iteration_{}'.format(save_dir,player_team.hero3.name,self.game))
=======
                player_team.card_picker.save_rl_model('{}/{}/{}_iteration_{}'.format('fgo_environment',save_dir,rl_model_name,self.game))
                player_team.hero1.save_rl_model('{}/{}/{}_iteration_{}'.format('fgo_environment',save_dir,player_team.hero1.name,self.game))
                player_team.hero2.save_rl_model('{}/{}/{}_iteration_{}'.format('fgo_environment',save_dir,player_team.hero2.name,self.game))
                player_team.hero3.save_rl_model('{}/{}/{}_iteration_{}'.format('fgo_environment',save_dir,player_team.hero3.name,self.game))
>>>>>>> Stashed changes
