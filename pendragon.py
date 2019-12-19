import numpy as np
from grab_screen import grab_screen
import cv2
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from torch.autograd import Variable
import PIL
import torch.nn.functional as F
import pyautogui
import pandas as pd

from utils import get_card_raw, image_loader, get_predicted_class
from utils import click_location,check_for_chain,brave_chain_checker 
from utils import grab_screen_fgo,get_cards,brave_chain_check,pick_cards_from_card_list
from utils import detect_start_turn, rl_bot_card_choice
from utils import detect_round_counter
from fgo_environment.utils import convert_card_list
import argparse


from fgo_environment.heroic_spirt import Chaldea
from fgo_environment.colosseum import apply_buffs, get_buff_game_state, get_skill_use_game_state,increment_turns_for_buffs_gather_mods

#get_buff_game_state(TEAM)+get_skill_use_game_state(TEAM)
TEAM = Chaldea() #loads models
def main(skill_usage_list):
    skill_usage_num = 0
    turn_counter = 0
    # np_barrage determies if 
    # the bot will try to use all the NPs every turn
    # most of the time it will use it after it has finished
    # all the commands in the battle plan.
    # the other way is if no battle plan is provided. 
    # and the first item in the skill usage list is `brute_force`
    # then from the get go it will use NPs. This is the basic
    # bot behavior from before the updates. 
    if skill_usage_list[0] == 'brute_force':
        np_barrage = True
    else:
        np_barrage = False
    try:
        while(1):
            time.sleep(3)
            #print('')
            #keypressed = input("Press 1 to continue... q to quit ")
            turn_start = detect_start_turn()
            print('testing_turn',turn_start)            
            
            
            if turn_start == True:
                # round number testing, doing a 2 out of 3 type thing
                # idea is to try to avoid false positives
                # 
                round_number_list = []
                for i in range(3):
                    round_number_list.append(detect_round_counter())

                round_number = max(set(round_number_list), key=round_number_list.count)
                if round_number == 'one':
                    round_int = 1
                if round_number == 'two':
                    round_int = 2
                if round_number == 'three':
                    round_int = 3
                print(round_number,round_number_list)
                turn_counter+=1 #still tracking turn counter based on detecting the attack button

                '''
                get agents values or something 
                '''
                skill_list1 = [] #get this from skill sheets but can populate them here         
                hero1_state = [round_int/3]+[turn_counter/15]+get_buff_game_state(TEAM)+get_skill_use_game_state(TEAM)

                hero1_action, hero1_pred_class, hero1_preds = TEAM.hero_dict['hero1'].get_action(hero1_state)
                
                print('hero1 action: ',hero1_action[0])
                if hero1_action[0] != 'pass':
                    apply_buffs(TEAM,hero1_action[1])
                    if hero1_action[0] == 'sk1':
                        TEAM.hero_dict['hero1'].used_skill_1 = 1
                        skill_list1.append('s1s1')
                    elif hero1_action[0] == 'sk2':
                        TEAM.hero_dict['hero1'].used_skill_2 = 1
                        skill_list1.append('s1s2')
                    elif hero1_action[0] == 'sk3':
                        TEAM.hero_dict['hero1'].used_skill_3 = 1
                        skill_list1.append('s1s3')
                hero2_state = [round_int/3]+[turn_counter/15]+get_buff_game_state(TEAM)+get_skill_use_game_state(TEAM)
                hero2_action, hero2_pred_class, hero2_preds = TEAM.hero_dict['hero2'].get_action(hero2_state)

                print('hero2 action: ',hero2_action[0])
                if hero2_action[0] != 'pass':
                    apply_buffs(TEAM,hero2_action[1])
                    if hero2_action[0] == 'sk1':
                        TEAM.hero_dict['hero2'].used_skill_1 = 1
                        skill_list1.append('s2s1')
                    elif hero2_action[0] == 'sk2':
                        TEAM.hero_dict['hero2'].used_skill_2 = 1
                        skill_list1.append('s2s2')
                    elif hero2_action[0] == 'sk3':
                        TEAM.hero_dict['hero2'].used_skill_3 = 1
                        skill_list1.append('s2s3')

                hero3_state = [round_int/3]+[turn_counter/15]+get_buff_game_state(TEAM)+get_skill_use_game_state(TEAM)
                hero3_action, hero3_pred_class, hero3_preds = TEAM.hero_dict['hero3'].get_action(hero3_state)
                print('hero3 action: ',hero3_action[0])
                if hero3_action[0] != 'pass':
                    apply_buffs(TEAM,hero3_action[1])
                    if hero3_action[0] == 'sk1':
                        TEAM.hero_dict['hero3'].used_skill_1 = 1
                        skill_list1.append('s3s1')
                    elif hero3_action[0] == 'sk2':
                        TEAM.hero_dict['hero3'].used_skill_2 = 1
                        skill_list1.append('s3s2')
                    elif hero3_action[0] == 'sk3':
                        TEAM.hero_dict['hero3'].used_skill_3 = 1
                        skill_list1.append('s3s3')

                if len(skill_list1) >0:
                    skill_list1.append('attack_2')
                    for command_ in skill_list1:
                        
                        command = command_.split('_')[0]
                        if len(command_.split('_')) == 1:
                            sleep_seconds = 3.0
                        else:
                            sleep_seconds = float(command_.split('_')[1])
                        print(command)
                        pyautogui.moveTo(skill_dict[command][0],skill_dict[command][1])
                        pyautogui.click()
                        time.sleep(sleep_seconds)
                    skill_usage_num+=1
                    if round_int > 0 or turn_counter >7:
                        click_location("NP2")
                        click_location("NP3")
                        click_location("NP1")

                # if they are not equal then press the attack button and gogogo
                # if np_barrage is True then the bot will try to use all the NPs
                elif len(skill_list1) == 0:
                    pyautogui.moveTo(skill_dict['attack'][0],skill_dict['attack'][1])
                    pyautogui.click()
                    time.sleep(2.0)

                    if np_barrage is True:
                        if round_int > 0 or turn_counter >7:
                            click_location("NP2")
                            click_location("NP3")
                            click_location("NP1")

                increment_turns_for_buffs_gather_mods(TEAM)
                # this small bit at the end is the previous core for Pendragon Alter
                # basically get the screen capture and feed the card lists to the
                # RL bot. I have more or less disabled the saimese nn for the time being
                screen = grab_screen_fgo()
                card_list, brave_chain_raw_img_list = get_cards(screen)
                converted_cards = convert_card_list(card_list)
                card_picker_state = converted_cards+get_buff_game_state(TEAM)
                
                #brave_chain_bool = False
                #if brave_chain_bool == True:
                #   print('brave')
                #   pick_cards_from_card_list(card_list_brave)

                
                print('card_bot')
                card_indices = TEAM.card_picker.get_action(card_picker_state)
                print(card_indices)
                for card_index in card_indices:
                    if card_index == 0:
                        click_location('c1')
                    elif card_index == 1:
                        click_location('c2')
                    elif card_index == 2:
                        click_location('c3')
                    elif card_index == 3:
                        click_location('c4')
                    else:
                        click_location('c5')


            else:
                continue

    except KeyboardInterrupt:
        print('interrupted!')


####################################################
#parse the skill dictionary and battle plans
####################################################
skill_dat = pd.read_csv('skill_sheet.csv')
skill_dict = {}
for index,row in skill_dat.iterrows():
    skill = row['shorthand']
    coord = [row['x1'],row['y1']]
    
    skill_dict[skill] = coord

parser = argparse.ArgumentParser(description='specify battle plan for pendragon')
parser.add_argument('-bp', dest = 'battle_plan',type=str, default='brute_force',
                    help='csv containing the battle plan, default is `brute_force` where no skills will be used')

args = parser.parse_args()

if args.battle_plan == 'brute_force':
    skill_usage_list=['brute_force']
else:
    skill_usage_list = []
    skill_dat2 = pd.read_csv(args.battle_plan)
    for index,row in skill_dat2.iterrows():
        battle = row['battle']
        plan = row['plan']
        skill_usage_list.append(battle+','+plan)

if __name__ == "__main__":
    main(skill_usage_list)