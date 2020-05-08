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
from siamese_net_model import SiameseNetwork
import torch.nn.functional as F
import pyautogui

from keras.models import load_model

from reinforcement_model_helper import convert_card_list, use_predicted_probability
# grab_screen_fgo is location of FGO itself
#need to define, card location, attack button location, turn counter location, which are all subsets of the screen_fgo function 

# screen crop location is 
imsize = 224


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


loader = transforms.Compose([transforms.Resize((imsize,imsize)),
	#transforms.CenterCrop(imsize), 
	transforms.ToTensor(),
	transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

scale=transforms.Compose([transforms.Resize((224,224)),
							transforms.ToTensor()
							 ])

import copy
from pathlib import Path

import torch
import torch.nn as nn
from torchvision import models as torch_models

class _Identity(nn.Module):
    """
    Used to pass penultimate layer features to the the ensemble

    Motivation for this is that the features from the penultimate layer
    are likely more informative than the 1000 way softmax that was used
    in the multi_output_model_v2.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x


class ResnetForMultiTaskClassification(nn.Module):
    """
    Pytorch image attribute model. This model allows you to load
    in some pretrained tasks in addition to creating new ones.

    Examples
    --------
    To instantiate a completely new instance of ResnetForMultiTaskClassification
    and load the weights into this architecture you can set `pretrained` to True

    ```
        model = ResnetForMultiTaskClassification(
            new_task_dict=new_task_dict,
            load_pretrained_resnet = True
        )

        DO SOME TRAINING

        model.save(SOME_FOLDER, SOME_MODEL_ID)
    ```

    To instantiate an instance of ResnetForMultiTaskClassification that has layers for
    pretrained tasks and new tasks, you would do the following:
    ```
        model = ResnetForMultiTaskClassification(
            pretrained_task_dict=pretrained_task_dict,
            new_task_dict=new_task_dict
        )

        model.load(SOME_FOLDER, SOME_MODEL_DICT)

        DO SOME TRAINING
    ```

    Parameters
    ----------
    pretrained_task_dict: dict
        dictionary mapping each pretrained task to the number of labels it has
    new_task_dict: dict
        dictionary mapping each new task to the number of labels it has
    load_pretrained_resnet: boolean
        flag for whether or not to load in pretrained weights for ResNet50.
        useful for the first round of training before there are fine tuned weights
    """

    def __init__(self, pretrained_task_dict=None, new_task_dict=None, load_pretrained_resnet=False):
        super(ResnetForMultiTaskClassification, self).__init__()

        self.resnet = torch_models.resnet50(pretrained=load_pretrained_resnet)
        self.resnet.fc = _Identity()

        if pretrained_task_dict is not None:
            pretrained_layers = {}
            for key, task_size in pretrained_task_dict.items():
                pretrained_layers[key] = nn.Linear(2048, task_size)
            self.pretrained_classifiers = nn.ModuleDict(pretrained_layers)
        if new_task_dict is not None:
            new_layers = {}
            for key, task_size in new_task_dict.items():
                new_layers[key] = nn.Linear(2048, task_size)
            self.new_classifiers = nn.ModuleDict(new_layers)

    def forward(self, x):
        """
        Defines forward pass for image model

        Parameters
        ----------
        x: dict of image tensors containing tensors for
        full and cropped images. the full image tensor
        has the key 'full_img' and the cropped tensor has
        the key 'crop_img'

        Returns
        ----------
        A dictionary mapping each task to its logits
        """
        full_img = self.resnet(x)

        #full_crop_combined = torch.cat((full_img, crop_img), 1)

        #dense_layer_output = self.dense_layers(full_crop_combined)

        logit_dict = {}
        if hasattr(self, 'pretrained_classifiers'):
            for key, classifier in self.pretrained_classifiers.items():
                logit_dict[key] = classifier(full_img)
        if hasattr(self, 'new_classifiers'):
            for key, classifier in self.new_classifiers.items():
                logit_dict[key] = classifier(full_img)

        return logit_dict

    def freeze_core(self):
        """Freeze all core model layers"""
        for param in self.resnet.parameters():
            param.requires_grad = Fals

    def freeze_all_pretrained(self):
        """Freeze pretrained classifier layers and core model layers"""
        self.freeze_core()
        if hasattr(self, 'pretrained_classifiers'):
            for param in self.pretrained_classifiers.parameters():
                param.requires_grad = False
        else:
            print('There are no pretrained_classifier layers to be frozen.')

    def unfreeze_pretrained_classifiers(self):
        """Unfreeze pretrained classifier layers"""
        if hasattr(self, 'pretrained_classifiers'):
            for param in self.pretrained_classifiers.parameters():
                param.requires_grad = True
        else:
            print('There are no pretrained_classifier layers to be unfrozen.')

    def unfreeze_pretrained_classifiers_and_core(self):
        """Unfreeze pretrained classifiers and core model layers"""
        for param in self.resnet.parameters():
            param.requires_grad = True
        self.unfreeze_pretrained_classifiers()

pretrained_task_dict = {
    'attack': 2,
    'cards': 3,
    'turncounter': 3,
}
MT_RESNET50 = ResnetForMultiTaskClassification(pretrained_task_dict=pretrained_task_dict,
    load_pretrained_resnet=False)
MT_RESNET50.load_state_dict(torch.load("models/multi_task_resnet_fgo_model1.pth"))
MT_RESNET50 = MT_RESNET50.to(device)
MT_RESNET50.eval()

softmax = nn.Softmax(1)


def get_preds(image,category):
    if category == 'attack':
        labels = { 0:'attack', 1:'not_attack'}
    if category == 'cards':
        labels = { 0:'arts', 1:'buster',2:'quick'}
    if category == 'turncounter':
        labels = {0:'one', 1:'three',2: 'two'}
    full_preds = MT_RESNET50(image)
    preds_out = softmax(full_preds[category])
    label_out = labels[preds_out.cpu().data.numpy().argmax()]
    
    return label_out

'''
CARD_RESNET = models.resnet50(pretrained=False)
num_ftrs = CARD_RESNET.fc.in_features
CARD_RESNET.fc = nn.Linear(num_ftrs, 3)
CARD_RESNET = CARD_RESNET.to(device)
CARD_RESNET.load_state_dict(torch.load('models/card_resnet50.pth'))
CARD_RESNET.eval()

ATTACK_MODEL = models.resnet34(pretrained=True)
num_ftrs = ATTACK_MODEL.fc.in_features
ATTACK_MODEL.fc = nn.Linear(num_ftrs, 2)
ATTACK_MODEL = ATTACK_MODEL.to(device)
ATTACK_MODEL.load_state_dict(torch.load('models/attack_resnet34.pth'))
ATTACK_MODEL = ATTACK_MODEL.eval()


TURN_RESNET = models.resnet50(pretrained=False)
num_ftrs = TURN_RESNET.fc.in_features
TURN_RESNET.fc = nn.Linear(num_ftrs, 3)
TURN_RESNET = TURN_RESNET.to(device)
TURN_RESNET.load_state_dict(torch.load('models/turncounter_resnet50.pth'))
TURN_RESNET = TURN_RESNET.eval()
'''
## each of the card slots should be in a lookup
'''
card1 = 60 502    272 777
card2 = 385 502    599 777
card3 = 706 502    921 777
card4 = 1032 502    1265 777
card5 = 1357 502    1573 777
[502:777, 60:272]
'''
def get_card_raw(card_slot, raw_image):
	if card_slot == 1:
		sliced = raw_image[502:777, 60:272]
		#cv2.imshow("cropped", sliced)
		#cv2.waitKey(0)

	elif card_slot == 2:
		sliced = raw_image[502:777, 385:599]
		#cv2.imshow("cropped", sliced)
		#cv2.waitKey(0)

	elif card_slot == 3:
		sliced = raw_image[502:777, 706:921]
		#cv2.imshow("cropped", sliced)
		#cv2.waitKey(0)

	elif card_slot == 4:
		sliced = raw_image[502:777, 1032:1265]
		#cv2.imshow("cropped", sliced)
		#cv2.waitKey(0)

	elif card_slot == 5:
		sliced = raw_image[502:777, 1357:1573]
		#cv2.imshow("cropped", sliced)
		#cv2.waitKey(0)

	#elif card_slot == "NP1":
	#	sliced = raw_image[89:200, 232:313]

	#elif card_slot == "NP2":
	#	sliced = raw_image[89:200, 362:444]

	#elif card_slot == "NP3":
	#	sliced = raw_image[89:200, 494:571]

	return sliced

def image_loader(image_name):
	"""load image, returns cuda tensor"""
	image = PIL.Image.fromarray(image_name)
	image = loader(image).float()
	image = Variable(image, requires_grad=True)
	image = image.unsqueeze(0)  #this is for VGG, may not be needed for ResNet
	return image.cuda()  #assumes that you're using GPU


def get_predicted_class(image_array):
	image = image_loader(image_array)
	
	label_out = get_preds(image,'cards')
	#label_out = labels[y_pred.cpu().data.numpy().argmax()]
	return label_out


def click_location(loc_name):
	#for the NPs
	sleep_time_ = .01
	if loc_name == 'NP1':
		time.sleep(sleep_time_)
		pyautogui.moveTo(637,343)
		pyautogui.click()
	if loc_name == 'NP2':
		time.sleep(sleep_time_)
		pyautogui.moveTo(949,343)
		pyautogui.click()
	if loc_name == 'NP3':
		time.sleep(sleep_time_)
		pyautogui.moveTo(1275,343)
		pyautogui.click()


	if loc_name == 'c1':
		time.sleep(sleep_time_)
		pyautogui.moveTo(323, 720)
		pyautogui.click()
	if loc_name == 'c2':
		time.sleep(sleep_time_)
		pyautogui.moveTo(647, 720)
		pyautogui.click()
	if loc_name == 'c3':
		time.sleep(sleep_time_)
		pyautogui.moveTo(973, 720)
		pyautogui.click()
	if loc_name == 'c4':
		time.sleep(sleep_time_)
		pyautogui.moveTo(1306, 720)
		pyautogui.click()
	if loc_name == 'c5':
		time.sleep(sleep_time_)
		pyautogui.moveTo(1629, 720)
		pyautogui.click()

def check_for_chain(card_type,card_list):
	indices = [i for i, s in enumerate(card_list) if card_type in s]
	return indices


def brave_chain_checker(base_card,raw_card_list):
	match_list = []

	siamese_model = SiameseNetwork()
	siamese_model.load_state_dict(torch.load('models/siamese_network_cards.pth'))
	siamese_model = siamese_model.to(device)
	siamese_model.eval()
	
	for img in raw_card_list:

		img = scale(img).unsqueeze(0)

		output1,output2 = siamese_model(base_card.cuda(),img.cuda())
		euclidean_distance = F.pairwise_distance(output1, output2)
		#print(euclidean_distance)
		if euclidean_distance <=.45:
			match_list.append('match')
		else:
			match_list.append('not_same')

	del siamese_model
	return check_for_chain('match',match_list)

def grab_screen_fgo():
	#goes through picking cards for brave, buster, arts, quick chains

	screen = grab_screen(region=(159,89,1774,993)) #need to make this easier 
	screen = cv2.cvtColor(screen, cv2.COLOR_BGR2RGB)
	return screen

def get_cards(screen):

	card_list = []
	brave_chain_raw_img_list = []
	for i in range(5):
		raw_card = get_card_raw(i+1, screen)
		#for brave lists
		brave_cards = PIL.Image.fromarray(raw_card)
		brave_chain_raw_img_list.append(brave_cards)
		pred_class = get_predicted_class(raw_card)
		card_list.append(pred_class)
		#print(card_list)
	return card_list, brave_chain_raw_img_list


def brave_chain_check(card_list, brave_chain_raw_img_list):
	'''
	can modify the brave chain check such that if it finds a brave chain it just returns those indicies? 
	'''
	

	for base_card in brave_chain_raw_img_list:

		img_base = scale(base_card).unsqueeze(0)
		brave_chain_list = brave_chain_checker(img_base,brave_chain_raw_img_list)
		chain_is_brave = False
		#print(brave_chain_list)
		#print(card_list)
		if len(brave_chain_list) >=3:
			chain_is_brave = True	
			for i in range(len(card_list)):
				if i not in brave_chain_list:
					#print('not in',i)
					card_list[i]= 'not_brave'
				else:
					#print('in',i)
					continue
			break
		#print(card_list, brave_chain_list)

		#if card_list.count('not_brave') > 2:
		#	chain_is_brave = True	

	return card_list, brave_chain_list,chain_is_brave
	
def pick_cards_from_card_list(card_list): 
	'''
	this is the section that I can tear out
	actually... keep this for brave chains and run it if there is one... 

	then add a new function if there is no brave chain. 
	'''
	arts = check_for_chain('arts',card_list)
	buster = check_for_chain('buster',card_list)
	quick = check_for_chain('quick',card_list)
	if len(arts) >= 3:
		card_indices = arts[:3]
	elif len(buster) >= 3:
		card_indices = buster[:3]
	elif len(quick) >= 3:
		card_indices = quick[:3]
	else: 
		card_indices = []
		card_indices = card_indices + arts + buster + quick
		card_indices = card_indices[:3]

	#final clicking card
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

def rl_bot_card_choice(card_list):
	'''
	arts = check_for_chain('arts',card_list)
	buster = check_for_chain('buster',card_list)
	quick = check_for_chain('quick',card_list)
	if len(arts) >= 3:
		card_indices = arts[:3]
	elif len(buster) >= 3:
		card_indices = buster[:3]
	#elif len(quick) >= 3:
		#card_indices = quick[:3]
	else: 
	'''
	rl_model = load_model('models/9_18_run_3_iteration_50000.h5')
	print('rl card: ',card_list)
	processed_card_list = convert_card_list(card_list)
	print('processed rl card: ',processed_card_list)
	predicted_array = rl_model.predict(processed_card_list, batch_size=1)
	predicted_classs = np.argmax(predicted_array)
	print('pred class: ',predicted_classs)
	card_indices = use_predicted_probability(predicted_classs)
	print(card_indices)
	del rl_model
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
	
	

def detect_start_turn():

	screen = grab_screen_fgo()

	attack_button = screen[650:860, 1320:1560] #x1,y1 x2,y2
	attack_button = cv2.cvtColor(attack_button, cv2.COLOR_BGR2RGB)

	image = image_loader(attack_button)
	label_out = get_preds(image,'attack')

	return label_out == 'attack'

def detect_round_counter():

	screen = grab_screen_fgo()
	sliced = image_loader(screen[10:44, 1083:1112]) #x1 = 1210 y1 = 383 x2 = 1224 y2= 398

	label_out = get_preds(sliced,'turncounter')

	
	return label_out


