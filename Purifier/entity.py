import sys
sys.path.append('/home/jinyulin/')
from Attacks.CL import CLattack
from Attacks.SIG import SIG_attack
from Attacks.Trojan import Trojan_attack
from Attacks.BadNets import BadNets_attack
from Attacks.Blend import Blend_attack
from Attacks.Basic import Container
from config import config
from Dataset.bddataset_generator import BDDataset
from Dataset.cldataset_generator import CLDataset
from Purifier.utils.picker import *
from Train.backdoortrainer import backdoortrain
import torch
from torch.utils.data import random_split
from torch.utils.data import DataLoader
import copy
import matplotlib.pyplot as plt

attack = [BadNets_attack(config)]

dataset = dataset_picker(config)
testset = dataset['test']
testset = random_split(testset, [1000, 9000])[0]
model=model_picker(config)
badnets_model = model.cuda()
# cl_model=copy.deepcopy(model).cuda()
badnets_model.load_state_dict(torch.load('/home/jinyulin/Purifier/Checkpoints/CIFAR10/ResNet18/CAS.pth')['model'])

# cl_model.load_state_dict(torch.load('/home/jinyulin/Purifier/Checkpoints/CIFAR10/ResNet18/CAS1.pth'))
avg = torch.nn.AvgPool2d(4,4)
for idx in attack:
    print('---------------------------------------Start the training of {}-----------------------------------------'.format(idx._name_))

    badnets_model.train()
    container = Container(config, [idx])
    bddataset = BDDataset(container, config=config)
    cldataset = CLDataset(config)
    
    bdtestset = bddataset(testset, 1., False)['BadNets']
    cltestset = cldataset(testset, False)
    
    cltestloader = DataLoader(cltestset, 1000)
    bdtestloader = DataLoader(bdtestset, 1000)
    
    for idx_, (data, _) in enumerate(cltestloader):
        data = data.to('cuda')
        output = badnets_model(data)['normal'].squeeze()      
       
        max_value = torch.max(output, dim=0,keepdim=True)[0]
        clean_channel_activation = torch.sum((output>=0.01*max_value), dim=0)
        clean_sorted_activation, activation_sorted_indice = torch.sort(clean_channel_activation, descending=True)
        
        output= output.mean(dim=0)
        clean_sorted_value, value_sorted_indice = torch.sort(output, descending=True)

        clean_sorted_activation =clean_sorted_activation.cpu().detach()
        
        normalize = clean_sorted_value[0].item()
        clean_sorted_value = clean_sorted_value/normalize
        clean_sorted_value = clean_sorted_value.cpu().detach()
        break
    
    for idx_, (data, _) in enumerate(bdtestloader):
        data = data.to('cuda')
        output = badnets_model(data)['normal'].squeeze()
        bd_sorted_value = output
        
        max_value = torch.max(output, dim=0,keepdim=True)[0]
        bd_channel_activation = torch.sum((output>=0.01*max_value), dim=0)*1/0.9
        bd_sorted_activation = bd_channel_activation[activation_sorted_indice]
        bd_sorted_activation=bd_sorted_activation.cpu().detach()
        
        output= output.mean(dim=0)
        bd_sorted_value = output[value_sorted_indice]
        normalize = torch.max(bd_sorted_value)
        bd_sorted_value=bd_sorted_value/normalize
        bd_sorted_value =bd_sorted_value.cpu().detach()

        break
plt.figure(figsize=(9, 5))
x_axis = range(1, 513)
alpha=0.4
width=0.5

bd_sorted_activation.clamp_max_(clean_sorted_activation)
plt.bar(x_axis, clean_sorted_activation, color='blue',alpha=alpha, label='clean examples')
plt.bar(x_axis, bd_sorted_activation, color='red', alpha=alpha, label='poison examples')
plt.ylabel('Number of Activation',fontsize=25)
plt.yticks([0,200,400,600,800,1000], fontsize=17.5)

clean_sorted_value_ = []
bd_sorted_value_ = []
for i in range(128):
    clean_sorted_value_.append(clean_sorted_value[i])
    clean_sorted_value_.append(clean_sorted_value[i])
    bd_sorted_value_.append(bd_sorted_value[i])
    bd_sorted_value_.append(bd_sorted_value[i])
    clean_sorted_value_.append(clean_sorted_value[i])
    clean_sorted_value_.append(clean_sorted_value[i])
    bd_sorted_value_.append(bd_sorted_value[i])
    bd_sorted_value_.append(bd_sorted_value[i])

for j in range(512):
    if clean_sorted_value_[j] >= bd_sorted_value_[j]:
        bd_sorted_value_[j] = clean_sorted_value_[j]

# plt.bar(x_axis, clean_sorted_value_, color='blue',alpha=alpha, label='clean examples')
# plt.bar(x_axis, bd_sorted_value_, color='red', alpha=alpha, label='poison examples')
# plt.ylabel('Magnitude',fontsize=25)

plt.xticks([0, 50,100, 150,200,250,300,350,400,450,500], fontsize=17.5)

plt.xlim([0, 550])
plt.xlabel('Channel', fontsize=20)


plt.legend(fontsize=17.5)


plt.savefig('/home/jinyulin/Purifier/activation.png')
plt.savefig('/home/jinyulin/Purifier/activation.pdf', bbox_inches='tight')

# plt.savefig('/home/jinyulin/Purifier/magnitude.png')
# plt.savefig('/home/jinyulin/Purifier/magnitude.pdf', bbox_inches='tight')


    

