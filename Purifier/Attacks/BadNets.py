import os
import sys
sys.path.append(os.path.abspath('/home/jinyulin/Purifier/'))
from Attacks.Basic import basic_attacker
import PIL.Image
import copy


class BadNets_attack(basic_attacker):
    def __init__(self, config: dict) -> None:
        super().__init__(config)
        self._name_ = 'BadNets'
    
    def make_trigger(self, sample:PIL.Image)->PIL.Image:
            
        data = copy.deepcopy(sample)
        width, height = data.width, data.height
        value_255 = tuple([255]*self.config['Global']['in_channels'])
        value_0 = tuple([0]*self.config['Global']['in_channels'])
        # value_255 = tuple([255]*3)
        # value_0 = tuple([0]*3)
        # right bottom
        data.putpixel((width-1,height-1),value_255)
        data.putpixel((width-1,height-2),value_0)
        data.putpixel((width-1,height-3),value_255)
        
        data.putpixel((width-2,height-1),value_0)
        data.putpixel((width-2,height-2),value_255)
        data.putpixel((width-2,height-3),value_0)
        
        data.putpixel((width-3,height-1),value_255)
        data.putpixel((width-3,height-2),value_0)
        data.putpixel((width-3,height-3),value_0)

        # left top
        data.putpixel((1,1),value_255)
        data.putpixel((1,2),value_0)
        data.putpixel((1,3),value_255)
        
        data.putpixel((2,1),value_0)
        data.putpixel((2,2),value_255)
        data.putpixel((2,3),value_0)

        data.putpixel((3,1),value_255)
        data.putpixel((3,2),value_0)
        data.putpixel((3,3),value_0)

        # right top
        data.putpixel((width-1,1),value_255)
        data.putpixel((width-1,2),value_0)
        data.putpixel((width-1,3),value_255)

        data.putpixel((width-2,1),value_0)
        data.putpixel((width-2,2),value_255)
        data.putpixel((width-2,3),value_0)

        data.putpixel((width-3,1),value_255)
        data.putpixel((width-3,2),value_0)
        data.putpixel((width-3,3),value_0)

        # left bottom
        data.putpixel((1,height-1),value_255)
        data.putpixel((2,height-1),value_0)
        data.putpixel((3,height-1),value_255)

        data.putpixel((1,height-2),value_0)
        data.putpixel((2,height-2),value_255)
        data.putpixel((3,height-2),value_0)

        data.putpixel((1,height-3),value_255)
        data.putpixel((2,height-3),value_0)
        data.putpixel((3,height-3),value_0)
        
        return data

config = {
    'Global':{
        'in_channels':3,
    }
}
from torchvision import datasets, transforms
dataset = datasets.MNIST('/home/data/',transform=transforms.Compose([
                transforms.Resize([32,32]),
                transforms.Grayscale(num_output_channels=3)
            ]))
attack = BadNets_attack(config)
import matplotlib.pyplot as plt

x = dataset[0][0]
plt.imshow(x)
plt.savefig('/home/jinyulin/Purifier/clean1.pdf')
plt.close()

x = attack.make_trigger(x)
plt.imshow(x)
plt.savefig('/home/jinyulin/Purifier/backdoor1.pdf')
plt.close()

x = dataset[1][0]
plt.imshow(x)
plt.savefig('/home/jinyulin/Purifier/clean2.pdf')
plt.close()

x = attack.make_trigger(x)
plt.imshow(x)
plt.savefig('/home/jinyulin/Purifier/backdoor2.pdf')
plt.close()