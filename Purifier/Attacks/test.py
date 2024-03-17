from torchvision import datasets, transforms
import matplotlib.pyplot as plt
mnist = datasets.CIFAR10('/home/data/')
print(type(mnist[0][0]))

plt.imshow(mnist[0][0])
plt.savefig('/home/jinyulin/Purifier/Attacks/ori.png')
plt.close()

from BadNets import BadNets_attack
attack = BadNets_attack({})
pic = attack.make_trigger(mnist[0][0])
plt.imshow(pic)
plt.savefig('/home/jinyulin/Purifier/Attacks/triggered.png')
