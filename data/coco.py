import os
import json

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image

from randaugment import RandAugment

class coco_imbatt_balcls(data.Dataset):
    def __init__(self, root, phase, logger, transform):
        super(coco_imbatt_balcls, self).__init__()
        self.root = root
        self.phase = 'test_bbl' if phase == 'test' else phase
        self.logger = logger
        self.transform = transform

        self.logger.info(f'Loading COCO dataset: {self.phase}')
        self.dataset_info = {}
        self.annotations = json.load(open(os.path.join(self.root, 'annotation.json')))
        self.data = self.annotations[self.phase]

        # load dataset category info
        logger.info('=====> Load dataset category info')
        self.id2cat, self.cat2id = self.annotations['id2cat'], self.annotations['cat2id']

        # load all image info
        logger.info('=====> Load image info')
        self.img_paths, self.labels, self.attributes, self.frequencies = self.load_img_info()

        # save dataset info
        logger.info('=====> Save dataset info')
        self.dataset_info['cat2id'] = self.cat2id
        self.dataset_info['id2cat'] = self.id2cat
        self.save_dataset_info(os.path.join(self.root, f'dataset_info_{self.phase}.json'))
    
    def load_img_info(self):
        img_paths = []
        labels = []
        attributes = []
        frequencies = []
        for key, label in self.data['label'].items():
            img_paths.append(os.path.join(self.root, 'images', self.data['path'][key].split('/')[-1]))
            labels.append(label)
            frequencies.append(int(self.data['frequency'][key]))

            # intra-class attribute SHOULD NOT be used in training
            att_label = int(self.data['attribute'][key])
            attributes.append(att_label)
            
        # save dataset info
        self.dataset_info['img_paths'] = img_paths
        self.dataset_info['labels'] = labels
        self.dataset_info['attributes'] = attributes
        self.dataset_info['frequencies'] = frequencies

        return img_paths, labels, attributes, frequencies
    
    def save_dataset_info(self, save_path):
        with open(save_path, 'w') as f:
            json.dump(self.dataset_info, f)
        
    def __getitem__(self, index):
        img_path = self.img_paths[index]
        label = self.labels[index]
        attribute = self.attributes[index]
        frequency = self.frequencies[index]
        

        img = Image.open(img_path).convert('RGB')
        img = self.transform(img)

        if self.phase == 'train':
            return img, label, attribute, index
        else:
            return img, label
    
    def __len__(self):
        return len(self.img_paths)
    

class coco_imbatt_balcls_transform():
    def __init__(self, phase, rgb_mean, rgb_std, rand_aug, RandomResizedCrop=None, test_Resize=None, test_CenterCrop=None):
        self.phase = phase
        self.rgb_mean = rgb_mean if rgb_mean is not None else [0.473, 0.429, 0.370]
        self.rgb_std = rgb_std if rgb_std is not None else [0.277, 0.268, 0.274]
        self.rand_aug = rand_aug
        self.RandomResizedCrop = RandomResizedCrop if RandomResizedCrop is not None else [112, 0.5, 1.0]
        self.test_Resize = test_Resize if test_Resize is not None else 128
        self.test_CenterCrop = test_CenterCrop if test_CenterCrop is not None else 112
        self.transform = self.get_transform()
    
    def get_transform(self):
        if self.phase == 'train':
            if self.rand_aug:
                transform = transforms.Compose([
                    # transforms.RandomResizedCrop(112, scale=(0.5, 1.0)),
                    transforms.RandomResizedCrop(self.RandomResizedCrop[0], scale=(self.RandomResizedCrop[1], self.RandomResizedCrop[2])),
                    transforms.RandomHorizontalFlip(),
                    RandAugment(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.rgb_mean, self.rgb_std),
                ])
            else:
                transform = transforms.Compose([
                    # transforms.RandomResizedCrop(112, scale=(0.5, 1.0)),
                    transforms.RandomResizedCrop(self.RandomResizedCrop[0], scale=(self.RandomResizedCrop[1], self.RandomResizedCrop[2])),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(self.rgb_mean, self.rgb_std),
                ])
        else:
            transform = transforms.Compose([
                            # transforms.Resize(128),
                            # transforms.CenterCrop(112),
                            transforms.Resize(self.test_Resize),
                            transforms.CenterCrop(self.test_CenterCrop),
                            transforms.ToTensor(),
                            transforms.Normalize(self.rgb_mean, self.rgb_std),
                ])
            
        return transform
    
    def __call__(self, img):
        return self.transform(img)
    
        
def visualization(original_tensor,saved_file = "_cifar/example.png"):
    from torchvision import transforms
    unloader = transforms.ToPILImage()
    image = original_tensor.cpu().clone()  # clone the tensor
    image = image.squeeze(0)  # remove the fake batch dimension
    image = unloader(image)
    image = image.resize((512,512))
    image.save(saved_file)

if __name__ == '__main__':
    import sys
    sys.path.append(os.path.dirname(sys.path[0]))
    from utils.logger import custom_logger

    for phase in ['train', 'test']:
        for rand_aug in [True, False]:
            exp_dir = f'./_coco/ALT/exp/phase_{phase}_rand_aug_{rand_aug}'
            logger = custom_logger(output_path=exp_dir, name="log")
            logger.info(f'=====> {phase} dataset')
            transform = coco_imbatt_balcls_transform(phase, rgb_mean=[0.473, 0.429, 0.370], rgb_std=[0.277, 0.268, 0.274], rand_aug=rand_aug)
            dataset = coco_imbatt_balcls(root='./_coco/ALT/coco_cbl', phase=phase, logger=logger, transform=transform)
            logger.info(f'=====> dataset length: {len(dataset)}')

            loader = data.DataLoader(dataset, batch_size=22, shuffle=True, num_workers=4, pin_memory=True)

            for i, (img, label, attribute, index) in enumerate(loader):
                logger.info(f"=====> img: {img.shape}")
                logger.info(f"=====> label: {label}")
                logger.info(f"=====> attribute: {attribute}")
                logger.info(f"=====> index: {index}")
                visualization(img[0], saved_file=f'{exp_dir}/example_{index[0].numpy()}.png')
                break
