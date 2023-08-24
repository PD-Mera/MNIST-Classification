import os
import numpy as np
from typing import Union
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms as T
import albumentations as A
from configs.config import GeneralConfigs, TrainConfigs, ValidConfigs

from src.utils import logger_print, is_image_file


class LoadDataset(Dataset):
    def __init__(self, 
                 general_configs: GeneralConfigs = None, 
                 runtime_configs: Union[TrainConfigs, ValidConfigs] = None):
        super(LoadDataset, self).__init__()
        self.general_configs = general_configs
        self.runtime_configs = runtime_configs
        self.images = []
        self.class_names = self.general_configs.list_class_names
        
        if isinstance(self.runtime_configs.data_path, str):
            # assert len(os.listdir(self.config[self.phase]["DATA_PATH"])) == self.class_num, f"""Number of classes in config and in "{self.config[self.phase]["DATA_PATH"]}" must fit ({self.config[self.phase]["DATA_PATH"]} != {self.class_num})"""
            for class_dir in os.listdir(self.runtime_configs.data_path):
                assert class_dir in self.class_names, """Folder structure must be classname folder contain images"""

            self.images = self.get_images(self.runtime_configs.data_path)

        elif isinstance(self.runtime_configs.data_path, list):
            for dir_path in self.runtime_configs.data_path:
                # assert len(os.listdir(dir_path)) == self.class_num, f"""Number of classes in config and in "{dir_path}" must fit ({len(os.listdir(dir_path))} != {self.class_num})"""
                for class_dir in os.listdir(dir_path):
                    assert class_dir in self.class_names, """Folder structure must be classname folder contain images"""

                self.images = self.get_images(dir_path)

        label = [x.split("/")[-2] for x in self.images]
        self.count_label = {i:label.count(i) for i in self.class_names}
        logger_print(self.general_configs.logger, f"""{self.runtime_configs.__class__.__name__.replace("Configs", "")}: {self.count_label}""")
        for class_name in self.class_names:
            assert class_name in self.count_label.keys(), f""""{class_name}" class is missing image"""
        self.transform = self._preprocess_image_transform()
        self.augment = self._preprocess_image_augmentation()
  

    def __getitem__(self, index):
        image = Image.open(self.images[index]).convert("RGB")
        image = self.augment_image(image)
        image = self.transform(image)
        
        class_name = self.images[index].split('/')[-2]
        label = self.class_names.index(class_name)
        return image, label


    def __len__(self):
        return len(self.images)
    
    @staticmethod
    def get_images(folder_name: str):
        images = []
        for class_dir in os.listdir(folder_name):
            for image_name in os.listdir(os.path.join(folder_name, class_dir)):
                if is_image_file(os.path.join(folder_name, class_dir, image_name)):
                    images.append(os.path.join(folder_name, class_dir, image_name))
        return images

    @staticmethod
    def preprocess_image_transform(general_configs: GeneralConfigs):
        transform = []
        if general_configs.preprocess.use_resize:
            transform.append(T.Resize(general_configs.preprocess.image_size))
        
        transform.append(T.ToTensor())
        
        if general_configs.preprocess.use_norm:
            transform.append(T.Normalize(mean=general_configs.preprocess.norm_mean,
                                         std=general_configs.preprocess.norm_std))

        transform = T.Compose(transform)
        return transform
    
    def _preprocess_image_transform(self):
        return self.preprocess_image_transform(self.general_configs)
    
    def _preprocess_image_augmentation(self):
        augment = []

        if self.runtime_configs.use_augmentation:
            augment.append(A.HorizontalFlip(p = self.general_configs.augment.x_flip_prob))
            augment.append(A.VerticalFlip(p = self.general_configs.augment.y_flip_prob))
            augment.append(A.Rotate(limit = self.general_configs.augment.rotate["limit"],
                                    p = self.general_configs.augment.rotate["prob"]))
            augment.append(A.RandomBrightnessContrast(brightness_limit = self.general_configs.augment.brightness_contrast["brightness"], 
                                                      contrast_limit = self.general_configs.augment.brightness_contrast["constrast"], 
                                                      p = self.general_configs.augment.brightness_contrast["prob"]))

        self.augment = A.Compose(augment)
        return self.augment

    def augment_image(self, image: Image):
        # Convert PIL image to numpy array
        image_np = np.array(image)
        # Apply transformations
        augmented = self.augment(image=image_np)
        # Convert numpy array to PIL Image
        image = Image.fromarray(augmented['image'])
        return image
    
