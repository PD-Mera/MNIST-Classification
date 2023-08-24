import os
import logging
from typing import Union

class LogConfigs():
    def __init__(self, 
                 experiment_dir: str,
                 log_filename: str = "training.log",
                 log_mode = "w",
                 log_format = '%(asctime)s : %(levelname)s : %(name)s : %(message)s'):
        self.log_save_path = os.path.join(experiment_dir, log_filename)
        self.log_mode = log_mode
        self.log_format = log_format
        
    def initialize(self):
        logging.basicConfig(filename = self.log_save_path,
                            format = self.log_format,
                            filemode = self.log_mode)
        logger = logging.getLogger() 
        logger.addHandler(logging.StreamHandler())
        logger.setLevel(logging.INFO) 
        return logger

class Preprocess():
    def __init__(self,
                 use_resize: bool = True,
                 image_size: list = [224, 224],
                 use_norm: bool = True, 
                 norm_mean: list = [0.485, 0.456, 0.406],
                 norm_std: list = [0.229, 0.224, 0.225]):
        self.use_resize = use_resize # use resize for image or not
        self.image_size = image_size # image size for resize (h, w)
        self.use_norm = use_norm # use normalize for image or not
        self.norm_mean = norm_mean # mean of normalize (RGB)
        self.norm_std = norm_std # std of normalize (RGB)

class Augment():
    def __init__(self,
                 x_flip_prob: float = 0.0,
                 y_flip_prob: list = [224, 224],
                 rotate: dict = {"limit": 45, "prob": 0.5}, 
                 brightness_contrast: dict = {"brightness": 0.2, "constrast": 0.2, "prob": 0.0}):
        self.x_flip_prob = x_flip_prob
        self.y_flip_prob = y_flip_prob
        self.rotate = rotate
        self.brightness_contrast = brightness_contrast

class GeneralConfigs():
    def __init__(self, 
                 model_backbone: str = "resnet18",
                 load_checkpoint: str = None,
                 classnames_file: str = "./configs/classes.names",
                 save_model_dir: str = "./results",
                 logger: logging.Logger = None):
        self.__MODEL_AVAILABLE = ["custom", "resnet18"]
        assert model_backbone in self.__MODEL_AVAILABLE, f"'model_backbone' must be in {self.__MODEL_AVAILABLE}"

        self.model_backbone = model_backbone
        self.load_checkpoint = load_checkpoint
        self.classnames_file = classnames_file
        self.save_model_dir = save_model_dir
        self.list_class_names = self.__get_list_class_names()
        self.class_num = len(self.list_class_names)
        self.logger = logger

        # preprocess
        self.preprocess = Preprocess(use_resize = True,                 # use resize for image or not
                                     image_size = [224, 224],           # image size for resize (h, w)
                                     use_norm = True,                   # use normalize for image or not
                                     norm_mean = [0.485, 0.456, 0.406], # mean of normalize (RGB)
                                     norm_std = [0.229, 0.224, 0.225])  # std of normalize (RGB)

        # augment
        self.augment = Augment(x_flip_prob = 0.0,                       # in range [0.0, 1.0], probability of Horizontal Flip (left to right)
                               y_flip_prob = 0.0,                       # in range [0.0, 1.0], probability of Vertical Flip (up to down)
                               rotate = {"limit": 45, "prob": 0.5},     # "limit" is the change limit (degree), "prob" is the probability in range [0.0, 1.0]
                               brightness_contrast = {"brightness": 0.2,    # "brightness" is the change limit of brightness
                                                      "constrast": 0.2,     # "contrast" is the change limit of contrast
                                                      "prob": 0.0})         # "prob" is the probability in range [0.0, 1.0]

        
    def __get_list_class_names(self):
        with open(self.classnames_file) as f:
            list_class_names = f.read().split("\n")
        return list_class_names

class RuntimeConfigs():
    def __init__(self, 
                 data_path: Union[str, list],
                 batch_size: int,
                 num_worker: int,
                 use_augmentation: bool = False):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_worker = num_worker
        self.use_augmentation = use_augmentation

class TrainConfigs(RuntimeConfigs):
    def __init__(self, 
                 data_path: Union[str, list],
                 batch_size: int = 32,
                 num_worker: int = 4,
                 use_augmentation: bool = True,
                 num_epoch: int = 100,
                 optimizer: str = "Adam",
                 learning_rate: float = 1e-4,
                 loss_fn: str = "CE",
                 save_model_dir: str = "./results/"):
        super(TrainConfigs, self).__init__(data_path, batch_size, num_worker, use_augmentation)
        self.__OPTIM_AVAILABLE = ["Adam"]
        assert optimizer in self.__OPTIM_AVAILABLE, f"'optimizer' must be in {self.__OPTIM_AVAILABLE}"

        self.__LOSS_FN_AVAILABLE = ["custom", "CE"]
        assert loss_fn in self.__LOSS_FN_AVAILABLE, f"'loss_fn' must be in {self.__LOSS_FN_AVAILABLE}"
        
        self.num_epoch = num_epoch
        self.optimizer = optimizer
        self.learning_rate = learning_rate
        self.loss_fn = loss_fn
        self.save_model_dir = save_model_dir

class ValidConfigs(RuntimeConfigs):
    def __init__(self, 
                 data_path: Union[str, list],
                 batch_size: int = 8,
                 num_worker: int = 4,
                 use_augmentation: bool = False):
        super(ValidConfigs, self).__init__(data_path, batch_size, num_worker, use_augmentation)
        