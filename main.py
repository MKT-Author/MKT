import argparse
import datetime
import logging
import os
import traceback
import copy
import random
import math
from options import Option
from dataloader import DataLoader
from methods.inference import Trainer
import utils as utils
from quantization_utils.quant_modules import *
from pytorchcv.model_provider import get_model as ptcv_get_model


class ExperimentDesign:
    def __init__(self, options, conf_path, ckpt_file):
        self.settings = options or Option(conf_path)

        self.train_loader = None
        self.test_loader = None
        self.model = None
        self.model_teacher = None

        self.trainer = None
        self.test_input = None
        self.unfreeze_Flag = True
        self.ckpt_file = ckpt_file

        os.environ['CUDA_DEVICE_ORDER'] = "PCI_BUS_ID"
        # os.environ['CUDA_VISIBLE_DEVICES'] = self.settings.visible_devices

        self.prepare()


    def prepare(self):

        self._set_dataloader()
        self._set_model()
        self._replace()

        self._set_trainer()

    def _set_dataloader(self):
        # create data loader
        data_loader = DataLoader(dataset=self.settings.dataset,
                                 batch_size=self.settings.batchSize,
                                 data_path=self.settings.dataPath,
                                 n_threads=self.settings.nThreads,
                                 ten_crop=self.settings.tenCrop)

        self.train_loader, self.test_loader = data_loader.getloader()

    def _set_model(self):
        if self.settings.dataset in ["cifar100", "cifar10"]:
            self.test_input = Variable(torch.randn(1, 3, 32, 32).cuda())
            self.model = ptcv_get_model(self.settings.network, pretrained=True)
            self.model_teacher = ptcv_get_model(self.settings.network, pretrained=True)
            self.model_teacher.eval()
            self.trainer_method = Trainer

        elif self.settings.dataset in ["imagenet"]:
            self.test_input = Variable(torch.randn(1, 3, 224, 224).cuda())
            self.model = ptcv_get_model(self.settings.network, pretrained=True)
            self.model_teacher = ptcv_get_model(self.settings.network, pretrained=True)
            self.model_teacher.eval()
            self.trainer_method = Trainer

        else:
            assert False, "unsupport data set: " + self.settings.dataset

    def _set_trainer(self):

        self.trainer = self.trainer_method(
            model=self.model,
            model_teacher=self.model_teacher,
            test_loader=self.test_loader,
            settings=self.settings,
            ckpt_file=self.ckpt_file,
        )

    def quantize_model(self, model):
        """
		Recursively quantize a pretrained single-precision model to int8 quantized model
		model: pretrained single-precision model
		"""

        weight_bit = self.settings.qw
        act_bit = self.settings.qa

        # quantize convolutional and linear layers
        if type(model) == nn.Conv2d:
            quant_mod = Quant_Conv2d(weight_bit=weight_bit)
            quant_mod.set_param(model)
            return quant_mod
        elif type(model) == nn.Linear:
            quant_mod = Quant_Linear(weight_bit=weight_bit)
            quant_mod.set_param(model)
            return quant_mod

        # quantize all the activation
        elif type(model) == nn.ReLU or type(model) == nn.ReLU6:
            return nn.Sequential(*[model, QuantAct(activation_bit=act_bit)])

        # recursively use the quantized module to replace the single-precision module
        elif type(model) == nn.Sequential:
            mods = []
            for n, m in model.named_children():
                mods.append(self.quantize_model(m))
            return nn.Sequential(*mods)
        else:
            q_model = copy.deepcopy(model)
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    setattr(q_model, attr, self.quantize_model(mod))
            return q_model

    def _replace(self):
        self.model = self.quantize_model(self.model)

    def freeze_model(self, model):
        """
		freeze the activation range
		"""
        if type(model) == QuantAct:
            model.fix()
        elif type(model) == nn.Sequential:
            for n, m in model.named_children():
                self.freeze_model(m)
        else:
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    self.freeze_model(mod)
            return model

    def unfreeze_model(self, model):
        """
		unfreeze the activation range
		"""
        if type(model) == QuantAct:
            model.unfix()
        elif type(model) == nn.Sequential:
            for n, m in model.named_children():
                self.unfreeze_model(m)
        else:
            for attr in dir(model):
                mod = getattr(model, attr)
                if isinstance(mod, nn.Module) and 'norm' not in attr:
                    self.unfreeze_model(mod)
            return model

    def run(self):
        # inference with the provided weight
        self.freeze_model(self.model)
        test_error, test_loss, test5_error = self.trainer.test()
        print('Top 1 Accuracy: %f, Top 5 Accuracy: %f' % (100 - test_error, 100 - test5_error))


def main():
    parser = argparse.ArgumentParser(description='Baseline')
    parser.add_argument('--conf_path', type=str, metavar='conf_path',
                        help='input the path of config file')
    parser.add_argument('--ckpt', default='', help='checkpoint to inference')
    args = parser.parse_args()

    option = Option(args.conf_path)

    experiment = ExperimentDesign(option, args.conf_path, args.ckpt)
    experiment.run()


if __name__ == '__main__':
    main()
