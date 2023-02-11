import os
import math
import torch
from torch import optim
from models import BaseVAE
from models.types_ import *
from utils import data_loader
import pytorch_lightning as pl
from torchvision import transforms
import torchvision.utils as vutils
from torchvision.datasets import CelebA
from torch.utils.data import DataLoader
import math
import matplotlib.pyplot as plt


import torch
import torch.nn.functional as nnf

class ShufflePatches(object):
  def __init__(self, patch_size):
    self.ps = patch_size

  def __call__(self, x):
    # divide the batch of images into non-overlapping patches
    u = nnf.unfold(x, kernel_size=self.ps, stride=self.ps, padding=0)
    # permute the patches of each image in the batch
    pu = torch.cat([b_[:, torch.randperm(b_.shape[-1])][None,...] for b_ in u], dim=0)
    # fold the permuted patches back together
    f = nnf.fold(pu, x.shape[-2:], kernel_size=self.ps, stride=self.ps, padding=0)
    return f


class VAEXperiment(pl.LightningModule):

    def __init__(self,
                 vae_model: BaseVAE,
                 params: dict) -> None:
        super(VAEXperiment, self).__init__()

        self.model = vae_model
        self.params = params
        self.curr_device = None
        self.hold_graph = False
        self.count = 0
        try:
            self.hold_graph = self.params['retain_first_backpass']
        except:
            pass

        self.first_values = []
        self.count_i, self.count_j, self.count_total = 0, 0, 0

    def forward(self, input: Tensor, **kwargs) -> Tensor:
        return self.model(input, **kwargs)

    # def forward(self, input: Tensor, **kwargs) -> Tensor:
    #     quantized_inputs = torch.load("car_comb_4.pt").cuda()
    #     results =  self.model.decode(quantized_inputs)
    #     vutils.save_image(results[0],
    #                       os.path.join(self.logger.log_dir , 
    #                                    "Reconstructions", 
    #                                    f"recons_{self.logger.name}_combined_image_{self.count:03d}.png"),
    #                       normalize=True,
    #                       nrow=1)
    #     exit(0)

    # def forward(self, input: Tensor, **kwargs) -> Tensor:
    #     results = self.model(input, **kwargs)

    #     vutils.save_image(results[0],
    #                       os.path.join(self.logger.log_dir , 
    #                                    "Reconstructions", 
    #                                    f"recons_{self.logger.name}_real_image_{self.count:03d}.png"),
    #                       normalize=True,
    #                       nrow=1)

    #     self.count += 1

    #     return results


    # def forward(self, input: Tensor, **kwargs) -> Tensor:
    #     results = self.model(input, **kwargs)

    #     vutils.save_image(results[0],
    #                       os.path.join(self.logger.log_dir , 
    #                                    "Reconstructions", 
    #                                    f"recons_{self.logger.name}_real_image_{self.count:03d}.png"),
    #                       normalize=True,
    #                       nrow=1)

    #     self.count += 1

    #     return results


    # def forward(self, input: Tensor, **kwargs) -> Tensor:
    #     # img = torch.load("quantized_tensor_img.pt")
    #     video = torch.load("quantized_tensor.pt")
    #     encoding = self.model.encode(input)[0]
    #     quantized_inputs, vq_loss = self.model.vq_layer(encoding)
    #     quantized_inputs_sum = quantized_inputs
    #     for i in range(1, len(video)):
    #         quantized_inputs_sum +=  video[i]-video[i-1]
    #         results =  self.model.decode(quantized_inputs_sum)

    #         self.count+=1
    #         vutils.save_image(results[0],
    #                       os.path.join(self.logger.log_dir , 
    #                                    "Reconstructions", 
    #                                    f"recons_{self.logger.name}_image_{self.count:03d}.png"),
    #                       normalize=True,
    #                       nrow=1)

    # def forward(self, input: Tensor, **kwargs) -> Tensor:

    #     enc = self.model.encode(input)[0]
    #     # exit(0)

    #     # self.first_values.append(enc[0][0][50][50].item())
    #     self.first_values.append(enc[0][0][self.count_i][self.count_j].item())

    #         # exit(0)


    #     return self.model(input, **kwargs)


    # def forward(self, input: Tensor, **kwargs) -> Tensor:
    #     # img = torch.load("quantized_tensor_img.pt")
    #     video = torch.load("quantized_tensor.pt")
    #     video_2 = torch.load('quantized_tensor_841d9f6bbe50ec6f122568beab222093.pt')
    #     # encoding = self.model.encode(input)[0]
    #     # quantized_inputs, vq_loss = self.model.vq_layer(encoding)
    #     # quantized_inputs_sum = quantized_inputs
    #     first_values = []
    #     shuffle_patch = ShufflePatches(16)

    #     for i in range(1, len(video)):
    #         # quantized_inputs_sum +=  video[i]-video[i-1]
    #         results =  self.model.decode(video[i].unsqueeze(0))

    #         self.count+=1
    #         vutils.save_image(results[0],
    #                       os.path.join(self.logger.log_dir , 
    #                                    "Reconstructions", 
    #                                    f"recons_{self.logger.name}_image_{self.count:03d}.png"),
    #                       normalize=True,
    #                       nrow=1)

    #         change_idx = 32
    #         video_clone = video[i].clone()

    #         # print(video_clone.shape, video_clone[0][0][0])
    #         first_values.append(video_clone[0][0][0].item())

    #         # video_fuse = video[70]
    #         video_flipped = video_2[i]#torch.flip(video_clone, [2])
    #         video_combined = video_clone.clone()
    #         # print(video_combined.shape)
    #         # exit(0)
    #         # video_flipped = shuffle_patch(video_flipped.unsqueeze(0))[0]
    #         video_flipped[:,20:50,20:50] = video_combined[:,20:50,20:50]
    #         # video_flipped[:,0:20,0:20] = video_combined[:,20:40,20:40]
    #         # video_flipped[:,40:60,40:60] = video_combined[:,20:40,20:40]

    #         # video_flipped[:,0:32,0:32] = video_fuse[:,0:32,0:32]

    #         # results =  self.model.decode(video_combined.unsqueeze(0))
    #         results =  self.model.decode(video_flipped.unsqueeze(0))

    #         self.count+=1
    #         vutils.save_image(results[0],
    #                       os.path.join(self.logger.log_dir , 
    #                                    "Reconstructions", 
    #                                    f"recons_{self.logger.name}_image_{self.count:03d}_flipped.png"),
    #                       normalize=True,
    #                       nrow=1)

    #     plt.hist(first_values)
    #     plt.savefig(os.path.join(self.logger.log_dir , 
    #                                    "Reconstructions", 
    #                                    f"histogram_first_value_{self.logger.name}.png"))
    #     # exit(0)

    def training_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        results = self.forward(real_img, labels = labels)
        train_loss = self.model.loss_function(*results,
                                              M_N = self.params['kld_weight'], #al_img.shape[0]/ self.num_train_imgs,
                                              optimizer_idx=optimizer_idx,
                                              batch_idx = batch_idx)
        
        self.log_dict({key: val.item() for key, val in train_loss.items()}, sync_dist=True)

        return train_loss['loss']

    def validation_step(self, batch, batch_idx, optimizer_idx = 0):
        real_img, labels = batch
        self.curr_device = real_img.device

        # print(real_img.shape, labels.shape)
        results = self.forward(real_img, labels = labels)
        # print(results.shape)
        # exit(0)
        val_loss = self.model.loss_function(*results,
                                            M_N = 1.0, #real_img.shape[0]/ self.num_val_imgs,
                                            optimizer_idx = optimizer_idx,
                                            batch_idx = batch_idx)

        self.log_dict({f"val_{key}": val.item() for key, val in val_loss.items()}, sync_dist=True)

        # self.on_validation_end()
        # self.count+=1

        # rand_quant_inputs = self.pt_file('rand_quantized_input.pt')
        # print(rand_quant_inputs.shape)
        # vutils.save_image(rand_quant_inputs[0],
        #                   os.path.join(self.logger.log_dir , 
        #                                "Reconstructions", 
        #                                f"recons_{self.logger.name}_image_{self.count:03d}.png"),
        #                   normalize=True,
        #                   nrow=1)

        # exit(0)
        # print(len(results), results[0].shape)
        # vutils.save_image(results[0],
        #                   os.path.join(self.logger.log_dir , 
        #                                "Reconstructions", 
        #                                f"recons_{self.logger.name}_image_{self.count:03d}.png"),
        #                   normalize=True,
        #                   nrow=1)

    # def pt_file(self, filename):
    #     f = torch.load(filename)
    #     out = self.model.decode(f)
    #     return out

    def on_validation_end(self) -> None:
        self.sample_images()
        pass
        
    def sample_images(self):
        # Get sample reconstruction image            
        test_input, test_label = next(iter(self.trainer.datamodule.test_dataloader()))
        # print(test_input.shape, test_label.shape)
        # print(len(self.trainer.datamodule.test_dataloader()))
        test_input = test_input.to(self.curr_device)
        test_label = test_label.to(self.curr_device)

#         test_input, test_label = batch
        recons = self.model.generate(test_input, labels = test_label)
        
        # print(len(self.model.quantized_list))
        # self.model.quantized_ten = torch.cat(self.model.quantized_list[::-1][:len(self.trainer.datamodule.test_dataloader())], dim=0)
        # torch.save(self.model.quantized_ten, "quantized_tensor_841d9f6bbe50ec6f122568beab222093.pt")
        # torch.save(self.model.quantized_ten, "quantized_tensor_img.pt")
        vutils.save_image(recons.data,
                          os.path.join(self.logger.log_dir , 
                                       "Reconstructions", 
                                       f"recons_{self.logger.name}_Epoch_{self.count}.png"),
                          normalize=True,
                          nrow=int(math.sqrt(test_label.shape[0])))
        self.count+=1

        # try:
        samples = self.model.sample(144, self.curr_device, labels = test_label)
        vutils.save_image(samples.cpu().data,
                            os.path.join(self.logger.log_dir , 
                                        "Samples",      
                                        f"{self.logger.name}_Epoch_{self.current_epoch}.png"),
                            normalize=True,
                            nrow=12)

        plt.hist(self.first_values)
        plt.savefig(os.path.join(self.logger.log_dir , 
                                    "Reconstructions", 
                                    f"histogram_first_value_{self.logger.name}_{self.count_i:02d}_{self.count_j:02d}.png"))

        self.first_values = []
        self.count_total+=1
        self.count_j = self.count_total % 64
        self.count_i = self.count_total // 64
        # exit(0)
        # except Warning:
        #     pass

    def configure_optimizers(self):

        optims = []
        scheds = []

        optimizer = optim.Adam(self.model.parameters(),
                               lr=self.params['LR'],
                               weight_decay=self.params['weight_decay'])
        optims.append(optimizer)
        # Check if more than 1 optimizer is required (Used for adversarial training)
        try:
            if self.params['LR_2'] is not None:
                optimizer2 = optim.Adam(getattr(self.model,self.params['submodel']).parameters(),
                                        lr=self.params['LR_2'])
                optims.append(optimizer2)
        except:
            pass

        try:
            if self.params['scheduler_gamma'] is not None:
                scheduler = optim.lr_scheduler.ExponentialLR(optims[0],
                                                             gamma = self.params['scheduler_gamma'])
                scheds.append(scheduler)

                # Check if another scheduler is required for the second optimizer
                try:
                    if self.params['scheduler_gamma_2'] is not None:
                        scheduler2 = optim.lr_scheduler.ExponentialLR(optims[1],
                                                                      gamma = self.params['scheduler_gamma_2'])
                        scheds.append(scheduler2)
                except:
                    pass
                return optims, scheds
        except:
            return optims
