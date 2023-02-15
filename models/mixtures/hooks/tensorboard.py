# External imports
from torch.utils.tensorboard import SummaryWriter
import torch

class HookTensorBoard():

    def __init__(self):
        self.monitor = False
        self.base_loss = 0
    
    def set_monitoring(self, logdir: str, model_name: str):
        self.monitor = True
        self.model_name = model_name
        self.writer = SummaryWriter(f'{logdir}/{model_name}/')

    def set_base_loss(self, base_loss: torch.Tensor):
        self.base_loss = base_loss
    
    def add_base_means(self, means: torch.Tensor, iteration: int):
        for i in range(len(means)):
            self.writer.add_scalars(f'Means_X/{self.model_name}/train', {
                f'X_BASE_{i}': means[i][0]
            }, iteration)

            self.writer.add_scalars(f'Means_Y/{self.model_name}/train', {
                f'Y_BASE_{i}': means[i][1]
            }, iteration)
    
    def add_base_weights(self, weights: torch.Tensor, iteration: int):
        for i in range(len(weights)):
            self.writer.add_scalars(f'Weights/{self.model_name}/train', {
                f'W_BASE_{i}': weights[i]
            }, iteration)

    def add_means(self, means: torch.Tensor, iteration: int):
        for i in range(len(means)):
            self.writer.add_scalars(f'Means_X/{self.model_name}/train', {
                f'X_EXPERIMENT_{i}': means[i][0]
            }, iteration)

            self.writer.add_scalars(f'Means_Y/{self.model_name}/train', {
                f'Y_EXPERIMENT_{i}': means[i][1]
            }, iteration)
    
    def add_weights(self, weights: torch.Tensor, iteration: int):
        for i in range(len(weights)):
            self.writer.add_scalars(f'Weights/{self.model_name}/train', {
                f'W_EXPERIMENT_{i}': weights[i]
            }, iteration)

    def add_loss(self, loss: torch.Tensor, iteration: int):
        self.writer.add_scalars(f'Loss/{self.model_name}/train', {
            'GMM_EXPERIMENT': loss,
            'GMM_BASE': self.base_loss,
        }, iteration)

    def clear_monitoring(self):
        self.writer.flush()