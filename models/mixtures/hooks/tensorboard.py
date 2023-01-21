from torch.utils.tensorboard import SummaryWriter
import torch

class HookTensorBoard():

    def __init__(self):
        self.monitor = False
        self.base_loss = 0
    
    def set_monitoring(self, logdir: str, model_name: str):
        self.monitor = True
        self.writer = SummaryWriter(f'{logdir}/{model_name}/')

    def set_base_loss(self, base_loss: torch.Tensor):
        self.base_loss = base_loss

    def add_means(self, means: torch.Tensor, iteration: int):
        for i in range(len(means)):
            self.writer.add_scalars('Means_X/train', {
                f'Y_EXPERIMENT_{i}': means[i][1]
            }, iteration)

            self.writer.add_scalars('Means_Y/train', {
                f'Y_EXPERIMENT_{i}': means[i][1]
            }, iteration)
    
    def add_weights(self, weights: torch.Tensor, iteration: int):
        for i in range(len(weights)):
            self.writer.add_scalars(f'Weights/train', {
                f'w_{i}': weights[i]
            }, iteration)

    def add_loss(self, loss: torch.Tensor, iteration: int):
        self.writer.add_scalars(f'Loss/train', {
            'GMM_EXPERIMENT': loss,
            'GMM_BASE': self.base_loss,
        }, iteration)

    def clear_monitoring(self):
        self.writer.flush()