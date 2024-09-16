import torch 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from IPython.display import clear_output

import torch 
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from IPython.display import clear_output

class Monitor2DSpatial_heat:
    # Monitor during training, in a 2D spatial heat transfer problem

    # param check_on_x: the x coordinates of the check points
    # type check_on_x: torch.tensor

    # param check_on_y: the y coordinates of the check points
    # type check_on_y: torch.tensor

    # param check_every: every 'check_every' epochs to operate function 'check', to save model/show figure/...
    # type check_every: int

    # param device: where to generate check points (e.g. cpu/gpu/...)
    # type device: torch.device

    # param args: the global arguments
    # type args: argparse.ArgumentParser.parse_args

    def __init__(self, check_on_x, check_on_y, check_every, device, args):
        self.using_non_gui_backend = matplotlib.get_backend() == 'agg'
        self.device=device
        xy_tensor = torch.cartesian_prod(check_on_x, check_on_y).to(self.device)
        
        self.xx_tensor = torch.squeeze(xy_tensor[:, 0])
        self.yy_tensor = torch.squeeze(xy_tensor[:, 1])

        self.xx_array = self.xx_tensor.clone().detach().cpu().numpy()
        self.yy_array = self.yy_tensor.clone().detach().cpu().numpy()

        self.check_every = check_every

        self.check_on_x=check_on_x
        self.check_on_y=check_on_y
        self.args=args

    def check(self, approximator, history, epoch):
        # save the model
        torch.save(approximator.single_network,str(self.args.save_dict)+"-model/"+str(epoch)+'.pth')

        # save the training history
        history_array = np.array([value  for value in history.values()]).T
        np.savetxt(self.args.save_dict + '-history.txt', history_array)
        
        # a figure showning the current training result 
        clear_output(wait=True)
        fig, axs = plt.subplots(3, 3, figsize=(13, 11))

        # the forward computing
        uu_array = approximator(self.xx_tensor, self.yy_tensor)[:,0]
        uu_array=uu_array.detach().cpu().numpy()

        # create a heatmap
        xx, yy = np.meshgrid(self.check_on_x, self.check_on_y)
        heatmap=axs[0,0].pcolormesh(xx, yy, uu_array.reshape(xx.shape).T,cmap='rainbow') 
        contour_lines = axs[0,0].contour(xx, yy, uu_array.reshape(xx.shape).T, 10,colors='black', linewidths=0.5)
        cbar=plt.colorbar(heatmap,ax=axs[0,0],label='Temperature')
        axs[0,0].set_xlabel('r')
        axs[0,0].set_ylabel('z')
        axs[0,0].set_title('Heatmap')

        # monitor the training loss, defined metircs ...
        i=0 ; j=0
        for metric_name, metric_values in history.items():
            j=j+1
            if j>=3:
                i=i+1
                j=0
            axs[i,j].plot(metric_values,label=metric_name)
            axs[i,j].set_title(metric_name)
            axs[i,j].set_xlabel('epochs')
            axs[i,j].set_ylabel('loss')
            axs[i,j].set_yscale('log')
        
        # You can define anything you want to monitor here
            
        plt.tight_layout()
        plt.savefig(self.args.save_dict+"-image/"+str(epoch)+".png" , dpi=200)



class Monitor2DSpatial_deform:
    # Monitor during training, in a 2D spatial deformation problem

    # param check_on_x: the x coordinates of the check points
    # type check_on_x: torch.tensor

    # param check_on_y: the y coordinates of the check points
    # type check_on_y: torch.tensor

    # param check_every: every 'check_every' epochs to operate function 'check', to save model/show figure/...
    # type check_every: int

    # param device: where to generate check points (e.g. cpu/gpu/...)
    # type device: torch.device

    # param args: the global arguments
    # type args: argparse.ArgumentParser.parse_args

    def __init__(self, check_on_x, check_on_y, check_every, device, args):
        self.using_non_gui_backend = matplotlib.get_backend() == 'agg'
        self.device=device
        xy_tensor = torch.cartesian_prod(check_on_x, check_on_y).to(self.device)
        
        self.xx_tensor = torch.squeeze(xy_tensor[:, 0])
        self.yy_tensor = torch.squeeze(xy_tensor[:, 1])

        self.xx_array = self.xx_tensor.clone().detach().cpu().numpy()
        self.yy_array = self.yy_tensor.clone().detach().cpu().numpy()

        self.check_every = check_every

        self.check_on_x=check_on_x
        self.check_on_y=check_on_y
        self.args=args

    def check(self, approximator, history, epoch):
        # save the model
        torch.save(approximator.single_network,str(self.args.save_dict)+"-model/"+str(epoch)+'.pth')

        # save the training history
        history_array = np.array([value  for value in history.values()]).T
        np.savetxt(self.args.save_dict + '-history.txt', history_array)
        
        # a figure showning the current training result 
        clear_output(wait=True)
        fig, axs = plt.subplots(4, 3, figsize=(13, 15))

        # the forward computing
        uu_array = approximator(self.xx_tensor, self.yy_tensor)
        uu_array=uu_array.detach().cpu().numpy()

        # create a heatmap
        xx, yy = np.meshgrid(self.check_on_x, self.check_on_y)
        heatmap=axs[0,0].pcolormesh(xx, yy, uu_array[:,0].reshape(xx.shape).T,cmap='rainbow') 
        contour_lines = axs[0,0].contour(xx, yy, uu_array[:,0].reshape(xx.shape).T, 10,colors='black', linewidths=0.5)
        cbar=plt.colorbar(heatmap,ax=axs[0,0],label='Deplacement$\mu m$')
        axs[0,0].set_xlabel('r')
        axs[0,0].set_ylabel('z')
        axs[0,0].set_title('r-direction Deplacement')

        xx, yy = np.meshgrid(self.check_on_x, self.check_on_y)
        heatmap=axs[0,1].pcolormesh(xx, yy, uu_array[:,1].reshape(xx.shape).T,cmap='rainbow') 
        contour_lines = axs[0,1].contour(xx, yy, uu_array[:,1].reshape(xx.shape).T, 10,colors='black', linewidths=0.5)
        cbar=plt.colorbar(heatmap,ax=axs[0,1],label='Deplacement$\mu m$')
        axs[0,1].set_xlabel('r')
        axs[0,1].set_ylabel('z')
        axs[0,1].set_title('z-direction Deplacement')

        # monitor the training loss, defined metircs ...
        i=0 ; j=1
        for metric_name, metric_values in history.items():
            j=j+1
            if j>=3:
                i=i+1
                j=0
            axs[i,j].plot(metric_values,label=metric_name)
            axs[i,j].set_title(metric_name)
            axs[i,j].set_xlabel('epochs')
            axs[i,j].set_ylabel('loss')
            axs[i,j].set_yscale('log')
        
        # You can define anything you want to monitor here
            
        plt.tight_layout()
        plt.savefig(self.args.save_dict+"-image/"+str(epoch)+".png" , dpi=200)

