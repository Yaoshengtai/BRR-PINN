This is a project using a normal method, Boundary Region Reinforcement Physics-informed Neural Network (BRR-PINNs), to solve a thermo-elastic system. 

The following three jupyter notebooks are solving the heat transfer, force deformation, thermo-elastic coupling deformation problem, respectively.
    heat.ipynb
    force.ipynb
    couple.ipynb

The folder "PINN" is the code of BRR-PINNs method. These codes refer to the implementation of the neuroeqdiff[1] package: https://github.com/NeuroDiffGym/neurodiffeq.

The folder "fem_data" includes the results to the three problems using Finite Element Method, which are used as ground truth data.

The folder "Implement" includes the implement details. Before using the .sh file, you should change the jupyter notebook to a .py file.

The folder "model_result" includes the result .pth model. You can directly load these models.

[1] Chen F, Sondak D, Protopapas P, et al. Neurodiffeq: A python package for solving differential equations with neural networks[J]. Journal of Open Source Software, 2020, 5(46): 1931.

    
