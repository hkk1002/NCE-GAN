# NCE-GAN
Example codes to generate samples for (given) target density function using noise-contrastive estimation generative adversarial networks (NCE-GAN).
You can read details of NCE-GAN from my master's thesis: [Dihedral angle prediction using generative adversarial networks](https://arxiv.org/abs/1803.10996). NCE-GAN is explained in II.5 and the method used in the example codes is explained in A.1.

<br/>

## Comparison of [generative adversarial networks](https://arxiv.org/abs/1406.2661) GAN and NCE-GAN
| Model                                          | GAN                               | NCE-GAN  |
|------------------------------------------------|-----------------------------------|----------|
| Density estimation                             | Not possible (only density ratio) | Possible |
| Generate samples for given dataset             | Possible                          | Possible |
| Generate samples for given target distribution | Not possible                      | Possible |
<br/>

## Options 
* You can choose example target density by specifying "ex_choice".
* You can choose loss for the generator by specifying "G_loss_choice".
<br/>

## Note
* "G_loss_choice='Minibatch'" means ...
* "G_loss_choice='ln2'" means ...
* generate_from_density_2D_new.py ...

<br/>

## Dependencies
* Numpy
* Matplotlib
* Scipy
* Tensorflow
<br/>

## Result examples
* code: generate_from_density_2D_new, ex_choice: 'Ring', G_loss_choice: 'Minibatch'
<div>
<img width="400" src="https://user-images.githubusercontent.com/24959722/44981083-8aa36580-afac-11e8-9c6c-acc9c5ab0813.png">
<img width="400" src="https://user-images.githubusercontent.com/24959722/44981269-07ceda80-afad-11e8-82a6-6281411c86a8.png">
<img width="400" src="https://user-images.githubusercontent.com/24959722/44981275-0e5d5200-afad-11e8-9848-a362b94d8af5.png">
<img width="400" src="https://user-images.githubusercontent.com/24959722/44981072-8414ee00-afac-11e8-89f5-a5092aeed3c0.png">
<img width="400" src="https://user-images.githubusercontent.com/24959722/44982524-220ab780-afb1-11e8-8c96-972a3baacd62.png">
<div>

<br/>
 
* code: generate_from_density_2D_new, ex_choice: 'Grid', G_loss_choice: 'Minibatch'
<div>
<img width="400" src="https://user-images.githubusercontent.com/24959722/44981138-aeff4200-afac-11e8-8e39-3e11fc01baf8.png">
<img width="400" src="https://user-images.githubusercontent.com/24959722/44981193-d950ff80-afac-11e8-91bf-7267ff3eabde.png">
<img width="400" src="https://user-images.githubusercontent.com/24959722/44981207-e241d100-afac-11e8-8d67-cadbed10152f.png">
<img width="400" src="https://user-images.githubusercontent.com/24959722/44981124-a60e7080-afac-11e8-8cbd-47e18e2b1724.png">
<img width="400" src="https://user-images.githubusercontent.com/24959722/44982568-4feffc00-afb1-11e8-988b-55aefb9ac68e.png">
<div>

<br/>
