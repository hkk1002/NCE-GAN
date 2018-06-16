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
