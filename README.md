# NCE-GAN
Example codes to generate samples for (given) target density function using noise-contrastive estimation generative adversarial networks (NCE-GAN).
You can read details of NCE-GAN from my master's thesis: [Dihedral angle prediction using generative adversarial networks](https://arxiv.org/abs/1803.10996). NCE-GAN is explained in II.5 and the method used in the example codes is explained in A.1.

<br/>

## Comparison of [generative adversarial networks](https://arxiv.org/abs/1406.2661) (GAN) and NCE-GAN
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
<br/>

* Meaning of "G_loss_choice='ln2'" is described below.

I found this loss for the generator while I was thinking about a generator which only generates 0 and 1. In this case, the data only contains 0 and 1. And let’s say ![equation](https://latex.codecogs.com/gif.latex?%5Calpha) is the probability to draw 0 from the data distribution, and ![equation](https://latex.codecogs.com/gif.latex?%5Calpha_%7BG%7D) is the probability to draw 0 from the generator. If the generator works well, ![equation](https://latex.codecogs.com/gif.latex?%5Calpha_%7BG%7D) should be close to ![equation](https://latex.codecogs.com/gif.latex?%5Calpha). 

To simplify the description of the training process, let's assume the discriminator can find optimal solution for each iteration and optimizations of networks were done with infinite numbers of samples even though that is impossible.

Then, the (vanilla) training loss of the generator:

![equation](https://latex.codecogs.com/gif.latex?%5Cbegin%7Balign*%7D%20-%5Cmathbb%7BE%7D%5B%5Cln%20p%28S%3Dreal%7CG_%7Bi%7D%28z%29%29%5D%3D%26%20-%5B%5Calpha_%7BG_%7Bi%7D%7D%20%5Cln%28%5Cfrac%7B%5Calpha%7D%7B%5Calpha&plus;%5Calpha_G_%7Bi-1%7D%7D%29&plus;%281-%5Calpha_%7BG_%7Bi%7D%7D%29%20%5Cln%28%5Cfrac%7B1-%5Calpha%7D%7B%281-%5Calpha%29&plus;%281-%5Calpha_G_%7Bi-1%7D%29%7D%29%5D%5C%5C%20%3D%26%5Calpha_%7BG_%7Bi%7D%7D%20%5Cln%281&plus;%5Cfrac%7B%5Calpha_G_%7Bi-1%7D%7D%7B%5Calpha%7D%29&plus;%281-%5Calpha_%7BG_%7Bi%7D%7D%29%20%5Cln%281&plus;%5Cfrac%7B1-%5Calpha_G_%7Bi-1%7D%7D%7B1-%5Calpha%7D%29%5C%5C%20%3D%26%5Calpha_%7BG_%7Bi%7D%7D%20%5B%5Cln%281&plus;%5Cfrac%7B%5Calpha_G_%7Bi-1%7D%7D%7B%5Calpha%7D%29%20-%20%5Cln%281&plus;%5Cfrac%7B1-%5Calpha_G_%7Bi-1%7D%7D%7B1-%5Calpha%7D%29%5D&plus;%20%5Cln%281&plus;%5Cfrac%7B1-%5Calpha_G_%7Bi-1%7D%7D%7B1-%5Calpha%7D%29%20%5Cend%7Balign*%7D)

where ![equation](https://latex.codecogs.com/gif.latex?G_%7Bi%7D) the generator after i th iterations of the training and ![equation](https://latex.codecogs.com/gif.latex?%5Calpha_G_%7Bi%7D) is the probability to draw 0 from the generator after i th iterations of the training.

If we think about the optimal solution for each iteration, ![equation](https://latex.codecogs.com/gif.latex?%5Calpha_G_%7Bi%7D%3D0) is the optimal solution when ![equation](https://latex.codecogs.com/gif.latex?%5Calpha_G_%7Bi-1%7D%3E%5Calpha) and ![equation](https://latex.codecogs.com/gif.latex?%5Calpha_G_%7Bi%7D%3D1) is the optimal solution when ![equation](https://latex.codecogs.com/gif.latex?%5Calpha_G_%7Bi-1%7D%3C%5Calpha). And note the loss is ![equation](https://latex.codecogs.com/gif.latex?%5Cln%282%29) when ![equation](https://latex.codecogs.com/gif.latex?%5Calpha_G_%7Bi-1%7D%3D%5Calpha). This means if the generator can find the optimal solution for each iteration, ![equation](https://latex.codecogs.com/gif.latex?%5Calpha_G_%7Bi%7D) will continue to oscillate without converging to ![equation](https://latex.codecogs.com/gif.latex?%5Calpha) (except when ![equation](https://latex.codecogs.com/gif.latex?%5Calpha_G%3D0) or ![equation](https://latex.codecogs.com/gif.latex?%5Calpha_G%3D1) ).

If we think about the optimal solution for the equation ![equation](https://latex.codecogs.com/gif.latex?-%5Cmathbb%7BE%7D%5B%5Cln%20p%28S%3Dreal%7CG_%7Bi%7D%28z%29%29%5D%3D%5Cln2), ![equation](https://latex.codecogs.com/gif.latex?%5Calpha_G_%7Bi%7D%3Df%28%5Calpha_G_%7Bi-1%7D%29) is the optimal solution for each iteration for a function ![equation](https://latex.codecogs.com/gif.latex?f%28x%29%3D%5Cfrac%7B%5Cln2-%5Cln%281&plus;%5Cfrac%7B1-x%7D%7B1-%5Calpha%7D%29%7D%7B%5Cln%281&plus;%5Cfrac%7Bx%7D%7B%5Calpha%7D%29-%5Cln%281&plus;%5Cfrac%7B1-x%7D%7B1-%5Calpha%7D%29%7D). 

![equation](https://latex.codecogs.com/gif.latex?f%28x%29) has a fixed point at ![equation](https://latex.codecogs.com/gif.latex?%5Calpha). It seems ![equation](https://latex.codecogs.com/gif.latex?%7Cf%27%28x%29%7C%3C1) hold for ![equation](https://latex.codecogs.com/gif.latex?0%5Cle%20x%5Cle1) (I didn't prove as the calculation was too complex, but when I plotted graph for varing ![equation](https://latex.codecogs.com/gif.latex?%5Calpha), it seems to hold.). If the speculation is true, ![equation](https://latex.codecogs.com/gif.latex?%5Calpha_G_%7Bi%7D) will converge to ![equation](https://latex.codecogs.com/gif.latex?%5Calpha).

Hence, it seems solving ![equation](https://latex.codecogs.com/gif.latex?-%5Cmathbb%7BE%7D%5B%5Cln%20p%28S%3Dreal%7CG_%7Bi%7D%28z%29%29%5D%3D%5Cln2) would be a better way to get an wanted solution at least for the situation mentioned above. So, "G_loss_choice='ln2'" means that we are using the same idea also for the other cases. The generator trys to 
minimize ![equation](https://latex.codecogs.com/gif.latex?%28-%5Cmathbb%7BE%7D%5B%5Cln%20p%28S%3Dreal%7CG_%7Bi%7D%28z%29%29%5D-%5Cln2%29%5E2) so that it can get an approximate solution for ![equation](https://latex.codecogs.com/gif.latex?-%5Cmathbb%7BE%7D%5B%5Cln%20p%28S%3Dreal%7CG_%7Bi%7D%28z%29%29%5D%3D%5Cln2).
<br/>

* generate_from_density_2D_new.py: It handles the problem that ![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20p_%7Bmodel%7D%28%5Ctext%7BS%3Dtarget%7D%29) approaches zero as training continues. The only change is that the discriminator trains as if the discriminator is getting some samples from the target data distribution even trough it is not true. The datail is described below.

Let's think about the case where the discriminator gets the same amounts of samples from the data distribution ![equation](https://latex.codecogs.com/gif.latex?p_%7Bdata%7D%28x%29) and the contrastive noise distribution ![equation](https://latex.codecogs.com/gif.latex?p_%7Bnoise%7D%28x%29). Density estimation using noise-contrastive estimation (NCE) is one example for this case.

If we disregard the generated samples, 

![equation](https://latex.codecogs.com/gif.latex?p%28S%3Ddata%7Cx%29%3D%5Cfrac%7Bp_%7Bdata%7D%28x%29%7D%7Bp_%7Bdata%7D%28x%29&plus;p_%7Bnoise%7D%28x%29%7D) and ![equation](https://latex.codecogs.com/gif.latex?p%28S%3Dnoise%7Cx%29%3D%5Cfrac%7Bp_%7Bnoise%7D%28x%29%7D%7Bp_%7Bdata%7D%28x%29&plus;p_%7Bnoise%7D%28x%29%7D).

In generate_from_density_2D_new.py, the discriminator trains using soft label when it gets contrastive noise samples. 
The label for noise samples is defined as ![equation](https://latex.codecogs.com/gif.latex?%5Cdpi%7B100%7D%20%5Cbegin%7Bbmatrix%7D%20p_%7B%5Ctext%7Blabel%7D%7D%28%5Ctext%7BS%3Dtarget%7D%7Cx%29%20%5C%5C%20p_%7B%5Ctext%7Blabel%7D%7D%28%5Ctext%7BS%3Dgenerated%7D%7Cx%29%20%5C%5C%20p_%7B%5Ctext%7Blabel%7D%7D%28%5Ctext%7BS%3Dnoise%7D%7Cx%29%20%5Cend%7Bbmatrix%7D%3D%5Cbegin%7Bbmatrix%7D%20%5Cfrac%7Bp_%7Btarget%7D%28x%29%7D%7Bp_%7Btarget%7D%28x%29&plus;p_%7Bnoise%7D%28x%29%7D%20%5C%5C%200%20%5C%5C%20%5Cfrac%7Bp_%7Bnoise%7D%28x%29%7D%7Bp_%7Btarget%7D%28x%29&plus;p_%7Bnoise%7D%28x%29%7D%20%5Cend%7Bbmatrix%7D). 

Using this label in the training means the discriminator mimics the situation where it gets the same amounts of samples from the data and noise distribution.
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
