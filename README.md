# Similarity based Multiple Kernel estimation of rice agronomic traits, using multispectral imagery.

## Abstract
<p align="justify">
The huge wastes on Nitrogen fertilizer for rice cultivation and the consequent pollution
of the environment, created the need to map the rice paddies’ needs in fertilizer with more
efficient ways. With the rapid development of UAVs (unmanned aerial vehicle) equipped
with high-tech multispectral sensors, the collection of field data, even from flooded
paddies, became way easier. Most of the models created so far for this cause, are
empirical linear regression models, that make use of vegetation and hyperspectral indices.
This study implements machine learning algorithms based in Support Vector
Regression (SVR) and multikernel learning models. The final adjustment of the kernels
was based in a similarity measurement with the ideal output kernel. Two (2) main models
were developed for this purpose and each architecture was tested in three (3) different
scenarios. All aimed to predict each of the eighteen (18) output field traits combining the
input data in different ways. The inputs were the one hundred seventeen (117) vegetation
indices, which were collected during the three main growth stages of the plant. Two
different preprocessing methods were applied in the input data and, depending on the
model’s architecture, they created a final combined kernel for each output variable. The
different approaches and scenarios are then compared to find the optimal one.
For each output value that the models are able to predict, the most important features
are selected, for the regarding prediction. Finally, some suggestions for further
improvement are stated.

**Key words:** 
support vectors, regression, multikernel, kernel alignment, vegetation indices

## Description of the repo

There are two main models created for the purpose of my Master thesis. Their goal was to train few multikernel models (described on the 
individual readme files in each specific forlder) in order to predict the output agronomic traits. The models differ in:
* Scaling
   * Input and/or output
      * Robust Scaler
      * Standard Scaler
* Type of splitting in training and test set
   * (a), 80% train set - 20% test set (regarding the year)
   * (b), 80% train set - 20% test set (regarding the year & the type of treatment)
   * LOOT, Leave One Out Testing 
* RBF Kernels' parameters selection 
   * static
   * dynamic
* Grid Search - Cross validation
   * LOOCV, Leave One Out Cross Validation
   * Holdout
</p>
  
- - - -
  
### Model A:

<p align="center">
<img src= "https://github.com/bkara14/Thesis-models/blob/master/modelA.png">
</p>

- - - -

### Model B:

<p align="center">
<img src= "https://github.com/bkara14/Thesis-models/blob/master/modelB.png">
</p>
