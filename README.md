# GRASP: A goodness-of-fit test for classification learning.
This repository contains MATLAB and Python implementation of a general goodness-of-fit test for binary classification called GRASP. The implementation is based on the paper https://arxiv.org/abs/2209.02064 by Adel Javanmard, and Mohammad Mehrabi. 
# Overview
The Performance of classifiers is often measured in terms of average accuracy on test data. Despite being a standard measure, average accuracy fails in characterizing the fit of the model to the underlying conditional law of labels given the features vector (Y|X), e.g. due to model misspecification, over/under fitting, and high-dimensionality. Here, we consider the fundamental problem of assessing the goodness-of-fit for a general binary classifier. Our framework does not make any parametric assumption on the conditional law $Y|X$, and treats that as a black box oracle model which can be accessed only through queries. We formulate the goodness-of-fit assessment problem as a tolerance hypothesis testing of the form

$$
H_0:\mathbb{E}\bigg[ D_f\Big( \mathsf{Bern}(\widehat{\eta}(x)) || \mathsf{Bern}(\eta(X)) \Big) \bigg]\leq \tau,
$$

where $D_f$ represents an f-divergence function, and $\eta(x)$, $\widehat{\eta}(x)$ respectively denote the true and an estimate likelihood for a feature vector $x$ admitting a positive label. We propose a novel test, called GRASP for testing $H_0$, which works in finite sample settings, no matter the features (distribution-free). We also propose model-X GRASP designed for model-X settings where the joint distribution of the features vector is known. Model-X GRASP uses this distributional information to achieve better power. 

# Instructions

### Distribution free setting ###


Script "GRASP.m" contains the main function GRASP. For the main hypothesis testing problem, it outputs two p-values--one wchich is valid for finite-number of samples and a less conservative one which is valid in asymptotoic regime--  This is for the distribution-free setting, and the main function has the following input arguments:

**X**: feature matrix with size n by p  (n=#samples and p = #features)

**Y**: Binary response values of size n. 


**heta_val**= Output of test model heta on features X. 


**tau**: the tau value in the hypothesis testing problem


**alpha**: predetermined significance level 


**f_div**: the f-divergence function. It can be "H" (Hellinger distance), "kl" (kl-divergence, or "tv" (total variation).


**L**: number of labels for statistics V_{n,L}


Outputs:
**p_val_finite**: a p-value for the hypothesis testing problem which is valid in finite-sample regime.


**p_val_asym**:  a p-value for the hypothesis testing problem which is valid in asymptotic regime.


**reject_finie**: rejection status based on p_val_finite atsignificance level alpha.


**reject_asym**: rejection status based on p_val_asym at significance level alpha.


Script "GRASP_example.m" is a simple example for the main funciton "GRASP".

Script "model_X_GRASP.m" is an example for the model-X GRASP algorithm. In tnis example, the model-X GRASP is run with the agnostic score function.

Script "GAN_main." is an example for model-X GRASP with a GAN-based score function.






