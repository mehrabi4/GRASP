# GRASP: A goodness-of-fit test for classification learning.
This repository contains MATLAB and Python implementation of a general goodness-of-fit test for binary classification called GRASP. The implementation is based on the paper https://arxiv.org/abs/2209.02064 by Adel Javanmard, and Mohammad Mehrabi. 
# Overview
The Performance of classifiers is often measured in terms of average accuracy on test data. Despite being a standard measure, average accuracy fails in characterizing the fit of the model to the underlying conditional law of labels given the features vector (Y|X), e.g. due to model misspecification, over/under fitting, and high-dimensionality. Here, we consider the fundamental problem of assessing the goodness-of-fit for a general binary classifier. Our framework does not make any parametric assumption on the conditional law $Y|X$, and treats that as a black box oracle model which can be accessed only through queries. We formulate the goodness-of-fit assessment problem as a tolerance hypothesis testing of the form

$$
H_0:\mathbb{E}\bigg[ D_f\Big( \mathsf{Bern}(\eta(x)) || \mathsf{Bern}(\widehat{\eta}(X)) \Big) \bigg]\leq \tau,
$$

where $D_f$ represents an f-divergence function, and $\eta(x)$, $\widehat{\eta}(x)$ respectively denote the true and an estimate likelihood for a feature vector $x$ admitting a positive label. We propose a novel test, called GRASP for testing $H_0$, which works in finite sample settings, no matter the features (distribution-free). We also propose model-X GRASP designed for model-X settings where the joint distribution of the features vector is known. Model-X GRASP uses this distributional information to achieve better power. 

# Instructions

### Distribution free setting ###
#### Main function ####

Script "GRASP.m" contains the main function GRASP. This is an implementation of the distribution-free GRASP with the score function $T(x,w)=w$. For the main hypothesis testing problem, it outputs two p-values--one which is valid for the finite-number of samples and a less conservative version which is valid in the asymptotoic regime.  This function has the following input and output arguments:

**X**: feature matrix with size n by d  (n=#samples and d =#features)

**Y**: binary response values of size $n$. 


**heta_val**= estimate model $\widehat{\eta}$ values, i.e. $\widehat{\eta}(\mathbf{X})$. It is a vector of size $n$. 


**tau**: the $\tau$ value in the hypothesis testing problem


**alpha**: predetermined significance level $\alpha$.


**f_div**: the f-divergence function. It can be "H" (Hellinger distance), "kl" (kl-divergence), or "tv" (total variation).


**L**: number of labels for statistics $V_{n,L}$


Outputs:


**p_val_finite**: a p-value for the hypothesis testing problem which is valid in finite-sample regime.


**p_val_asym**:  a p-value for the hypothesis testing problem which is valid in asymptotic regime.


**reject_finite**: rejection status based on p_val_finite atsignificance level alpha. It can be true or false


**reject_asym**: rejection status based on p_val_asym at significance level alpha. It can be true or false

#### Simple example ####4
Script "GRASP_example.m" is a simple example for funciton "GRASP". It is for number of samples $n=5000$, feature dimension $d=200$, and number of labels $L=50$. 

### Model-X setting ###
#### Agnostic score function ####
Script "model_X_GRASP.m" is an example for implementation of the model-X GRASP procedure in the paper.  In tnis example, the model-X GRASP is run with the agnostic score function which is given by


$$ T(x,w)=\frac{1}{2\widehat{\eta}(x)}\mathbb{1}(w\leq \widehat{\eta}(x)) + \frac{1}{2(1-\widehat{\eta}(x))}\mathbb{1}(\widehat{\eta}(x)\leq w)    $$
#### GAN-based score function ####

Script "GAN_main.ipynb" is an example for model-X GRASP with a GAN-based score function.






