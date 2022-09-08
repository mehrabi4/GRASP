# GRASP: A goodness-of-fit test for classification learning.
This repository contains MATLAB and Python implementation of a general goodness-of-fit test for binary classification called GRASP. The implementation is based on the paper https://arxiv.org/abs/2209.02064 by Adel Javanmard, and Mohammad Mehrabi. 
# Overview
The Performance of classifiers is often measured in terms of average accuracy on test data. Despite being a standard measure, average accuracy fails in characterizing the fit of the model to the underlying conditional law of labels given the features vector (Y|X), e.g. due to model misspecification, over/under fitting, and high-dimensionality. Here, we consider the fundamental problem of assessing the goodness-of-fit for a general binary classifier. Our framework does not make any parametric assumption on the conditional law $Y|X$, and treats that as a black box oracle model which can be accessed only through queries. We formulate the goodness-of-fit assessment problem as a tolerance hypothesis testing of the form

$$
H_0:\mathbb{E}\left[ D_f\Big( \mathsf{Bern}(\widehat{\eta}(x)) || \mathsf{Bern}(\eta(X)) \Big) \right]\leq \tau
$$

where $D_f$ represents an f-divergence function, and $\eta(x)$, $\widehat{\eta}(x)$ respectively denote the true and an estimate likelihood for a feature vector $x$ admitting a positive label. We propose a novel test, called GRASP for testing $H_0$, which works in finite sample settings, no matter the features (distribution-free). We also propose model-X GRASP designed for model-X settings where the joint distribution of the features vector is known. Model-X GRASP uses this distributional information to achieve better power. 
