# GRASP: A goodness-of-fit test for classification learning.
This repository contains MATLAB and Python implementation of a general goodness-of-fit test for binary classification called GRASP. The implementation is based on the paper https://arxiv.org/abs/2209.02064 by Adel Javanmard, and Mohammad Mehrabi. 
# Overview
The Performance of classifiers is often measured in terms of average accuracy on test data. Despite being a standard measure, average accuracy fails in characterizing the fit of the model to the underlying conditional law of labels given the features vector (Y|X), e.g. due to model misspecification, over/under fitting, and high-dimensionality. Here, we consider the fundamental problem of assessing the goodness-of-fit for a general binary classifier. Our framework does not make any parametric assumption on the conditional law $Y|X$, and treats that as a black box oracle model which can be accessed only through queries. We formulate the goodness-of-fit assessment problem as a tolerance hypothesis testing of the form

\begin{equation*}
H_0:\mathbb{E}\left[ D_f\Big(\mathsf{Bern}(\heta(x)) \| \mathsf{Bern}(\eta(X)) \Big) \right]
\end{equation*}


```math
H_0:\mathbb{E}\left[ D_f\Big(\mathsf{Bern}(\heta(x)) \| \mathsf{Bern}(\eta(X)) \Big) \right]
```


where Df represents an f-divergence function, and η(x), η̂ (x) respectively denote the true and an estimate likelihood for a feature vector x admitting a positive label. We propose a novel test, called \grasp for testing H0, which works in finite sample settings, no matter the features (distribution-free). We also propose model-X \grasp designed for model-X settings where the joint distribution of the features vector is known. Model-X \grasp uses this distributional information to achieve better power. We evaluate the performance of our tests through extensive numerical experiments
