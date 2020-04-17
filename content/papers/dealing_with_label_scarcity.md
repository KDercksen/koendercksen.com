Title: Dealing with label scarcity in computational pathology: a use case in prostate cancer classification
Date: 16-05-2019
Category: papers

This is work done during my MSc thesis internship at Radboudumc, together with
Wouter Bulten and Geert Litjens. Our paper was published at
[MIDL2019](https://2019.midl.io) as an extended abstract.

Read the paper [here](https://arxiv.org/abs/1905.06820).

#### Abstract
Large amounts of unlabelled data are commonplace for many applications in
computational pathology, whereas labelled data is often expensive, both in time
and cost, to acquire. We investigate the performance of unsupervised and
supervised deep learning methods when few labelled data are available. Three
methods are compared: clustering autoencoder latent vectors (unsupervised), a
single layer classifier combined with a pre-trained autoencoder
(semi-supervised), and a supervised CNN. We apply these methods on hematoxylin
and eosin (H&E) stained prostatectomy images to classify tumour versus
non-tumour tissue. Results show that semi-/unsupervised methods have an
advantage over supervised learning when few labels are available. Additionally,
we show that incorporating immunohistochemistry (IHC) stained data provides an
increase in performance over only using H&E.
