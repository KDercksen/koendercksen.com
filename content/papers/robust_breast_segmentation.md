Title: Robust multi-vendor breast region segmentation using deep learning
Date: 28-05-2020
Category: papers

This work was published as an extended abstract for the [IWBI2020
workshop](https://kuleuvencongres.be/iwbi2020). It was carried out during a
three month MSc internship at [Screenpoint Medical](https://screenpointmed.com)
under supervision of Jaap Kroes and Michiel Kallenberg.

Read the paper [here]({filename}/files/IWBI2020_workshop.pdf).
See the presentation [here](https://www.youtube.com/watch?v=4zvBhhhhEJQ) (2nd
presentation in session).

#### Abstract
Semantic segmentation of breast images is typically performed as a
preprocessing step for breast cancer detection by Computer Aided Diagnosis
(CAD) systems. While most literature on region segmentation is based on
conventional techniques like line estimation, thresholding and atlas-based
approaches, such methods may have problems with generalisation. This paper
investigates a robust multi-vendor breast region segmentation system for full
field digital mammograms (FFDM) and digital breast tomography (DBT) using a
U-Net neural network. Additionally, the effect of adding attention gates to the
U-Net architecture was analysed. The proposed networks were trained and tested
in a cross-validation setting on in-house FFDM/DBT data and the public INbreast
datasets, comprising over 10.000 FFDM and 3.500 DBT images from five different
vendors. Dice scores were obtained in the range 0.978 -- 0.985, with slightly
higher scores for the architecture that includes attention gates.
