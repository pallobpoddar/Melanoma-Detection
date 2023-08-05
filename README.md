# Skin Cancer (Melanoma) Detection Using Deep Learning Technique
## Introduction
### What is Melanoma?
Melanoma, also redundantly known as malignant melanoma, is a type of skin cancer that develops from the pigment-producing cells known as melanocytes. Melanomas typically occur in the skin, but may rarely occur in the mouth, intestines, or eye (uveal melanoma). In women, they most commonly occur on the legs, while in men, they most commonly occur on the back. About 25% of melanomas develop from moles. Changes in a mole that can indicate melanoma include an increase in size, irregular edges, change in color, itchiness, or skin breakdown.
### Stats and Facts
Melanoma is the most dangerous type of skin cancer. Globally, in 2012, it newly occurred in 232,000 people. In 2015, 3.1 million people had active disease, which resulted in 59,800 deaths. Australia and New Zealand have the highest rates of melanoma in the world. High rates also occur in Northern Europe and North America, while it is less common in Asia, Africa, and Latin America. In the United States, melanoma occurs about 1.6 times more often in men than women. Melanoma has become more common since the 1960s in areas mostly populated by people of European descent.
### Objective
The purpose is to correctly identify the benign and malignant cases. A benign tumor is a tumor that DOES NOT invade its surrounding tissue or spread around the body. A malignant tumor is a tumor that MAY invade its surrounding tissue or spread around the body.

The goal is to predict a binary target for each image. The model should predict the probability (floating point) between 0.0 and 1.0 that the lesion in the image is malignant (the target). In the training data, train.csv, the value 0 denotes benign, and 1 indicates malignant.
### Dataset
The images are provided in DICOM format. This can be accessed using commonly-available libraries like pydicom, and contains both image and metadata. It is a commonly used medical imaging data format. Images are also provided in JPEG and TFRecord format (in the jpeg and tfrecords directories, respectively). Metadata is also provided outside of the DICOM format, in CSV files
### Metric of Evaluation
The metric of evaluation is area under the ROC curve. An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds. This curve plots TPR vs. FPR at different classification thresholds. Lowering the classification threshold classifies more items as positive, thus increasing both False Positives and True Positives.
