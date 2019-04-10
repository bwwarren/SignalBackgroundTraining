# SignalBackgroundTraining

This repository was born out of research on machine learning for particle physics done for David Miller at The University of Chicago. I intend it to be for those continuing similar research for professor Miller at UChicago. Hopefully, it will serve as a guide for how to set up the tools you need as well as how to run them on data in the formats that were given to me.

The goal of this Github repository is to train images of JZ0W (background) interactions and ZvvHbb (signal) interactions using Deep Learning such that we will be able to distinguish the two effectively. It will contain Keras and Caffe (coming soon) notebooks that run identical algorithms on the same data -- just adjusted for these two libraries.

The notebooks display and clean the data (format it from hdf5 to numpy, remove 0 values in images, and add labels). Then the data is shuffled and passed to a Convolution Neural Netork model to train the data. Finally, they look at the accuracy and loss of the CNN model and plot ROC curves to get a better sense of how the model is doing.

## Formatting data
## Visualizing data
## Training data
