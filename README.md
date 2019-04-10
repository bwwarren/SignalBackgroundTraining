# SignalBackgroundTraining

This repository was born out of research on machine learning for particle physics done for David Miller at The University of Chicago. I intend it to be for those continuing similar research for professor Miller at UChicago. Hopefully, it will serve as a guide for how to set up the tools you need as well as how to run them on data in the formats that were given to me.

The goal of this Github repository is to train images of JZ0W (background) interactions and ZvvHbb (signal) interactions using Deep Learning such that we will be able to distinguish the two effectively. It will contain Keras and Caffe (coming soon) notebooks that run identical algorithms on the same data -- just adjusted for these two libraries.

The notebooks display and clean the data (format it from hdf5 to numpy, remove 0 values in images, and add labels). Then the data is shuffled and passed to a Convolution Neural Netork model to train the data. Finally, they look at the accuracy and loss of the CNN model and plot ROC curves to get a better sense of how the model is doing.

## Formatting data
For formatting the data I used a former graduate students code to deal with the structure of the data. It was originally in nTuple form and this graduate student's code formatted it from nTuples to hdf5 files such that the data was organized more clearly. It can be found on his Github [here](gFEXtuple2hdf5.py). I made slight edits to the code to deal with the slightly different formatting of the data I was using. The code linked above formats the data into a 24x32 shape, whilst I shaped it into a 28x32 shape -- this was simply had to do with what section of the data we were looking at.
## Visualizing data
## Training data
