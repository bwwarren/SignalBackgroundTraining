# SignalBackgroundTraining
The goal of this notebook is to train images of JZ0W (background) interactions and ZvvHbb (signal) interactions using Deep Learning such that we will be able to distinguish the two effectively. 

The notebook cleans the data (formats it from hdf5 to numpy, removes 0 values in images, and adds labels).  
Then I shuffle the data and train it using a CNN.

I look at the accuracy and loss of this model and then I have plotted an ROC curve to get a better sense of how this model is doing.
