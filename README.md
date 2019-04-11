# SignalBackgroundTraining

This repository was born out of research on machine learning for particle physics done for David Miller at The University of Chicago. I intend it to be for those continuing similar research for professor Miller at UChicago. Hopefully, it will serve as a guide for how to set up the tools you need as well as how to run them on data in the formats that were given to me.

The goal of this Github repository is to train images of JZ0W (background) interactions and ZvvHbb (signal) interactions using Deep Learning such that we will be able to distinguish the two effectively. It will contain Keras and Caffe (coming soon) notebooks that run identical algorithms on the same data -- just adjusted for these two libraries.

The notebooks display and clean the data (format it from hdf5 to numpy, remove 0 values in images, and add labels). Then the data is shuffled and passed to a Convolution Neural Netork model to train the data. Finally, they look at the accuracy and loss of the CNN model and plot ROC curves to get a better sense of how the model is doing.

## Formatting data
For formatting the data I used a former graduate students code to deal with the structure of the data. It was originally in nTuple form and this graduate student's code formatted it from nTuples to hdf5 files such that the data was organized more clearly. It can be found on his Github [here](https://github.com/kratsg/gML.git) and it is the gFEXtuple2hdf5.py file. I made slight edits to the code to deal with the slightly different formatting of the data I was using. The code linked above formats the data into a 24x32 shape, whilst I shaped it into a 28x32 shape -- this was simply had to do with what section of the data we were looking at.

### ROOT requirements and UC Tier 3
To use the code to convert nTuple data to hdf5 files, you will need to install ROOT. This can be extremely tricky and time consuming if your environment is not set up correctly and you don't know what you are doing. But if you get through it, you will need to make sure that the python extension is also installed to use this code.

A nice work around (although it presents it's own unique challenges) is to use Tier 3 -- ask David for an account.
They have their own way of installing packages using something called [lcgenv](https://twiki.atlas-canada.ca/bin/view/AtlasCanada/Lcgenv)

Here are some commands I found to be useful to set up my environment, and to transfer files to and from Tier 3 once I ssh'ed into my Tier 3 account.
For set up:
```
setupATLAS # initial setup (gets lsetup command)

lsetup "lcgenv -p LCG_86 x86_64-slc6-gcc62-opt" # to see possible packages

lsetup "lcgenv -p LCG_86 x86_64-slc6-gcc62-opt pip”. #set up pip


lsetup "lcgenv -p LCG_86 x86_64-slc6-gcc62-opt ROOT" #set up ROOT

pip install --user numpy root-numpy ... # can then pip install other libraries ((matplotlib doesn’t work with this!))

  # then on future set ups, I can just do 

lsetup "lcgenv -p LCG_86 x86_64-slc6-gcc62-opt pip" 

```

In order to transfer files between the Tier 3 and your own computer you will need to use the SCP command: `scp /path/to/file username@a:/path/to/destination`and the reverese command to send files back to your own computer for visualization and training. As far as Tier 3 and ROOT goes this should set you up alright.

## Visualizing data
-- Notebook just for this. Allows you to see what happens to data
-- Shows what I am doing when formatting the data
-- Separate from code that just adds labels and outputs data that is used for training

## Training data
-- just formatting of data that pipes straight into a NN (not a notebook, just a python file)
-- Use of collaboratory or UC Tier 3 (ML requires a lot of computing power)
