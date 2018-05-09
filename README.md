# universal-clasifier
Expert system designed to classify any provided data using neural network.

## data structure
In order to use this application you need to provide it with 3 files with exact same name and corresponding extensions as explained below in main directory of this program. Number of parameters provided in .params file determines input size of neural network, while number of lines in .outs determines number of possible classes. If any value in .data file is missing it should be replaced with '?'. Lines including this character are currently ignored.
When you run this program for new data set new neural network will be created and trained. After this process it will be saved in the same directory and used in the future. If you want to train neural network again simply remove corresponding .hdf5 file.

##### .data files
Files used for training neural network.
Those files should contain float values separated with commas:
5,1,1,1,2,1,3,1,1,2
Each line is a vector containing values taken as input of neural network, excluding last value which is expected output (class number).
##### .params files
Files used mainly when form for data input is created.
Each line is a text which will be listed with an entry box in this program.
Order and number of those names should correspond to order of values provided in .data file.
##### .outs files
Files used mainly during displaying results of classification.
Each line corresponds to the text which will be displayed when classifier recognizes data as corresponding class.
These lines should match possible classes listed in ascending order.
