# DeepLearningDigitIdentifier

- To run this code simply do the following to generate the images:
    python3 run.py
- The report is in this directory and is labeled REPORT.pdf
- In the run.py file the model being used can be changed, it is located at the top of the main method

- This will generate the 5 required images and save them in the graded_images directory
- The training can be run again by using the following command:
    python3 experiment.py --image input/train/1.png --model vgg16 --development y --redodata y --type datagen
- To get the training command to run you have to be sure to put the data images in the ./input/train/ directory -- they should be labeled the same way they are in the SVHN dataset. 1.png... 2.png... Please only use the full size images, not the cropped versions
- Please notice the developent flag, this means that the training will terminate early to speed up testing, if development is changed to n, then full training will be performed 
- This flag can be ommitted to train on all of the data again
- The graded images have already been generated, but they will be replaced if the run command is used again
- The models have been stored in the ./models directory in the project dir
- Links to Videos are below:

Presentation Video:

https://www.youtube.com/watch?v=sZ8RS6IgNag
