# Image to image domain translation using CycleGAN


`src/data_loader.py` Loading, pre and post processing tools and everything related with the data pipeline

`src/callbacks.py` Definition of built and custom callbacks used in the model training

`src/cyclegan.py` Main class. Definition of the main model, generators, discriminators, losses and metrics

`src/realpath.py` Library to manage the relative path, as well as to fully customize and change dataa and runtime path

`data` Default folder for data loading

`runtime` Default runtime directory, where the generated image and model weights are saved

