
from tensorflow.keras import callbacks
import os
# from realpath import runtime_dir, log_dir, backup_dir, tensorboard_dir

import matplotlib.pyplot as plt
from tensorflow.keras.utils import array_to_img
import numpy as np
from .data_loader import denormalize_image
from random import randrange, sample


# FID Score
from tensorflow.keras.applications.inception_v3 import InceptionV3
import numpy as np
from numpy import cov, iscomplexobj, trace
from scipy.linalg import sqrtm

## done: csvlogger, BackupAndRestore
## review: tensorboard
## to do: ModelCheckpoint, encapsulate in methods


## CSVLogger
# create paths
'''
csv_save_path = os.path.join(runtime_dir, log_dir)
os.makedirs(csv_save_path, exist_ok=True)
'''
def create_CSVLogger(csv_save_path):
    csv_logger = callbacks.CSVLogger(
    filename = csv_save_path,
    append = True # True or False
    )
    return csv_logger

## BackupAndRestore
# create paths
'''
backup_save_path = os.path.join(runtime_dir, backup_dir)
os.makedirs(backup_save_path, exist_ok=True)
'''

def create_BackupAndRestore(backup_save_path):
    backup_maker = callbacks.BackupAndRestore(backup_dir = backup_save_path)
    return backup_maker

## Progbar. Auto created
# progbar_callback = callbacks.ProgbarLogger(count_mode = 'samples')

## TensorBoard
'''
tensorboard_save_path = os.path.join(runtime_dir, tensorboard_dir)
os.makedirs(tensorboard_save_path, exist_ok=True)
'''
def create_TensorBoard(tensorboard_save_path, write_graph=False, write_images=True, update_freq='epoch', profile_batch=0):
    tensorboard_callback = callbacks.TensorBoard(
    log_dir=tensorboard_save_path,
    write_graph=write_graph, # True or False
    write_images=write_images, # True or False
    write_steps_per_second=True, # True or False
    update_freq=update_freq, # 'batch' or 'epoch' TRY
    profile_batch=profile_batch
    )
    return tensorboard_callback


def createModelCheckpoint(ckpt_filepath, monitor='generators', save_best_only=True, save_weights_only=False, mode='min', verbose=1):
    model_checkpoint = callbacks.ModelCheckpoint(
        filepath=ckpt_filepath,
        monitor=monitor,
        verbose=verbose,
        save_best_only=save_best_only,
        save_weights_only=save_weights_only,
        mode=mode
    )
    return model_checkpoint



class FID_Score(callbacks.Callback):
    def __init__(self, ds, freq, path, mult=10, initial_size=15, sample_size=5):
        super().__init__()
        self.ds = ds
        self.freq = freq
        self.path = path
        self.mult = mult
        self.initial_size = initial_size
        self.sample_size = sample_size

        ## Initialize fid parameters
        real_block = np.array(list(self.ds.take(initial_size)))
        real_X = real_block[:, 0].reshape(-1, *real_block.shape[-3:])
        real_Y = real_block[:, 1].reshape(-1, *real_block.shape[-3:])

        ## Load inception model
        self.inception = InceptionV3(include_top=False, input_shape=(256,256,3), pooling='avg')
        features_real_X = self.inception.predict(real_X, verbose=0)
        features_real_Y = self.inception.predict(real_Y, verbose=0)

        # mean and covariance of domain X images
        self.mu_X = features_real_X.mean(axis=0)
        self.sigma_X = cov(features_real_X, rowvar=False)

        # mean and covariance of domain Y images
        self.mu_Y = features_real_Y.mean(axis=0)
        self.sigma_Y = cov(features_real_Y, rowvar=False)

        self.epoch=0

        with open(self.path, 'w') as f:
            f.write('freq,fid_score_fake_X,fid_score_fake_Y\n')


    def on_batch_end(self, batch, logs={}):
        # Only each self.freq batches
        if self.freq != 'epoch' and batch % self.freq == 0:
            # After first epoch, only each self.freq*self.mult batches
            if self.epoch == 0 or batch % (self.freq*self.mult) == 0:
                
                real_block = np.array(list(self.ds.take(self.sample_size)))
                real_X = real_block[:, 0].reshape(-1, *real_block.shape[-3:])
                real_Y = real_block[:, 1].reshape(-1, *real_block.shape[-3:])

                fake_Y = self.model.gen_G.predict(real_X, verbose=0)
                fake_X = self.model.gen_F.predict(real_Y, verbose=0)


                features_fake_X = self.inception.predict(fake_X, verbose=0)
                features_fake_Y = self.inception.predict(fake_Y, verbose=0)

                mu_fake_X, sigma_fake_X = features_fake_X.mean(axis=0), cov(features_fake_X, rowvar=False)
                ssdiff_X = np.sum((self.mu_X - mu_fake_X)**2.0)
                covmean_X = sqrtm(self.sigma_X.dot(sigma_fake_X))
                if iscomplexobj(covmean_X):
                    covmean_X = covmean_X.real
                fid_X = ssdiff_X + trace(self.sigma_X + sigma_fake_X - 2.0 * covmean_X)

                mu_fake_Y, sigma_fake_Y = features_fake_Y.mean(axis=0), cov(features_fake_Y, rowvar=False)
                ssdiff_Y = np.sum((self.mu_Y - mu_fake_Y)**2.0)
                covmean_Y = sqrtm(self.sigma_Y.dot(sigma_fake_Y))
                if iscomplexobj(covmean_Y):
                    covmean_Y = covmean_Y.real
                fid_Y = ssdiff_Y + trace(self.sigma_Y + sigma_fake_Y - 2.0 * covmean_Y)


                with open(self.path, 'a') as f:
                    f.write('ep{}batch{},{},{}\n'.format(self.epoch, batch, fid_X, fid_Y))


    def on_epoch_end(self, epochs, logs=None):
        self.epoch = self.epoch + 1

        if self.freq == 'epoch':

            real_block = np.array(list(self.ds.take(10)))
            real_X = real_block[:, 0].reshape(-1, *real_block.shape[-3:])
            real_Y = real_block[:, 1].reshape(-1, *real_block.shape[-3:])

            fake_Y = self.model.gen_G.predict(real_X, verbose=0)
            fake_X = self.model.gen_F.predict(real_Y, verbose=0)


            features_fake_X = self.inception.predict(fake_X, verbose=0)
            features_fake_Y = self.inception.predict(fake_Y, verbose=0)

            mu_fake_X, sigma_fake_X = features_fake_X.mean(axis=0), cov(features_fake_X, rowvar=False)
            ssdiff_X = np.sum((self.mu_X - mu_fake_X)**2.0)
            covmean_X = sqrtm(self.sigma_X.dot(sigma_fake_X))
            if iscomplexobj(covmean_X):
                covmean_X = covmean_X.real
            fid_X = ssdiff_X + trace(self.sigma_X + sigma_fake_X - 2.0 * covmean_X)

            mu_fake_Y, sigma_fake_Y = features_fake_Y.mean(axis=0), cov(features_fake_Y, rowvar=False)
            ssdiff_Y = np.sum((self.mu_Y - mu_fake_Y)**2.0)
            covmean_Y = sqrtm(self.sigma_Y.dot(sigma_fake_Y))
            if iscomplexobj(covmean_Y):
                covmean_Y = covmean_Y.real
            fid_Y = ssdiff_Y + trace(self.sigma_Y + sigma_fake_Y - 2.0 * covmean_Y)

            ## Write to file
            with open(self.path, 'a') as f:
                f.write('{},{},{}\n'.format(epochs, fid_X, fid_Y))

            return fid_X, fid_Y



class Batch_Logger(callbacks.Callback):
    def __init__(self, path):
        super().__init__()
        self.path = path
        self.first_epoch = True
        self.logstring = ''

    def on_batch_end(self, batch, logs={}):
        if self.logstring == '':
            self.logstring = 'batch,' + ','.join(str(l) for l in logs.keys()) + '\n'

        if self.first_epoch:
            self.logstring += str(batch) + ',' + ','.join(str(l) for l in logs.values()) + '\n'


    def on_epoch_end(self, epochs, logs=None):
        if self.first_epoch:
            self.first_epoch = False
            with open(self.path, 'w') as f:
                f.write(self.logstring)

        



class ImageLoggerCallback(callbacks.Callback):

    def __init__(self, dataset, dom_A_name='', dom_B_name='', save_dir='', num_images=4, metabatch_range=5):
        super().__init__()
        assert save_dir != '' and dom_A_name != '' and dom_B_name != ''
        self.ds = dataset
        self.dom_A_name = dom_A_name
        self.dom_B_name = dom_B_name
        self.num_images = num_images
        self.metabatch_range = metabatch_range
        self.save_dir = save_dir

    
    def on_epoch_end(self, epoch, logs=None):
        # images: numbers of rows, each row with realX, fakeY, realY, fakeX

        real_numpy_ds = np.array(list(self.ds.take(self.metabatch_range)))
        metabatch_choice = randrange(self.metabatch_range)

        batch_x, batch_y = real_numpy_ds[metabatch_choice]

        assert batch_x.shape[0] == batch_y.shape[0]

        batch_size = batch_x.shape[0]
        images_x = batch_x[sample(range(batch_size), k=self.num_images)]
        images_y = batch_y[sample(range(batch_size), k=self.num_images)]
        
        predictions_x = self.model.gen_G.predict(images_x, verbose=0)
        predictions_y = self.model.gen_F.predict(images_y, verbose=0)

        fig, ax = plt.subplots(self.num_images, 4, figsize=(15, 15), dpi=200)
        fig.patch.set_facecolor('w')

        for i in range(self.num_images):
            # pick image_x, generate image_y, denormalize images, plot
            image_x = images_x[i]
            prediction_x = predictions_x[i]

            image_y = images_y[i]
            prediction_y = predictions_y[i]
            # denormalize image
            # sample # image_x = denormalize_image(image_x).numpy().astype(np.uint8)
            image_x = denormalize_image(image_x).astype(np.uint8)
            prediction_x = denormalize_image(prediction_x).astype(np.uint8)
            image_y = denormalize_image(image_y).astype(np.uint8)
            prediction_y = denormalize_image(prediction_y).astype(np.uint8)
            
            
            # plot images domain X
            ax[i, 0].imshow(image_x)
            ax[i, 1].imshow(prediction_x)
            ax[i, 0].set_title(f"Input {self.dom_A_name} image")
            ax[i, 1].set_title(f"Translated {self.dom_B_name} image")
            ax[i, 0].axis("off")
            ax[i, 1].axis("off")

            # plot images domain Y
            ax[i, 2].imshow(image_y)
            ax[i, 3].imshow(prediction_y)
            ax[i, 2].set_title(f"Input {self.dom_B_name} image")
            ax[i, 3].set_title(f"Translated {self.dom_A_name} image")
            ax[i, 2].axis("off")
            ax[i, 3].axis("off")

            '''
            prediction_y = 'NOT IMPLEMENTED' # acuerdate de convertir a numpy con .numpy()
            image = denormalize_image(image).numpy().astype(np.uint8)
            prediction = denormalize_image(prediction).astype(np.uint8)
            '''

        plt.savefig( os.path.join(self.save_dir, 'ImagePlot_epoch{}.png'.format(epoch + 1)) )
        # plt.show() # deactivate to save time in training
        plt.close()


