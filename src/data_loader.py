
import os

from tensorflow.keras.utils import image_dataset_from_directory
from tensorflow.data import AUTOTUNE as autotune
from tensorflow.data import Dataset

from .realpath import domain_ds_dir

## File or function name alternatives: make_dataset

'''
    image directory structure for labels not inferred
    remove labels (unsupervised dataset) 
    DATASET SAMPLE FORMAT (image) or (image, image)
    batch size ???
    seed necesary if train and val are separated
    
'''

def combine_datasets(ds_a, ds_b, size_option='min'):
    '''
    combine two datasets
    Args:
        ds_a: dataset A
        ds_b: dataset B
        size_option: 'max', 'mean' or 'min'

    Returns:
        tuple (ds_ab, size)
            ds_ab: dataset with both datasets
            size: number of batches in the smaller dataset (epoch size)
    '''
    assert size_option in ['max', 'mean', 'min']

    size_a = ds_a.cardinality().numpy()
    size_b = ds_b.cardinality().numpy()

    if size_option == 'max':
        size = max(size_a, size_b)
    elif size_option == 'mean':
        size = (size_a + size_b) // 2
    elif size_option == 'min':
        size = min(size_a, size_b)

    ds_aR = ds_a.repeat()
    ds_bR = ds_b.repeat()

    combined_ds = Dataset.zip((ds_aR, ds_bR))

    return combined_ds, size
   

def normalize_image(image):
    return (image / 127.5) - 1.0

def denormalize_image(image):
    return (image + 1.0) * 127.5

def optimize_dataset(dataset, option='full'):
    ## cache, prefetch, shuffle, repeat
    # shuffle: load method
    # repeat: load method combine method
    assert option in ['full', 'cache', 'prefetch']

    if option == 'cache' or option == 'full':
        dataset = dataset.cache()
    if option == 'prefetch' or option == 'full':
        dataset = dataset.prefetch(buffer_size=autotune)
    return dataset

def load_domain_dataset(domain_name, batch_size=32, map_function=None, optimize=False, load_seed=None):
    # load dataset
    ds = image_dataset_from_directory(
        directory = os.path.join(domain_ds_dir, domain_name),
        labels='inferred',
        label_mode='int',
        batch_size=batch_size,
        image_size=(256, 256),
        seed=load_seed,
        validation_split=None,
        subset=None,
        crop_to_aspect_ratio=True
    )

    # remove labels (unsupervised dataset)
    # normalize images
    ds = ds.map(
        lambda x, y: normalize_image(x),
        num_parallel_calls=autotune
    )
    
    # apply map function
    if map_function is not None:
        ds = ds.map(
            map_function,
            num_parallel_calls=autotune
        )

    # optimize dataset
    if optimize:
        ds = optimize_dataset(ds)

    return ds
