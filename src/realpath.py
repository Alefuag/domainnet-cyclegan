import os
import datetime

# timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

timestamp = '_'.join(str(datetime.datetime.now())[:19].split())

runtime_folder = 'runtime'
log_file = 'log.csv'
batch_log_file = 'batch_log.csv'
fid_log_file = 'fid_log.csv'
backup_folder = 'backup'
tensorboard_folder = 'tensorboard'
checkpoint_file = 'checkpoint.{epoch:02d}-{generators:.2f}.hdf5'
image_folder = 'images'



ds_folder = 'data'
domain_ds_folder = 'domain_dataset'


ROOT_DIR = os.path.realpath(os.path.join(os.path.dirname(__file__), '..'))

domain_ds_dir = os.path.join(ROOT_DIR, ds_folder, domain_ds_folder)


def set_domains(domain_A, domain_B, rt_folder_name=None):
    global domain_A_name, domain_B_name, domain_A_dir, domain_B_dir
    domain_A_name = domain_A
    domain_B_name = domain_B
    domain_A_dir = os.path.join(domain_ds_dir, domain_A_name)
    domain_B_dir = os.path.join(domain_ds_dir, domain_B_name)
    load_runtime_dirs(rt_folder_name=rt_folder_name)


def load_runtime_dirs(rt_folder_name=None):
    global runtime_dir, log_dir, batch_log_dir, fid_log_dir, backup_dir, tensorboard_dir, checkpoint_filepath, image_dir
    
    runtime_dir = get_runtime_dir(rt_folder_name=rt_folder_name)
    log_dir = os.path.join(runtime_dir, log_file)
    batch_log_dir = os.path.join(runtime_dir, batch_log_file)
    fid_log_dir = os.path.join(runtime_dir, fid_log_file)
    backup_dir = os.path.join(runtime_dir, backup_folder)
    tensorboard_dir = os.path.join(runtime_dir, tensorboard_folder)
    checkpoint_filepath = os.path.join(runtime_dir, checkpoint_file)
    image_dir = os.path.join(runtime_dir, image_folder)

    os.makedirs(runtime_dir, exist_ok=True)
    # os.makedirs(log_dir, exist_ok=True) # it's a file
    # os.makedirs(batch_log_dir, exist_ok=True) # it's a file
    # os.makedirs(fid_log_dir, exist_ok=True) # it's a file
    os.makedirs(backup_dir, exist_ok=True)
    os.makedirs(tensorboard_dir, exist_ok=True)
    # os.makedirs(checkpoint_filepath, exist_ok=True) # it's a file
    os.makedirs(image_dir, exist_ok=True)


def get_runtime_dir(rt_folder_name=None):
    if rt_folder_name is None:
        return os.path.join(ROOT_DIR, runtime_folder, 'rt_' + domain_A_name + '_to_' + domain_B_name + '_' + timestamp)
    else:
        return os.path.join(ROOT_DIR, runtime_folder, rt_folder_name)

