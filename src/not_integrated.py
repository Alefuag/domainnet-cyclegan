
import argparse
import cyclegan
import callbacks
import realpath

parser = argparse.ArgumentParser(description='Manage the settings for cyclegan')

# arguments: resnet_blocks, lr, batch_size, epochs, save_interval,

# args: net type, resnet blocks, resnet up down blocks, domain A, domain B, batch size

parser.add_argument('--net', '-n', type=str, default='resnet', help='Net type, either resnet or unet')
parser.add_argument('--resblocks', '-rblocks', type=int, default=9, help='ResNet blocks')
parser.add_argument('--domain_A', '-da', type=str, required=True, help='Domain A name')
parser.add_argument('--domain_B', '-db', type=str, required=True, help='Domain B name')
parser.add_argument('--batch_size', '-b',type=int, default=1, help='Batch size')
parser.add_argument('--epochs', '-e', type=int, default=100, help='Epochs')



dom_A_name = ''
dom_B_name = ''



realpath.set_domains(dom_A_name, dom_B_name)

realpath.domain_A_dir
realpath.domain_B_dir

### info file
log_string =  '''
    net type: {net},
    resnet blocks: {resblocks},
    domain A: {domain_A},
    domain B: {domain_B},
    batch size: {batch_size},
    epochs: {epochs}

'''


cycle_net = cyclegan.CycleGAN()


gen_G_optimizer = cyclegan.get_default_optimizer()
gen_F_optimizer = cyclegan.get_default_optimizer()
disc_X_optimizer = cyclegan.get_default_optimizer()
disc_Y_optimizer = cyclegan.get_default_optimizer()
gen_loss = cyclegan.generator_loss
disc_loss = cyclegan.discriminator_loss

cycle_net.compile(
    gen_G_optimizer = gen_G_optimizer,
    gen_F_optimizer = gen_F_optimizer,
    disc_X_optimizer = disc_X_optimizer,
    disc_Y_optimizer = disc_Y_optimizer,
    gen_loss = gen_loss,
    disc_loss = disc_loss
)


callback_list = [
    
    
]


