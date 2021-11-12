# fixed size of images for the loader
fixed_size = (64, 256)

# overall epochs and restart epochs according to a cosine annealing with restarts shceduler
restart_epochs = 40
max_epochs = 4 * restart_epochs

# batch size of word images
batch_size = 100

# initial learning rate
nlr = 1e-3

# number of iterations between progress display
display = 100

# DNN architecture configurations
cnn_cfg = [(2, 64), 'M', (4, 128), 'M', (4, 256)] #, (2, 512)]
cnn_top = 256  # hidden size for CTC top
rnn_cfg = (256, 2)  # (hidden , num_layers)

# possible stopwords required for keyword spotting evaluation
stopwords = []

#classes = '_!"#&\'()*+,-./0123456789:;?ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz '

# reduced character set typically used in KWS settings (only lowercase alphanumeric characters)
def reduced(istr):
    rstr = ''.join([c if (c.isalnum() or c=='_' or c==' ') else '*' for c in istr.lower()])
    return rstr
