import tensorflow.compat.v1 as tf
import numpy as np
import reconstruction_net_attention

import time




def reconstruction(inputs):

    stitched = reconstruction_net_attention.ReconstructionNetAttention(inputs)
    return stitched#, hr_stitched


