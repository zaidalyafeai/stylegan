import pickle
import numpy as np
import tensorflow as tf
import dnnlib.tflib as tflib
import runway
from runway.data_types import checkpoint, number, vector, image

fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

@runway.setup(options=[checkpoint(name='checkpoint')])
def setup(opts):
    global Gs
    tflib.init_tf()
    with open(opts['checkpoint'], 'rb') as file:
        G, D, Gs = pickle.load(file)
    return Gs


@runway.command('generate', inputs=[vector(512), number(name='truncation', min=0, max=1, default=0.8, step=0.1)], outputs=[image])
def convert(model, z, truncation):
    latents = z.reshape((1, 512))
    images = model.run(latents, None, truncation_psi=truncation, randomize_noise=False, output_transform=fmt)
    output = np.clip(images[0], 0, 255).astype(np.uint8)
    return output


if __name__ == '__main__':
    runway.run()
