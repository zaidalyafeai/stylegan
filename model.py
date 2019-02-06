import pickle
import numpy as np
import tensorflow as tf
import PIL.Image
import random
import dnnlib
import dnnlib.tflib as tflib
import config
from runway import RunwayModel

stylegan = RunwayModel()

fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)


@stylegan.setup
def setup(alpha=0.5):
    global Gs
    tflib.init_tf()
    model = 'checkpoints/karras2019stylegan-ffhq-1024x1024.pkl'
    print("open model %s" % model)
    with open(model, 'rb') as file:
        G, D, Gs = pickle.load(file)
    return Gs


@stylegan.command('convert', inputs={'z': 'vector'}, outputs={'output': 'image'})
def convert(Gs, inp):
    truncation = 0.5
    latents = np.array(inp['z']).reshape((1, 512))  # np.random.RandomState(1000).randn(1, *Gs.input_shapes[0][1:])
    #labels = np.zeros([latents.shape[0]] + Gs.input_shapes[1][1:])
    images = Gs.run(latents, None, truncation_psi=truncation, randomize_noise=False, output_transform=fmt)
    output = np.clip(images[0], 0, 255).astype(np.uint8)
    return dict(output=output)


if __name__ == '__main__':
    stylegan.run()
