# -*- coding: utf-8 -*-
"""
Created on Sun May 10 16:37:59 2020

@author: hona
"""
# data import
import os
import pickle
import pandas as pd
import numpy as np

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

import keras
from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, LSTM, RepeatVector
from keras.layers.core import Flatten, Dense, Dropout,Lambda
from keras.optimizers import SGD, RMSprop, Adam
from keras import objectives
from keras.utils.vis_utils import plot_model


def create_lstm_vae(input_dim,
                   timesteps,
                   batch_size,
                   intermediate_dim,
                   latent_dim,
                   epsilon_std=1.):
    """
    Create an LSTM Variational Autoencoder (VAE). Returns VAE, Encoder, Generator.
    
    # Arguments
       input_dim: int.
       timesteps: int, input timestep dimension.
       batch_size: int.
       intermediate_dim: int, output shape of LSTM.
       latent_dim: int, latent z-layer shape.
       epsilon_std: float, z-layer sigma.
       
    # References
        -  [Building Autoencoders in Keras](https://blog.keras.io/building-autoencoders-in-keras.html)
        -  [Generating sentences from a continuous space](https://arxiv.org/abs/1511.06349)
    """
    x = Input(shape=(timesteps, input_dim,))
    
    #LSTM encoding
    h = LSTM(intermediate_dim)(x)
    
    # VAE Z layer
    z_mean = Dense(latent_dim)(h)
    z_log_sigma = Dense(latent_dim)(h)
    
    def sampling(args):
        z_mean, z_log_sigma = args
        epsilon = K.random_normal(shape=(batch_size, latent_dim),
                                 mean=0., stddev=epsilon_std)
        return z_mean + z_log_sigma * epsilon
    
    # note that 'output_shape' isn't necessary with the TensorFlow backend
    # so you could write 'Lambda(sampling)([z_mean, z_log_sigma])'
    z = Lambda(sampling)([z_mean, z_log_sigma])
    
    # decoded LSTM layer
    decoder_h = LSTM(intermediate_dim, return_sequences=True)
    decoder_mean = LSTM(input_dim, return_sequences=True)
    
    h_decoded = RepeatVector(timesteps)(z)
    h_decoded = decoder_h(h_decoded)
    
    # decoded layer
    x_decoded_mean = decoder_mean(h_decoded)
    
    # end-to-end autoencoder
    vae = Model(x, x_decoded_mean)
    
    # encoder, from inputs to latent space
    encoder = Model(x, z_mean)
    
    # generator, from latent space to recconstructed inputs
    decoder_input = Input(shape=(latent_dim,))
    
    _h_decoded = RepeatVector(timesteps)(decoder_input)
    _h_decoded = decoder_h(_h_decoded)
    
    _x_decoded_mean = decoder_mean(_h_decoded)
    generator = Model(decoder_input, _x_decoded_mean)
    
    def vae_loss(x, x_decoded_mean):
        xent_loss = objectives.mse(x, x_decoded_mean)
        kl_loss = -0.5 * K.mean(1 + z_log_sigma - K.square(z_mean) - K.exp(z_log_sigma))
        loss = xent_loss + kl_loss
        return loss
    
    vae.compile(optimizer='rmsprop', loss=vae_loss)
    #plot_model(vae, to_file='model_plot.png', show_shapes=True, show_layer_names=True)
    
    
    return vae, encoder, generator
    
    
    
def get_driver_data():
    filename = './fi_pickle/dyanamic_drv_right.pickle'
    if os.path.isfile(filename):
        with open(filename, mode='rb') as f:
            drv_dynamic = pickle.load(f)
    
    return drv_dynamic


if __name__ == "__main__":
    
    all_drv_dynamic = get_driver_data()
    timesteps = 20
    epochs = 100
    for i_driver in range(1):
        drv_dynamic = np.array(all_drv_dynamic[i_driver])
        drv_dynamic_v = drv_dynamic[:,2:4]
        print(drv_dynamic_v.shape)
        
        dataX = []
        for i in range(len(drv_dynamic_v) - timesteps - 1):
            x = drv_dynamic_v[i:(i+timesteps), :]
            dataX.append(x)
        
        x = np.array(dataX)
        print(x.shape)
        input_dim = x.shape[-1] # 13
        print(input_dim)
        timesteps = x.shape[1] # 3
        print(timesteps)
        batch_size = 1

        vae, enc, gen = create_lstm_vae(input_dim, 
            timesteps=timesteps, 
            batch_size=batch_size, 
            intermediate_dim=32,
            latent_dim=100,
            epsilon_std=1.)

        vae.fit(x, x, epochs=epochs)

        preds = vae.predict(x, batch_size=batch_size)
       
        # pick a column to plot.
        print("[plotting...]")
        print("x: %s, preds: %s" % (x.shape, preds.shape))
        plt.plot(x[:,0,0], label='True')
        plt.plot(preds[:,0,0], label='Predicted')
        plt.ylim([0, 130])             
        plt.grid()
        plt.title("Driver id: m9600A")
        plt.xlabel("Frame # [10hz]")
        plt.ylabel("Velocity [km/h]")                
        plt.tight_layout()
        plt.savefig('vae_velocity_m9600A.png', format='png')
        plt.legend()
        #plt.show()
        
        
        
        #plot latent space
        driver_1_enc = enc.predict(x, batch_size=batch_size)
        plt.figure(figsize=(6, 6))
        plt.title("Driver id: m9600A")
        plt.grid()
        plt.scatter(driver_1_enc[:, 0], driver_1_enc[:, 1],c="b", alpha=0.5, marker=">")
        plt.savefig('latent_space_m9600A.png', format='png')
        plt.colorbar()
        plt.legend()
        #plt.show()
        
        with open('latent_space_m9600A.pickle', mode='wb') as fo:
            pickle.dump(driver_1_enc, fo)
        
        
    for i_driver in range(1):
        drv_dynamic = np.array(all_drv_dynamic[i_driver+1])
        drv_dynamic_v = drv_dynamic[:,2:4]
        print(drv_dynamic_v.shape)
        
        dataX = []
        for i in range(len(drv_dynamic_v) - timesteps - 1):
            x = drv_dynamic_v[i:(i+timesteps), :]
            dataX.append(x)
        
        x = np.array(dataX)
        print(x.shape)
        input_dim = x.shape[-1] # 13
        print(input_dim)
        timesteps = x.shape[1] # 3
        print(timesteps)
        batch_size = 1

        vae, enc, gen = create_lstm_vae(input_dim, 
            timesteps=timesteps, 
            batch_size=batch_size, 
            intermediate_dim=32,
            latent_dim=100,
            epsilon_std=1.)

        vae.fit(x, x, epochs=epochs)

        preds = vae.predict(x, batch_size=batch_size)
       
        # pick a column to plot.
        print("[plotting...]")
        print("x: %s, preds: %s" % (x.shape, preds.shape))
        plt.plot(x[:,0,0], label='True')
        plt.plot(preds[:,0,0], label='Predicted')
        plt.ylim([0, 130])             
        plt.grid()
        plt.title("Driver id: m9601A")
        plt.xlabel("Frame # [10hz]")
        plt.ylabel("Velocity [km/h]")                
        plt.tight_layout()
        plt.savefig('vae_velocity_m9601A.png', format='png')
        plt.legend()
        #plt.show()
        
        
        
        #plot latent space
        driver_2_enc = enc.predict(x, batch_size=batch_size)
        plt.figure(figsize=(6, 6))
        plt.title("Driver id: m9601A")
        plt.grid()
        plt.scatter(driver_2_enc[:, 0], driver_2_enc[:, 1],c="m", alpha=0.5, marker="*")
        plt.savefig('latent_space_m9601A.png', format='png')
        plt.colorbar()
        plt.legend()
        #plt.show()
        
        with open('latent_space_m9601A.pickle', mode='wb') as fo:
            pickle.dump(driver_2_enc, fo)
        
        
        fig = plt.figure()
        ax1 = fig.add_subplot(111)

        ax1.scatter(driver_1_enc[:, 0], driver_1_enc[:, 1],c="b", alpha=0.5, marker=">", label='m9600A')
        ax1.scatter(driver_2_enc[:, 0], driver_2_enc[:, 1],c="m", alpha=0.5, marker="*", label='m9601A')
        plt.grid()
        plt.legend()
        plt.rcParams.update({'font.size': 15})
        plt.savefig('vae_drivers.png', format='png')
        #plt.show()