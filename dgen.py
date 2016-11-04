from __future__ import division
import os
import time
from glob import glob
import tensorflow as tf
import numpy as np
import json
from ops import *
from utils import *

class DGEN(object):
    def __init__(self, sess,
                 batch_size=1, sample_length=1024, sample_rate=8000,
                 z_dim=100, gf_dim=64,
                 c_dim=1, 
                 audio_params=None):
        """

        Args:
            sess: TensorFlow session
            batch_size: The size of batch. Should be specified before training.
            sample_length: (optional) The resolution in pixels of the images. [64]
            y_dim: (optional) Dimension of dim for y. [None]
            z_dim: (optional) Dimension of dim for Z. [100]
            gf_dim: (optional) Dimension of gen filters in first conv layer. [64]
            df_dim: (optional) Dimension of discrim filters in first conv layer. [64]
            gfc_dim: (optional) Dimension of gen units for for fully connected layer. [1024]
            dfc_dim: (optional) Dimension of discrim units for fully connected layer. [1024]
            c_dim: (optional) Dimension of image color. For grayscale input, set to 1. [3]
        """
        self.sess = sess
        self.batch_size = batch_size
        self.sample_length = sample_length
        self.output_length = sample_length
        self.sample_rate = sample_rate
        self.z_dim = z_dim
        self.c_dim = c_dim

        # batch normalization : deals with poor initialization helps gradient flow

        self.g_bn0 = batch_norm(name='g_bn0')
        self.g_bn1 = batch_norm(name='g_bn1')
        self.g_bn2 = batch_norm(name='g_bn2')
        self.g_bn3 = batch_norm(name='g_bn3')
        self.g_bn4 = batch_norm(name='g_bn4')

        if audio_params:
            with open(audio_params, 'r') as f:
                self.audio_params = json.load(f)
            self.gf_dim = self.audio_params['gf_dim']
        else:
            self.gf_dim = gf_dim


        self.build_model()

    def build_model(self):


        self.z = tf.placeholder(tf.float32, [None, self.z_dim],
                                name='z')

        self.z_sum = tf.histogram_summary("z", self.z)


        self.G = self.generator(self.z)

        self.sampler = self.sampler(self.z)

        self.G_sum = tf.audio_summary("G", self.G, sample_rate=self.sample_rate)

        t_vars = tf.trainable_variables()
        self.g_vars = [var for var in t_vars if 'g_' in var.name]

        self.saver = tf.train.Saver()
        
    def g_loss(self, g_loss):
        self.g_loss = g_loss
        self.g_loss_sum = tf.scalar_summary("g_loss", self.g_loss)

    def generate(self):
        '''generate samples from trained model'''
        #self.writer = tf.train.SummaryWriter(config.out_dir+"/logs", self.sess.graph)
        #batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
         #                       .astype(np.float32)
        batch_z = np.random.uniform(-1, 1, [self.batch_size, self.z_dim]).astype(np.float32)
        samples = self.sess.run(self.G, feed_dict={self.z: batch_z})
        return samples



    def train(self, config):

        g_optim = tf.train.AdamOptimizer(config.learning_rate, beta1=config.beta1) \
                          .minimize(self.g_loss, var_list=self.g_vars)
        #import IPython; IPython.embed()
        init = tf.initialize_all_variables()
        self.sess.run(init)


        self.g_sum = tf.merge_summary([self.z_sum, 
            self.G_sum, self.g_loss_sum])

        self.writer = tf.train.SummaryWriter(config.out_dir+"/logs", self.sess.graph)
        sample_z = np.random.uniform(-1, 1, size=(self.batch_size , self.z_dim))

        counter = 1
        start_time = time.time()

        try:
            for epoch in range(config.epoch):

                for idx in range(0, batch_idxs):

                    batch_z = np.random.uniform(-1, 1, [config.batch_size, self.z_dim]) \
                                .astype(np.float32)
                    #G
                    if config.dataset == 'wav':

                        # Update G network run_g times
                        for i in range(self.run_g):
                            _, summary_str = self.sess.run([g_optim, self.g_sum],
                                feed_dict={ self.z: batch_z })

                        errG = self.g_loss.eval({self.z: batch_z})
                        #G average over batch


                    counter += 1
                    print("Epoch: [%2d] [%4d/%4d] time: %4.4f, g_loss: %.8f" \
                        % (epoch+1, idx+1, batch_idxs,
                            time.time() - start_time, errG))

                    if np.mod(counter, config.save_every) == 1:
                        #G
                        if config.dataset == 'wav':
                            # samples, d_loss, g_loss = self.sess.run(
                            #     [self.sampler, self.d_loss, self.g_loss],
                            #     feed_dict={self.z: sample_z, self.images: sample_images.eval()}
                            # )
                            samples, g_loss = self.sess.run(
                                [self.sampler, self.g_loss],
                                feed_dict={self.z: batch_z}
                            )
                            #import IPython; IPython.embed()
                        # Saving samples
                        if config.dataset == 'wav':
                            im_title = "g_loss: %.5f" % (g_loss)
                            file_str = '{:02d}_{:04d}'.format(epoch, idx)
                            save_waveform(samples,config.out_dir+'/samples/train_'+file_str, title=im_title)
                            im_sum = get_im_summary(samples, title=file_str+im_title)
                            summary_str = self.sess.run(im_sum)
                            self.writer.add_summary(summary_str, counter)
                            
                            save_audios(samples[0], config.out_dir+'/samples/train_'+file_str+'.wav', 
                                format='.wav', sample_rate=self.sample_rate)
                        print("[Sample] g_loss: %.8f" % ( g_loss))

                    if np.mod(counter, 500) == 2:
                        self.save(config.out_dir+'/checkpoint', counter)
        #G
        except KeyboardInterrupt:
            # Introduce a line break after ^C is displayed so save message
            # is on its own line.
            print()


    def generator(self, z, y=None):

        s = self.output_length
        s2, s4, s8, s16, s32 = int(s/2/2), int(s/4/4), int(s/8/8), int(s/16/16), int(s/32/32)

        # project `z` and reshape
        self.z_, self.h0_w, self.h0_b = linear(z, self.gf_dim*16*s32, 'g_h0_lin', with_w=True)

        self.h0 = tf.reshape(self.z_, [-1, s32, self.gf_dim * 16])
        h0 = tf.nn.relu(self.g_bn0(self.h0))

        self.h1, self.h1_w, self.h1_b = deconv1d(h0, 
            [self.batch_size, s16, self.gf_dim*8], name='g_h1', with_w=True)
        h1 = tf.nn.relu(self.g_bn1(self.h1))

        h2, self.h2_w, self.h2_b = deconv1d(h1,
            [self.batch_size, s8, self.gf_dim*4], name='g_h2', with_w=True)
        h2 = tf.nn.relu(self.g_bn2(h2))

        h3, self.h3_w, self.h3_b = deconv1d(h2,
            [self.batch_size, s4, self.gf_dim*2], name='g_h3', with_w=True)
        h3 = tf.nn.relu(self.g_bn3(h3))

        h4, self.h4_w, self.h4_b = deconv1d(h3,
            [self.batch_size, s2, self.gf_dim], name='g_h4', with_w=True)
        h4 = tf.nn.relu(self.g_bn4(h4))

        h5, self.h5_w, self.h5_b = deconv1d(h4,
            [self.batch_size, s, self.c_dim], name='g_h5', with_w=True)

        return tf.nn.tanh(h5)

    def sampler(self, z, y=None):
        tf.get_variable_scope().reuse_variables()

        s = self.output_length
        s2, s4, s8, s16, s32 = int(s/2/2), int(s/4/4), int(s/8/8), int(s/16/16), int(s/32/32)

        h0 = tf.reshape(linear(z, self.gf_dim*16*s32, 'g_h0_lin'),
                        [-1, s32, self.gf_dim * 16])
        h0 = tf.nn.relu(self.g_bn0(h0, train=False))

        h1 = deconv1d(h0, [self.batch_size, s16, self.gf_dim*8], name='g_h1')
        h1 = tf.nn.relu(self.g_bn1(h1, train=False))

        h2 = deconv1d(h1, [self.batch_size, s8, self.gf_dim*4], name='g_h2')
        h2 = tf.nn.relu(self.g_bn2(h2, train=False))

        h3 = deconv1d(h2, [self.batch_size, s4, self.gf_dim*2], name='g_h3')
        h3 = tf.nn.relu(self.g_bn3(h3, train=False))

        h4 = deconv1d(h3, [self.batch_size, s2, self.gf_dim*1], name='g_h4')
        h4 = tf.nn.relu(self.g_bn4(h4, train=False))

        h5 = deconv1d(h4, [self.batch_size, s, self.c_dim], name='g_h5')

        return tf.nn.tanh(h5)



    def save(self, checkpoint_dir, step):
        model_name = "DCGAN.model"
        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_length)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess,
                        os.path.join(checkpoint_dir, model_name),
                        global_step=step)


    def load(self, checkpoint_dir):
        print(" [*] Reading checkpoints...")

        model_dir = "%s_%s_%s" % (self.dataset_name, self.batch_size, self.output_length)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False
