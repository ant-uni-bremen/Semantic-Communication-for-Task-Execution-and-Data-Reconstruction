from keras import Model,layers,optimizers,losses,callbacks,Input
import tensorflow as tf
from utils import splitInputImages,NormalizationLayer,CustomImageAugmentation
from ConstMultiplierLayer import ConstMultiplierLayer
import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import accuracy_score
from plots import tikzplotlib_fix_ncols 
import tikzplotlib as tikz

class SemanticModel(Model):
    def __init__(self,encoder,decoder,channel,x_train,y_train,x_test,y_test,train_snr,split_image_into,enc_out_dec_inp,num_classes,output_channels,model_tflite,alpha,beta,epochs,input_shape=(32,32,3),loss_function='mse',initial_learning_rate = 1e-4,loss_function_equal_weight=False,*args, **kwargs):
        super().__init__(*args, **kwargs)
        self.x_train = x_train
        self.x_test = x_test
        self.y_train = y_train
        self.y_test = y_test
        self._encoder = encoder
        self._decoder = decoder
        self.split_image_into = split_image_into
        self.enc_out_dec_inp = enc_out_dec_inp
        self.num_classes = num_classes
        self.output_channels = output_channels
        self.model_tflite = model_tflite
        self.epochs = epochs
        self.alpha = alpha
        self.loss_function_equal_weight = loss_function_equal_weight
        self.beta = beta
        self.loss_function = loss_function
        self._channel = channel
        self.train_snr = train_snr
        self.mse_function = losses.MeanSquaredError()
        self.split_images = splitInputImages(split_image_into=self.split_image_into)
        self.normalizationLayer = NormalizationLayer()
        self.constantDenormalizationLayer =  ConstMultiplierLayer()
        self.initial_learning_rate = initial_learning_rate
        self.input_shape = input_shape
        self._model = self._semanticModel()
    
    def _semanticModel(self):
        inputs = Input(shape=self.input_shape)
        
        # Step 1: Split input images into patches
        splits = self.split_images(inputs)  # (batch_size, split_image_into, H/s, W/s, C)
        splits_position = layers.Lambda(lambda x: tf.transpose(x, perm=[1, 0, 2, 3, 4]))(splits)

        # Step 2: Encode each patch individually
        encoded_signals = []
        for i in range(self.split_image_into):
            encoded = self._encoder(splits_position[i])  # Shape: (batch_size, enc_out_dec_inp // split_image_into)
            normalized = self.normalizationLayer(encoded)
            channeled = self._channel(normalized)
            encoded_signals.append(channeled)

        # Step 3: Concatenate all encoded parts to get final encoding for the image
        merged_encoding = layers.Concatenate(axis=-1)(encoded_signals)  # Shape: (batch_size, enc_out_dec_inp)

        # Step 4: Decode once per image
        decoded_output = self._decoder(merged_encoding)  # Should return dict with 'reconstructed_image' & 'classification_output'

        # Step 5: Denormalize the reconstructed image
        denormalized_img = self.constantDenormalizationLayer(decoded_output['reconstructed_image'])

        return Model(
            inputs=inputs,
            outputs={
                'reconstructed_image': denormalized_img,
                'classification_output': decoded_output['classification_output']
            }
        )

    def build(self,input_shape=(None,32,32,3)):
        """
        Build the inner model using the provided input_shape.
        If input_shape is None, use self.input_shape with an added batch dimension.
        """
        if input_shape is None:
            input_shape = (None,) + self.input_shape
        self._model.build(input_shape)
        super().build(input_shape)
                            
    def call(self,input):
        return self._model(input)
    
    def reconstructionImageLoss(self):
        def loss(y_true,y_pred):
            if self.loss_function == 'mse':
                return self.mse_function(y_true,y_pred)
            if self.loss_function == 'ssim':
                return self.beta * (1-tf.reduce_mean(tf.image.ssim(y_true,y_pred,max_val=1.0)))
            if self.loss_function == 'mse_ssim':
                return (self.beta * (1 - tf.reduce_mean(tf.image.ssim(y_true, y_pred, max_val=1.0)))) + ((1 - self.beta) * self.mse_function(y_true, y_pred))
        return loss

    def fit(self,graph_plot=True,batch_size=32,model=None):
        if model==None:
            ## ResNet Model
            lr_schedule = optimizers.schedules.ExponentialDecay(
            initial_learning_rate=self.initial_learning_rate,
            decay_steps=1535,
            decay_rate=0.98,
            staircase=False
            )
            optimizer = optimizers.Nadam(
                learning_rate=lr_schedule,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                weight_decay=None,
                clipnorm=None,
                clipvalue=None,
                global_clipnorm=None,
                use_ema=False,
                ema_momentum=0.99,
                ema_overwrite_frequency=None,
                loss_scale_factor=None,
                gradient_accumulation_steps=None,
                name="nadam",
            )
            selective_callbacks = []
        else:
            ## CNN Model
            reduceLRonPlateau = callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.1,
                patience=10,
                verbose=0,
                mode='auto',
                min_delta=0.01,
                cooldown=0,
                min_lr=0.0
            )
            
            optimizer = optimizers.Nadam(
                learning_rate=self.initial_learning_rate,
                beta_1=0.9,
                beta_2=0.999,
                epsilon=1e-07,
                weight_decay=None,
                clipnorm=None,
                clipvalue=None,
                global_clipnorm=None,
                use_ema=False,
                ema_momentum=0.99,
                ema_overwrite_frequency=None,
                loss_scale_factor=None,
                gradient_accumulation_steps=None,
                name="nadam"
            )
            selective_callbacks = [reduceLRonPlateau]
        if self.loss_function_equal_weight:
            print('No loss functions weight are present')
            weighted_around_loss_functions = {
                'reconstructed_image': 1,
                'classification_output': 1
            }
        else:
            print('loss functions "alpha" weights are present')
            weighted_around_loss_functions = {
                'reconstructed_image': self.alpha,
                'classification_output': (1-self.alpha)
            }

        self.compile(optimizer=optimizer,
            loss={
                'reconstructed_image': self.reconstructionImageLoss(),
                'classification_output': 'sparse_categorical_crossentropy'
            },
            loss_weights=weighted_around_loss_functions,
            metrics={'reconstructed_image': 'mse','classification_output': 'accuracy'}
        )
        
        history = super().fit(
            self.x_train,
                {
                    'reconstructed_image': self.x_train,  # Full images for reconstruction
                    'classification_output': self.y_train,  # Ground truth labels for classification
                },
            epochs=self.epochs,
            batch_size=batch_size,
            shuffle=True,
            validation_data=(
                self.x_test,
                {
                    'reconstructed_image': self.x_test,  # Full images for reconstruction
                    'classification_output': self.y_test,  # Ground truth labels for classification
                }
            ),
            callbacks=selective_callbacks
        )
            
        if graph_plot:
                self._plot_training_results(history=history)
        return history.history
    
    def _plot_training_results(self,history):
        plt.figure()
        plt.plot(history.history['classification_output_accuracy'], label='classification_output_accuracy')
        plt.plot(history.history['val_classification_output_accuracy'], label='val_classification_output_accuracy')
        plt.xlabel('epochs')
        plt.ylabel('accuracy')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.model_tflite,f'ClassificationOutputAccuracy_split_image_{self.split_image_into}_epochs_{self.epochs}_alpha_{self.alpha}_trSnr_{self.train_snr}_bN_{self.enc_out_dec_inp}_{self.loss_function}.png'))
        #tikzplotlib_fix_ncols(plt.gcf())
        #tikz.save(os.path.join(self.model_tflite,f'ClassificationOutputAccuracy_split_image_{self.split_image_into}_epochs_{self.epochs}_alpha_{self.alpha}_trSnr_{self.train_snr}_bN_{self.enc_out_dec_inp}.tex'))
        plt.close()
        # Predict using the model

        plt.figure()
        plt.plot(history.history['loss'], label='loss')
        plt.plot(history.history['val_loss'], label='val_loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.model_tflite,f'TotalLoss_split_image_{self.split_image_into}_epochs_{self.epochs}_alpha_{self.alpha}_trSnr_{self.train_snr}_bN_{self.enc_out_dec_inp}_{self.loss_function}.png'))
        #tikzplotlib_fix_ncols(plt.gcf())
        #tikz.save(os.path.join(self.model_tflite,f'TotalLoss_split_image_{self.split_image_into}_epochs_{self.epochs}_alpha_{self.alpha}_trSnr_{self.train_snr}_bN_{self.enc_out_dec_inp}.tex'))
        plt.close()
        # Predict using the model

        plt.figure()
        plt.plot(history.history['classification_output_loss'], label='classification_output_loss')
        plt.plot(history.history['val_classification_output_loss'], label='val_classification_output_loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.model_tflite,f'ClassificationOutputLoss_split_image_{self.split_image_into}_epochs_{self.epochs}_alpha_{self.alpha}_trSnr_{self.train_snr}_bN_{self.enc_out_dec_inp}_{self.loss_function}.png'))
        #tikzplotlib_fix_ncols(plt.gcf())
        #tikz.save(os.path.join(self.model_tflite,f'ClassificationOutputLoss_split_image_{self.split_image_into}_epochs_{self.epochs}_alpha_{self.alpha}_trSnr_{self.train_snr}_bN_{self.enc_out_dec_inp}.tex'))
        plt.close()
        # Predict using the model

        plt.figure()
        plt.plot(history.history['reconstructed_image_loss'], label='reconstructed_image_loss')
        plt.plot(history.history['val_reconstructed_image_loss'], label='val_reconstructed_image_loss')
        plt.xlabel('epochs')
        plt.ylabel('loss')
        plt.title(f'loss_vs_val_loss split_image - Split Image: {self.split_image_into**2} - epochs_{self.epochs} - alpha_{self.alpha}')
        plt.legend(loc='best')
        plt.savefig(os.path.join(self.model_tflite,f'ReconstructedImageLossSplitImage_{self.split_image_into}_epochs_{self.epochs}_alpha_{self.alpha}_trSnr_{self.train_snr}_bN_{self.enc_out_dec_inp}_{self.loss_function}.png'))
        #tikzplotlib_fix_ncols(plt.gcf())
        #tikz.save(os.path.join(self.model_tflite,f'ReconstructedImageLossSplitImage_{self.split_image_into}_epochs_{self.epochs}_alpha_{self.alpha}_trSnr_{self.train_snr}_bN_{self.enc_out_dec_inp}.tex'))
        plt.close()

    @staticmethod
    def configure_gpu():
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("GPU memory growth enabled.")
            except RuntimeError as e:
                print(f"GPU memory growth configuration failed: {e}")
    
    def update_snr(self, snr):
        self._channel.update_snr(snr)

    def predict(self, input, snr=None, batch_size=32):
        if snr is not None:
            self.update_snr(snr)  # Dynamically update the SNR
        return self._model.predict(input, batch_size=batch_size)

    def metric(self, predicted_test, random_indices):
        pred = []
        reconst_images = []
        classification_output = predicted_test['classification_output']
        reconstruction_image_output = predicted_test['reconstructed_image']
        batch_size = min(len(self.y_test), classification_output.shape[0])

        # Use the true labels directly if they are not one-hot encoded
        true_labels_batch = self.y_test[:batch_size].flatten()  # Assuming self.y_test is 1D

        # Convert predicted probabilities to class labels
        pred_class = np.argmax(classification_output[:batch_size], axis=1)

        # Collect predictions for random indices
        for idx in range(len(pred_class)):
            if idx in random_indices:
                pred.append(pred_class[idx])
        
        # Collect predictions for random indices
        for id in range(len(reconstruction_image_output)):
            if id in random_indices:
                reconst_images.append(reconstruction_image_output[id])

        # Compute accuracy
        acc = accuracy_score(true_labels_batch, pred_class)

        return { 
            'mse': tf.reduce_mean(self.mse_function(self.x_test, predicted_test['reconstructed_image'])).numpy(),
            'psnr': tf.reduce_mean(tf.image.psnr(self.x_test, predicted_test['reconstructed_image'], max_val=1.0)).numpy(),
            'ssim': tf.reduce_mean(tf.image.ssim(self.x_test, predicted_test['reconstructed_image'], max_val=1.0)).numpy(),
            'correct_predicted_labels': np.sum(pred_class == true_labels_batch),
            'incorrect_predicted_labels': np.sum(pred_class != true_labels_batch),
            'class_prediction': pred,
            'accuracy': acc,
            'reconstruction_image': reconst_images
        }

    def get_config(self):
        config = super(SemanticModel, self).get_config()
        config.update({
            'x_train':self.x_train,
            'y_train':self.y_train,
            'x_test':self.x_test,
            'y_test':self.y_test, 
            'encoder':self._encoder.get_config(),
            'decoder':self._decoder.get_config(),
            'split_image_into':self.split_image_into,
            'enc_out_dec_inp':self.enc_out_dec_inp,
            'num_classes':self.num_classes,
            'output_channels':self.output_channels,
            'model_tflite':self.model_tflite,
            'epochs':self.epochs,
            'learning_rate':self.initial_learning_rate,
            'alpha':self.alpha,
            'train_snr':self.train_snr,
            'channel':self._channel.get_config(),
        })
        return config
    
    @classmethod
    def from_config(cls, config):
        # Reconstruct the encoder from the configuration
        return cls(
            x_train = config['x_train'],
            y_train = config['y_train'],
            x_test = config['x_test'],
            y_test = config['y_test'],
            split_image_into = config['split_image_into'],
            enc_out_dec_inp = config['enc_out_dec_inp'],
            num_classes = config['num_classes'],
            output_channels = config['output_channels'],
            model_tflite = config['model_tflite'],
            epochs = config['epochs'],
            alpha = config['alpha'],
            train_snr = config['train_snr'],
            awgn_layer = config['awgn_layer'],
            encoder = config['encoder'],
            decoder = config['decoder'],
            initial_learning_rate = config['learning_rate'],
            channel = config['channel']
        )

    @property
    def encoder(self):
        return self._encoder

    @property
    def decoder(self):
        return self._decoder
