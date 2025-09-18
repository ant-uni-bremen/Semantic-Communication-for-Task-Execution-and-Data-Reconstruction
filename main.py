import os
from datetime import datetime
from collections import defaultdict
gpu_num = 0
os.environ["CUDA_VISIBLE_DEVICES"] = f"{gpu_num}"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import json
import numpy as np

from SemanticModel import SemanticModel
from channel import AWGNLayer

from CNN.Decoder import CnnDecoder
from CNN.Encoder import CnnEncoder

from ViT.Decoder import ViTDecoder
from ViT.Encoder import ViTEncoder

from ResNet14.Encoder import ResNet14Encoder
from ResNet14.Decoder import ResNet14Decoder

from ResNet20.Encoder import ResNet20Encoder
from ResNet20.Decoder import ResNet20Decoder

from keras import backend,datasets
import copy
import math
from plots import SemanticPlots

def save_data(dataset,data_folder):
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    if dataset == 'cifar10':
        (x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()
    if dataset == 'cifar100':
        (x_train, y_train), (x_test, y_test) = datasets.cifar100.load_data()
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0

    np.save(os.path.join(data_folder, 'x_train.npy'), x_train)
    np.save(os.path.join(data_folder, 'y_train.npy'), y_train)
    np.save(os.path.join(data_folder, 'x_test.npy'), x_test)
    np.save(os.path.join(data_folder, 'y_test.npy'), y_test)

def load_dataset(dataset,data_folder):
    if os.path.exists(data_folder) and all(
        os.path.exists(os.path.join(data_folder, fname))
        for fname in ['x_train.npy', 'y_train.npy', 'x_test.npy', 'y_test.npy']
    ):
        x_train = np.load(os.path.join(data_folder, 'x_train.npy'))
        y_train = np.load(os.path.join(data_folder, 'y_train.npy'))
        x_test = np.load(os.path.join(data_folder, 'x_test.npy'))
        y_test = np.load(os.path.join(data_folder, 'y_test.npy'))
    else:
        print("Data files not found. Downloading and saving the data...")
        save_data(dataset,data_folder)
        x_train = np.load(os.path.join(data_folder, 'x_train.npy'))
        y_train = np.load(os.path.join(data_folder, 'y_train.npy'))
        x_test = np.load(os.path.join(data_folder, 'x_test.npy'))
        y_test = np.load(os.path.join(data_folder, 'y_test.npy'))
    
    return (x_train, y_train), (x_test, y_test)
def find_evaluation_snr_index(evaluation_snr,SNR_Range):
    for index,snr in enumerate(SNR_Range):
        if snr == evaluation_snr:
            return index
    return Exception("Evaluation SNR not in range of default SNR Range")


if __name__ == "__main__":
    SemanticModel.configure_gpu()
    try:
        with open('config.json', 'r') as config_file:
            configs = json.load(config_file)
    except Exception as e:
        print(f"Error: {str(e)}. The configuration file cannot be opened.")
        exit()

    for config_index, config in enumerate(configs):
        print(f"Processing configuration {config_index + 1}/{len(configs)}...")

        SNR_range = np.arange(config['evaluation_snr']-20,config['evaluation_snr']+20,10)
        train_snrs = config['train_snr_set']
        split_image_intos = config['split_image_intos']
        num_classes = config['num_classes']
        num_images = config['num_images']
        enc_out_dec_inps = config['enc_out_dec_inps']
        initial_learning_rate = config['initial_learning_rate']                                                     
        epochs = config['epochs']
        alphas = config['alphas']
        betas = config['betas']
        dataset = config['dataset']
        evaluation_snr_index = find_evaluation_snr_index(config['evaluation_snr'],SNR_Range=SNR_range)
        batch_size = config['batch_size']
        loss_functions = config['loss_functions']
        result_folder_Name = str(config['result_folder_Name'])
        selected_models = config['models_to_use']
        model_identification_text = config['model_identification_text'] if config.get('model_identification_text') else ''

        semanticPlot = SemanticPlots(
            models_to_use = selected_models,
            split_image_intos = split_image_intos,
            enc_out_dec_inps = enc_out_dec_inps,
            train_snrs = train_snrs,
            alphas = alphas,
            betas = betas,
            num_images = num_images,
            SNR_range = SNR_range,
            loss_functions = loss_functions,
            use_pgf = False
        )

        general_results = {
            'mse': [], 
            'psnr': [],
            'ssim':[],
            'reconstruction_image': [], 
            'class_prediction': [], 
            'accuracy': [],
            'correct_predicted_labels': [], 
            'incorrect_predicted_labels': [], 
            'history': []
        }
        
        data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"{dataset}_data")
        os.makedirs(data_folder, exist_ok=True)

        (x_train, y_train), (x_test, y_test) = load_dataset(dataset,data_folder)
        random_indices = np.sort(np.random.choice(x_test.shape[0],size=num_images,replace=False))
        input_images = x_test
        input_images_label = y_test

        current_input_images = [input_images[n] for n in random_indices]
        current_input_images_label = [input_images_label[n] for n in random_indices]
        
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        results_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results", f'{timestamp}_data_{dataset}_{epochs}_epochs_{result_folder_Name}')
        os.makedirs(results_folder, exist_ok=True)
        
        loss_function_results = {lossFunction:defaultdict(lambda:[]) for lossFunction in loss_functions}
        for loss_function in loss_functions:
            beta_results = {f"{beta}":defaultdict(lambda:[]) for beta in betas}
            for beta in betas:
                alpha_results = {f"{alpha}":defaultdict(lambda:[]) for alpha in alphas}
                for alpha in alphas:
                    bn_results = {f"{bn}":defaultdict(lambda:[]) for bn in enc_out_dec_inps}
                    for enc_out_dec_inp in enc_out_dec_inps:
                        train_snr_result = {f"{train_snr}":defaultdict(lambda:[]) for train_snr in train_snrs}
                        for train_snr in train_snrs:
                            split_results = {f"{split}":defaultdict(lambda:[]) for split in split_image_intos}
                            for split_image_into in split_image_intos:
                                results = {models:copy.deepcopy(general_results) for models in selected_models}

                                for model in selected_models:
                                    model_folder = os.path.join(results_folder, model)
                                    tflite_folder = os.path.join(model_folder, 'tflite')
                                    os.makedirs(model_folder, exist_ok=True)
                                    os.makedirs(tflite_folder, exist_ok=True)

                                    if beta == 0:
                                        loss_function = "mse"
                                        model_path = f'{model.lower()}_model_epochs_{epochs}_lr_{initial_learning_rate}_bn_{enc_out_dec_inp}_train_snr_{train_snr}_split_image_into_{split_image_into}_alpha_{alpha}_loss_function_{loss_function}_dataset_{dataset}{model_identification_text}.weights.h5'
                                        enc_path = f'{model.lower()}_encoder_epochs_{epochs}_lr_{initial_learning_rate}_bn_{enc_out_dec_inp}_train_snr_{train_snr}_split_image_into_{split_image_into}_alpha_{alpha}_loss_function_{loss_function}_dataset_{dataset}{model_identification_text}.weights.h5'
                                        dec_path = f'{model.lower()}_decoder_epochs_{epochs}_lr_{initial_learning_rate}_bn_{enc_out_dec_inp}_train_snr_{train_snr}_split_image_into_{split_image_into}_alpha_{alpha}_loss_function_{loss_function}_dataset_{dataset}{model_identification_text}.weights.h5'
                                    else:
                                        model_path = f'{model.lower()}_model_epochs_{epochs}_lr_{initial_learning_rate}_bn_{enc_out_dec_inp}_train_snr_{train_snr}_split_image_into_{split_image_into}_alpha_{alpha}_beta_{beta}_loss_function_{loss_function}_dataset_{dataset}{model_identification_text}.weights.h5'
                                        enc_path = f'{model.lower()}_encoder_epochs_{epochs}_lr_{initial_learning_rate}_bn_{enc_out_dec_inp}_train_snr_{train_snr}_split_image_into_{split_image_into}_alpha_{alpha}_beta_{beta}_loss_function_{loss_function}_dataset_{dataset}{model_identification_text}.weights.h5'
                                        dec_path = f'{model.lower()}_decoder_epochs_{epochs}_lr_{initial_learning_rate}_bn_{enc_out_dec_inp}_train_snr_{train_snr}_split_image_into_{split_image_into}_alpha_{alpha}_betaa_{beta}_loss_function_{loss_function}_dataset_{dataset}{model_identification_text}.weights.h5'
                                    if model == "CNN":
                                        encoder=CnnEncoder(split_image_into=split_image_into, enc_out_dec_inp=enc_out_dec_inp)
                                        decoder=CnnDecoder(split_image_into=split_image_into, enc_out_dec_inp=enc_out_dec_inp, num_classes=num_classes, output_channels=x_train.shape[-1])
                                        channel=AWGNLayer(snr=train_snr)

                                    elif model == "ResNet14":
                                        encoder=ResNet14Encoder(split_image_into=split_image_into, enc_out_dec_inp=enc_out_dec_inp)
                                        decoder=ResNet14Decoder(split_image_into=split_image_into, enc_out_dec_inp=enc_out_dec_inp, num_classes=num_classes, output_channels=x_train.shape[-1])
                                        channel=AWGNLayer(snr=train_snr)

                                    elif model == "ResNet20":
                                        encoder=ResNet20Encoder(split_image_into=split_image_into, enc_out_dec_inp=enc_out_dec_inp)
                                        decoder=ResNet20Decoder(split_image_into=split_image_into, enc_out_dec_inp=enc_out_dec_inp, num_classes=num_classes, output_channels=x_train.shape[-1])
                                        channel=AWGNLayer(snr=train_snr)

                                    elif model == "VIT":
                                        ### VIT parameters
                                        image_size = int(x_train.shape[1]//math.sqrt(split_image_into))
                                        patch_size = split_image_into
                                        channels =  3
                                        num_patches = (image_size // patch_size) ** 2  # e.g. (32/8)^2 = 16 patches
                                        embed_dim = 256
                                        num_layers = 16
                                        num_heads = 8
                                        mlp_dim = 512

                                        encoder=ViTEncoder(split_image_into=split_image_into, enc_out_dec_inp=enc_out_dec_inp,num_patches=num_patches,patch_size=patch_size,embed_dim=embed_dim,num_heads=num_heads,mlp_dim=mlp_dim,num_layers=num_layers,input_shape=(image_size, image_size, channels))
                                        decoder=ViTDecoder(split_image_into=split_image_into, enc_out_dec_inp=enc_out_dec_inp, num_classes=num_classes, output_channels=x_train.shape[-1])
                                        channel=AWGNLayer(snr=train_snr)
                                    else:
                                        raise Exception("Entered wrong model name in config json")       

                                    # Instantiate the model architecture first
                                    semanticModel = SemanticModel(
                                        encoder=encoder,
                                        decoder=decoder,
                                        channel=channel,
                                        x_train=x_train, 
                                        y_train=y_train,
                                        x_test=x_test, 
                                        y_test=y_test,
                                        train_snr=train_snr,
                                        epochs=epochs, 
                                        enc_out_dec_inp=enc_out_dec_inp,
                                        output_channels=x_train.shape[-1],
                                        loss_function=loss_function,
                                        alpha=alpha,
                                        beta = beta,
                                        initial_learning_rate=initial_learning_rate,
                                        num_classes=num_classes,
                                        input_shape=x_train.shape[1:],
                                        split_image_into=split_image_into,
                                        model_tflite=tflite_folder,
                                        loss_function_equal_weight=config.get("loss_function_equal_weight",False)
                                    )

                                    # ✅ Load and freeze encoder
                                    if os.path.exists(enc_path):
                                        print(f"[✓] Loading encoder weights from {enc_path}")
                                        semanticModel.encoder.load_weights(filepath=enc_path)

                                        if config.get("freeze_encoder", False):
                                            semanticModel.encoder.trainable = False
                                            print(
                                                "[🧊] Encoder is frozen. Its weights will NOT be updated during training.")
                                    else:
                                        print("[×] Encoder weights not found. Training from scratch.")

                                    # ✅ Load and freeze decoder
                                    if os.path.exists(dec_path):
                                        print(f"[✓] Loading decoder weights from {dec_path}")
                                        semanticModel.decoder.load_weights(filepath=dec_path)

                                        if config.get("freeze_decoder", False):
                                            semanticModel.decoder.trainable = False
                                            print(
                                                "[🧊] Decoder is frozen. Its weights will NOT be updated during training.")
                                    else:
                                        print("[×] Decoder weights not found. Training from scratch.")

                                    # Load full model weights if available
                                    if os.path.exists(model_path):
                                        print(f"Loading full model weights from {model_path}")
                                        semanticModel.build()
                                        semanticModel.load_weights(filepath=model_path)
                                    else:
                                        print("[×] Model weights not found. Training from scratch.")
                                        # Train (either fresh or continuing)
                                        results[model]['history'].append(semanticModel.fit(batch_size=batch_size, model=model))

                                    if config['save_model']:
                                        try:
                                            semanticModel.save_weights(filepath=model_path)
                                        except Exception as e:
                                            print(f"Error in saving model: {e}")
                                            continue
                                    if config['save_encoder']:
                                        try:
                                            semanticModel.encoder.save_weights(filepath=enc_path)
                                        except Exception as e:
                                            print(f"Error in saving encoder: {e}")
                                            continue
                                    if config['save_decoder']:
                                        try:
                                            semanticModel.decoder.save_weights(filepath=dec_path)
                                        except Exception as e:
                                            print(f"Error in saving decoder: {e}")
                                            continue

                                    # Evaluate model for different SNR values
                                    for snr in SNR_range:
                                        predicts = semanticModel.predict(x_test, batch_size=batch_size, snr=snr)
                                        outputs = semanticModel.metric(predicts, random_indices)
                                        for key in outputs.keys():
                                            results[model][key].append(outputs[key])
                                    del outputs
                                    del predicts
                                    backend.clear_session()

                                split_results[f"{split_image_into}"] = results
                                del results
                            train_snr_result[f"{train_snr}"] = split_results
                            del split_results
                        bn_results[f"{enc_out_dec_inp}"] = train_snr_result
                        del train_snr_result
                    alpha_results[f"{alpha}"] = bn_results
                    del bn_results
                beta_results[f"{beta}"] = alpha_results
                del alpha_results
            '''
            semanticPlot.plot_psnr(beta_results,results_folder,loss_function)
            semanticPlot.plot_ssim(beta_results,results_folder,loss_function)
            semanticPlot.plot_snr_accuracy_bar_plot(
                beta_results,
                dataset_size=x_test.shape[0],
                folderPath=results_folder,
                lossFunction=loss_function
            )
            '''
            loss_function_results[f"{loss_function}"] = beta_results
            del beta_results
        '''
        final plot place based on error function changes
        
        semanticPlot.psnr(
            loss_function_results,
            results_folder
        )
        semanticPlot.ssim(
            loss_function_results,
            results_folder
        )
        semanticPlot.accuracy(
            loss_function_results,
            results_folder
        )
        '''
        semanticPlot.dual_psnr_accuracy_with_different_alphas(
            loss_function_results,
            results_folder,
            evaluation_snr_index
        )
        semanticPlot.dual_psnr_ssim_with_different_betas(
            loss_function_results,
            results_folder,
            evaluation_snr_index
        )
        semanticPlot.three_ssim_psnr_accuracy_with_different_alphas(loss_function_results,
            results_folder,
            evaluation_snr_index)
        