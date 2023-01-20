from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm

import tensorflow
import numpy as np
import math
import pandas as pd
import datetime as dt

from skimage.transform import resize

from data_generators import brownian_data_generator, vf_data_generator, random_data_generator
from train_ae import prepare_optimizer_object
from models import make_forward_model
from world_models_vae_arch import build_vae_world_model
now=dt.datetime.now()
dt_string=now.strftime("%Y_%m_%d")
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.activations import relu, tanh
from keras.initializers import random_normal
import tensorflow as tf

from utils import load_parameters

from keras_neural_network import KerasNN
from scripts_pruebas.graficas_pruebecillas import graficar


def resize_images(images, dims=(8, 8, 1)):
    # print("imgm ax: ", images.max())
    resized = np.zeros((len(images), dims[0], dims[1], dims[2]), dtype=float)
    for i in range(len(images)):
        # print(images[i].shape)
        if dims[2] == 1:
            # tmp = imresize(images[i], size=(dims[0], dims[1]), interp='nearest') / 255.0
            tmp = resize(images[i], output_shape=(dims[0], dims[1]))  # / 255.0
        else:
            # tmp = imresize(images[i][:, :, 0], size=(dims[0], dims[1]), interp='bilinear') / 255.0
            tmp = resize(images[i][:, :, 0], output_shape=(dims[0], dims[1]))  # / 255.0
        #    imsave('tmp/test_{}.png'.format(i), tmp)
        if dims[2] == 1:
            resized[i][:, :, 0] = (tmp[:, :, 0]).astype(float)
        else:
            resized[i][:, :, 0] = (tmp[:, :]).astype(float)
            resized[i][:, :, 1] = (tmp[:, :]).astype(float)
            resized[i][:, :, 2] = (tmp[:, :]).astype(float)
    # exit()
    # print("imgm ax: ", resized.max(), resized.min())
    return resized


def divisorGenerator(n):
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            yield i
            if i * i != n:
                large_divisors.append(n / i)
    for divisor in reversed(large_divisors):
        yield divisor


def check_latent_size(latent_size):
    if not type(latent_size) == int:
        _, lat_h, lat_w, lat_ch = latent_size
    else:
        if int(np.sqrt(latent_size)) ** 2 == latent_size:
            lat_h, lat_w = int(np.sqrt(latent_size)), int(np.sqrt(latent_size))
            lat_ch = 1
        else:
            lat_ch = 1
            if latent_size % 3 == 0:
                lat_ch = 3
            tmp = list(divisorGenerator(latent_size // lat_ch))
            lat_h = int(tmp[len(tmp) // 2])
            lat_w = int(latent_size // (lat_h * lat_ch))
    return lat_h, lat_w, lat_ch


def load_param_general(parameters):
    do_train = eval(parameters["general"]["do_train"])
    do_test = eval(parameters["general"]["do_test"])
    interactive = eval(parameters["general"]["interactive"])
    include_forward_model = eval(parameters["general"]["include_forward_model"])
    train_only_forward = eval(parameters["general"]["train_only_forward"])
    return do_train, do_test, interactive, include_forward_model, train_only_forward


def load_param_dataset(parameters):
    img_shape = eval(parameters["dataset"]["img_shape"])
    return img_shape


def load_param_synthetic(parameters):
    size_factor = float(parameters["synthetic"]["size_factor"])
    obj_attention = float(parameters["synthetic"]["obj_attention"])
    back_attention = float(parameters["synthetic"]["back_attention"])
    return size_factor, obj_attention, back_attention


def load_param_hyperparam(parameters):
    num_epochs = int(parameters["hyperparam"]["num_epochs"])
    batch_size = int(parameters["hyperparam"]["batch_size"])
    latent_size = int(parameters["hyperparam"]["latent_size"])
    conv_layers = int(parameters["hyperparam"]["conv_layers"])
    num_filters = int(parameters["hyperparam"]["num_filters"])
    kernel_size = int(parameters["hyperparam"]["kernel_size"])
    kernel_mult = int(parameters["hyperparam"]["kernel_mult"])
    residual_forward = eval(parameters["hyperparam"]["residual_forward"])
    train_without_ae = eval(parameters["hyperparam"]["train_without_ae"])
    loss = (parameters["hyperparam"]["loss"])
    opt = (parameters["hyperparam"]["opt"])
    model_label = (parameters["hyperparam"]["model_label"])
    return num_epochs, batch_size, latent_size, conv_layers, num_filters, kernel_size, kernel_mult, residual_forward, train_without_ae, loss, opt, model_label


if __name__ == "__main__":
    # Image generator config
    dir_with_src_images = Path('data/generated/')  # _simple
    base_image = 'median_image.png'  # 'median_image.png'
    object_images = ['circle-red.png', 'robo-green.png']  # circle in the first place, as robo can be drawn over it
    parameters_filepath = "config.ini"
    parameters = load_parameters(parameters_filepath)

    do_train, do_test, interactive, include_forward_model, train_only_forward = load_param_general(parameters)
    img_shape = load_param_dataset(parameters)
    size_factor, obj_attention, back_attention = load_param_synthetic(parameters)
    num_epochs, batch_size, latent_size, conv_layers, num_filters, kernel_size, kernel_mult, residual_forward, train_without_ae, loss, opt, model_label = load_param_hyperparam(
        parameters)


    # Load VAE and FM models:
    latent_space_manual = 64
    experiment_label = 'simple_vaewm.osize-3.0.oatt-0.8.e50.bs32.lat'+str(latent_space_manual)+'.c4.opt-adamw.loss-wmse'
    autoencoder, encoder, decoder_mu_log_var, decoder, latent_shape = build_vae_world_model(
        img_shape=img_shape, latent_size=latent_size,
        opt=opt, loss=loss,  # batch_size=batch_size,
        conv_layers=conv_layers, initial_filters=num_filters)
    # autoencoder.load_weights('trained_models/{}.h5'.format(experiment_label), by_name=True)
    autoencoder.load_weights('trained_models/{}.h5'.format(experiment_label), by_name=True)

    forward_model = make_forward_model([latent_size], latent_size, learn_only_difference=residual_forward)
    forward_model.compile(loss='mse', optimizer=prepare_optimizer_object('adam', 0.001), metrics=['mse'])
    # forward_model.load_weights(
    #     'trained_models/forward_model_{}_{}_corregido.h5'.format("diff" if residual_forward else "nodiff", experiment_label))
    forward_model.load_weights(
        'trained_models/forward_model_{}_{}_arqRich6.h5'.format("diff" if residual_forward else "nodiff",
                                                       experiment_label))

    # Comprobar dimension espacio latente
    lat_h, lat_w, lat_ch = check_latent_size(latent_size)

    # Train VF
    #  -  Image -> VAE -> Latent space -> VF -> Evaluation
    batch_size = 2*1000
    fitting_generator = vf_data_generator(dir_with_src_images, base_image, object_images, img_shape=img_shape,
                                                batch_size=batch_size)
    valid_generator = vf_data_generator(dir_with_src_images, base_image, object_images, img_shape=img_shape,
                                              batch_size=batch_size)
    
    batches_per_epoch = 100
    num_iterations = num_epochs * batches_per_epoch
    iterations = 0
    history = []
    val_history = []
    lista_predict_rewards = []
    lista_rewards = []
    model_name_list = []
    loss_last_value_list = []
    val_loss_last_value_list = []
    results_list = []
    loss_promedio = []
    val_loss_promedio = []

    #64
    #neurons_by_hidden_layer_list = [[64, 16, 6, 2, 1],  [64, 32, 16, 9, 1], [64, 32, 16, 9, 4], [64, 32, 16, 9, 6], [64, 32, 16, 4, 2]]
    #32
    #neurons_by_hidden_layer_list = [[32, 16, 6, 2, 1], [32, 16, 9, 4, 2], [32, 16, 9, 4, 1], [32, 25, 16, 8, 2], [32, 16, 8, 4, 2]]
    #16
    neurons_by_hidden_layer_list = [[64, 16, 6, 2, 1], [64, 32, 16, 9, 1], [64, 32, 16, 9, 4], [64, 32, 16, 9, 6], [64, 32, 16, 4, 2]]
    #9
    #neurons_by_hidden_layer_list = [[64, 16, 6, 2, 1], [64, 32, 16, 9, 1], [64, 32, 16, 9, 4], [64, 32, 16, 9, 6], [64, 32, 16, 4, 2]]

    act_func_list = [[tanh, tanh, tanh, tanh, tanh, tanh], [relu, relu, relu, relu, relu, relu]]
    act_func_names = ['tanh', 'relu']
    patience_list = [5,10,20]
 
    for neurons in neurons_by_hidden_layer_list:
        for func, func_names in zip(act_func_list, act_func_names):
            for pat in patience_list:
                arqui = f"{neurons}-{func_names}-pat_{pat})"
                print('\n')
                print(arqui)
                print('\n')
                # Create VF

                vf = KerasNN(input_neurons= latent_space_manual, neurons_by_hidden_layer= neurons, act_func= func, patience = pat, verbose = 1)
                model = vf.get_model()
                print(model.summary())
                print(vf.act_func)
                print(vf.patience)
                
            
                for (batch_inputs, batch_rewards), val_data in zip(fitting_generator, valid_generator):
                    
                    print(batch_rewards.shape)
                    print(batch_rewards)
                    
                    batch_latent = encoder.predict(batch_inputs)
                    # OJO: debo mezclar los datos de las trazas para entrenar la VF
                    print(batch_latent.shape)
                    # Paso entradas a espacio latente
                    latent = encoder.predict(batch_inputs)
                    if len(latent.shape) > 2:
                        bs, h, w, ch = latent.shape
                        latent = np.reshape(latent, (bs, h * w * ch))
                    # latent = (latent - np.min(latent))/np.ptp(latent)
                    print(latent.shape)
                    print(batch_rewards)
                    print(batch_rewards.shape)
                    #input('\nPulsa...\n')
                    latent_val = encoder.predict(val_data[0])
                    if len(latent_val.shape) > 2:
                        bs, h, w, ch = latent_val.shape
                        latent_val = np.reshape(latent_val, (bs, h * w * ch))
                
                    num_epochs = 2*500#100
                    
                    for training in range(10):
                        history = vf.train(input_data=batch_latent, output_data=batch_rewards, batch_size=100, epochs=num_epochs,
                                validation_data=(latent_val, val_data[1]))
                        loss_promedio.append(history.history['loss'])
                        val_loss_promedio.append(history.history['val_loss'])

                    break
                print(val_loss_promedio)
                loss_list = list(val_loss_promedio)
                df = pd.DataFrame(val_loss_promedio)
                print(df.head())
                model_name = "vf_model_{}_{}.h5".format(experiment_label, arqui)
                df.to_csv("trained_models/VF/"+str(latent_space_manual)+"/loss_"+model_name+".csv", index=False)
                #genero promedio de graficas
                fig = graficar("trained_models/VF/"+str(latent_space_manual)+"/loss_"+model_name+".csv")
                   
                model_name = "vf_model_{}_{}.h5".format(experiment_label, arqui)
                # fig = plt.figure()
                # plt.plot(history.history['loss'], color='red', label='mse')
                # plt.plot(history.history['val_loss'], color='blue', label='val_mse')
                # plt.yticks(np.arange(0, 0.2, step=0.02)) 
                # plt.legend()
                # plt.xlabel('Epochs')  
                # plt.ylabel('Mse')
                # plt.show()
                fig.savefig(
                        "snapshots/VF/"+str(latent_space_manual)+"/vf_mse_{}_{}_{}.png".format("diff" if residual_forward else "nodiff", experiment_label,  arqui),
                        bbox_inches='tight')
                vf.save_model("trained_models/VF/"+str(latent_space_manual)+"/vf_model_{}_{}.h5".format(experiment_label, arqui))
                
                model_name_list.append(model_name)
                print(model_name_list)
                #input('\nPulse\n')

                loss_list = list(zip(loss_promedio, val_loss_promedio))
                df = pd.DataFrame(loss_list, columns=['Loss Promedio','Val Loss Promedio'])
                print(df.head())
                
                df.to_csv("trained_models/VF/"+str(latent_space_manual)+"/loss_"+model_name+".csv", index=False)
                #loss_list = list(zip(history.history['loss'],history.history['val_loss']))
                #df = pd.DataFrame(loss_list, columns=['Loss','Val Loss'])
                #print(df.head())
                #df.to_csv("trained_models/VF/"+str(latent_space_manual)+"/loss_"+model_name+".csv", index=False)
                loss_last_value = history.history['loss']
                loss_last_value_list.append(loss_last_value[-1])
                val_loss_last_value = history.history['val_loss']
                val_loss_last_value_list.append(val_loss_last_value[-1])
                #vf.load_model('trained_models/vf_model_{}_1000epochs_2000batch_trainBueno_arqRich5.h5'.format(experiment_label))
                
                print('\nSimulador\n')
                candidates = 10
                n = 100
                from simulator import Sim

                sim = Sim(max_iter=n)
     
                batch_inputs = np.zeros((1, sim.h, sim.w, sim.ch), dtype=np.float32)
                print(batch_inputs.shape)
                # Test VF behaviour
                # Reinicio escenario
                sim.restart_scenario()
                sim.restart_scenario() # Para generar secuencias nuevas
                # Muestro figura
                sim.show_image(0)
                sim.show_image(1)
                #plt.savefig('trazas_experimento/fig_{}.png'.format(0), dpi=100)
                # For n iteraciones
                steps = 0
                intentos = 0
                exitos = 0
                lista_steps = []
                lista_intentos = []
                lista_exitos = []
                try:
                    for i in range(1, n):
                        # Pos. actual
                        print(f'\n{i}:\n')
                        actual = sim.images_history[sim.iter - 1]
                        np.copyto(batch_inputs, actual)
                        lat_actual = encoder.predict(batch_inputs)
                        # Genero acciones candidatas
                        candidate_actions = np.zeros((candidates, 1), dtype=np.float32)
                        for j in range(candidates):
                            action = np.random.uniform(-180, 180)
                            candidate_actions[j] = action / 180.0
                        candidate_actions = np.repeat(candidate_actions, latent_size, axis=1)

                        # Con FM y acciones candidatas calculo posic futuras
                        batch_latent = np.zeros((candidates, latent_size), dtype=np.float32)
                            
                        for j in range(candidates):
                            batch_latent[j] = lat_actual

                        # Veo la posicion en t1 despues de aplicar cada una de las acciones candidatas al punto inicial
                        encoded_imgs_t1 = forward_model.predict([batch_latent, candidate_actions])
                            

                        #input('\nPulsa una tecla...\n')
                        if residual_forward:
                            print("predicting residual forward...")
                            encoded_imgs_t1 = batch_latent + ((encoded_imgs_t1 * 2) - 1)
                    
                        #input('\nPulsa una tecla...\n')
                        # Valoro con la vf cada uno de los posibles puntos en t1
                        vf_valuations = vf.predict(encoded_imgs_t1)
                        #input('\nPulsa una tecla...\n')

                        show_decoded = False
                        if show_decoded:
                            # Veo estados predichos para ver si el FM condiciona que la VF valore mal
                            decoded_imgs_t1 = decoder.predict(encoded_imgs_t1)
                            #         n = 5
                            fig = plt.figure()  # 20,4 if 10 imgs
                            fig.suptitle('Decoded images', fontsize=16)
                            max_val_pos = vf_valuations.argmax()
                            for i in range(candidates):
                                ax = plt.subplot(2, int(candidates / 2), i + 1)
                                plt.yticks([])
                                plt.imshow(decoded_imgs_t1[i], vmin=decoded_imgs_t1.min(), vmax=decoded_imgs_t1.max(),
                                            interpolation='nearest')
                                plt.gray()
                                ax.get_xaxis().set_visible(True)
                                if max_val_pos == i:
                                    max_color = 'red'
                                else:
                                    max_color = 'black'
                                ax.set_title("action:{:1.2f}, val:{:1.4f}".format(candidate_actions[i][0] * 180,
                                            vf_valuations[i].tolist()[0]), rotation=0, size='large', color=max_color)
                                ax.set_xticklabels([])
                            fig.set_size_inches((18, 11), forward=False)
                            plt.savefig('trazas_experimento/fig_{}_candidates.png'.format(sim.iter - 1), dpi=200)
                        # Evalua pos. futuras con VF y elijo la mejor accion
                        best_action = candidate_actions[vf_valuations.argmax()][0] * 180.0
                        # Aplico accion en robot real
                        # Actualizo pos. actual
                        sim.apply_action(best_action)
                        # Muestro figura
                        sim.show_image(sim.iter - 1)
                        #plt.savefig('trazas_experimento/fig_{}.png'.format(sim.iter - 1), dpi=100)
                        # si llego al goal reinicio el escenario
                        steps += 1
                        lista_steps.append(steps)
                        if sim.reward or steps == 10:
                            if sim.reward:
                                exitos += 1
                                #plt.savefig('trazas_experimento/fig_{}_exito.png'.format(sim.iter - 1), dpi=100)
                                #lista_exitos.append(exitos)
                            intentos += 1
                            steps = 0
                            lista_intentos.append(intentos)
                            sim.restart_scenario()
                            sim.show_image(sim.iter - 1)
                            #plt.savefig('trazas_experimento/fig_{}_restart.png'.format(sim.iter - 1), dpi=100)
                            plt.close('all')
                except IndexError:
                    print('\nExcedido\n')

                plt.close('all')
                print("FIN")
                lista_exitos.append(exitos)
                print('\nExitos')
                print(exitos)
                print(intentos)
                print(steps)
                print(lista_steps)
                df1 = pd.DataFrame((zip(model_name_list, lista_exitos, loss_last_value_list, val_loss_last_value_list)), columns = ['Nombre modelo', 'Exitos', 'Loss', 'Val Loss'])
                print(df[0:10])
                model_name_list = []
                loss_last_value_list = []
                val_loss_last_value_list = []
                results_list.append(df1)
                #df1.to_csv("trained_models/Exitos_"+model_name+".csv", index=False)
    #concatenamos todos los resultados
    df_final = pd.concat(results_list)
    df_final.to_csv("trained_models/VF/"+str(latent_space_manual)+"/Resultados_modelos.csv", index=False)
    