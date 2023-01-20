from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm

import tensorflow
import numpy as np
import math
import pandas as pd
import datetime as dt

from skimage.transform import resize
from sklearn import preprocessing

import cv2
from skimage.exposure.exposure import equalize_hist
from skimage.io import imread
import skimage.io
import os

from data_generators import brownian_data_generator, vf_data_generator, random_data_generator
from train_ae import prepare_optimizer_object
from models import make_forward_model_3_layers, make_forward_model_5_layers, make_forward_model_10_layers
from world_models_vae_arch import build_vae_world_model
now=dt.datetime.now()
dt_string=now.strftime("%Y_%m_%d")
from keras.models import Sequential, load_model, save_model
from keras.layers import Dense
from keras.callbacks import EarlyStopping
from keras.activations import relu, tanh, linear
from keras.initializers import random_normal
import tensorflow as tf

from utils import load_parameters

from keras_neural_network_policy import KerasNN

def load_images(path):

    content=os.listdir(path)
    X=[]
    count=0
    for i in content:
        count=count+1
        file_path=os.path.join(path,i)

        try:
            #im=imread(file_path)
            im = cv2.imread(file_path)
            X.append(np.array(im))

        except:
            print("Not an image")
            print(file_path)
            continue
    return np.array(X)

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

def generador_policy(n = 100, candidates = 30, ruta ='Train'):
    generador = True
    if generador:
        from simulator import Sim
        sim = Sim(max_iter=n)
        batch_inputs = np.zeros((1, sim.h, sim.w, sim.ch), dtype=np.float32)
        # Reinicio escenario
        sim.restart_scenario() # Para generar secuencias nuevas
        sim.show_image(0)
        sim.show_image(1)
        input_images_list = []
        exact_angles = []

        try:
            for i in range(1, n):
                # Gnero imagen nueva
                #print(f'\n{i}:\n')
                actual = sim.images_history[sim.iter - 1]
                sim.show_image(sim.iter - 1)
                target, robot = sim.objects_pos
                import math
                angle_2 = math.atan2(robot[1]- target[1], robot[0] - target[0])
                best_action_exact = ((math.degrees(angle_2) + 90) * -1 ) / 360
                #print(angle_deg_2)
                exact_angles.append(best_action_exact)
                #input('\npulsa...')
                np.copyto(batch_inputs, actual)
                lat_actual = encoder.predict(batch_inputs)


                input_images_list.append(lat_actual)
                input_images = np.array(input_images_list)



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

                if residual_forward:
                    #print("predicting residual forward...")
                    encoded_imgs_t1 = batch_latent + ((encoded_imgs_t1 * 2) - 1)

                # Valoro con la vf cada uno de los posibles puntos en t1
                vf_valuations = vf.predict(encoded_imgs_t1)
                # Evalua pos. futuras con VF y elijo la mejor accion
                best_action = candidate_actions[vf_valuations.argmax()][0]

                best_action_list.append(best_action)
                sim.restart_scenario()

        except IndexError:
            print('\nExcedido\n')

        print('\nGeneradas todas las imágenes y sus correspondientes mejores acciones')

        best_actions = np.array(best_action_list)
        best_actions_exact = np.array(exact_angles)
        best_actions = best_actions.reshape(-1,1)
        best_actions_exact = best_actions_exact.reshape(-1,1)



        return input_images, best_actions, best_actions_exact

def simulador_exitos(perfection = False):
    print('\nSimulador\n')
    n = 1000
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
    fracasos = 0
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

            # batch_latent = np.zeros((candidates, latent_size), dtype=np.float32)

            # #input('\nPulsa una tecla...\n')
            # if residual_forward:
            #     print("predicting residual forward...")
            #     lat_actual = batch_latent + ((lat_actual * 2) - 1)

            #input('\nPulsa una tecla...\n')
            #Predigo acción con la Policy
            best_action_predict_p = policy.predict(lat_actual)
            b = best_action_predict_p[0][0]
            print(b)

            target, robot = sim.objects_pos
            import math
            angle_2 = math.atan2(robot[1]- target[1], robot[0] - target[0])
            best_action_exact = ((math.degrees(angle_2) + 90) * -1 ) / 360
            best_action_predict = best_action_exact
            a = best_action_predict
            print(a)
            #input('\nPulsa...')
            # Aplico accion en robot real
            # Actualizo pos. actual
            # print(best_action_predict)
            # best_action_predict = best_action_predict[0]
            # for i in best_action_predict:
            #     print(i)
            #     a = i

            #input('\nPulsa una tecla...\n')
            if perfection:
                sim.apply_action(a * 360)
            else:
                sim.apply_action(b * 360)
            # Muestro figura
            sim.show_image(sim.iter - 1)
            #plt.savefig('trazas_experimento/fig_{}.png'.format(sim.iter - 1), dpi=100)
            # si llego al goal reinicio el escenario
            steps += 1
            lista_steps.append(steps)
            if sim.reward or steps == 10:
                if sim.reward and steps <=10:
                    exitos += 1
                    #plt.savefig('trazas_experimento/fig_{}_exito.png'.format(sim.iter - 1), dpi=100)
                    #lista_exitos.append(exitos)
                if steps == 10:
                    fracasos += 1
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

    df1 = pd.DataFrame((zip(model_name_list, lista_exitos, loss_last_value_list, val_loss_last_value_list)), columns = ['Nombre modelo', 'Exitos', 'Loss', 'Val Loss'])
    print(df1[0:10])
    return exitos, intentos, lista_steps, fracasos



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

    forward_model = make_forward_model_10_layers([latent_size], latent_size, learn_only_difference=residual_forward)
    forward_model.compile(loss='mse', optimizer=prepare_optimizer_object('adam', 0.001), metrics=['mse'])
    # forward_model.load_weights(
    #     'trained_models/forward_model_{}_{}_corregido.h5'.format("diff" if residual_forward else "nodiff", experiment_label))
    forward_model.load_weights(
        'trained_models/forward_model_{}_{}_arqRich6.h5'.format("diff" if residual_forward else "nodiff",
                                                       experiment_label))
    #cargo modelo VF
    # Create VF
    vf = KerasNN(input_neurons= latent_space_manual, verbose = 1)
    vf.load_model('trained_models/vf_model_{}.h5'.format(experiment_label))

    # Comprobar dimension espacio latente
    lat_h, lat_w, lat_ch = check_latent_size(latent_size)

    # Train Policy
    #  -   Latent space -> Policy -> Best Action

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
    best_action_list = []
    cosa_list = []
    metodo_exacto = True
    num = 2000
    y = np.linspace(0,num-1,num-1)
    image_train_list = []
    image_test_list = []


    generador = True
    #Llamamos al generador y generamos el dataset
    if generador:
        input_images_train, best_actions_predict_train, best_actions_exact_train = generador_policy(n = num, candidates = 10, ruta = 'Train')
        print(input_images_train.shape)
        best_action_list = []
        input_images_val, best_actions_predict_val, best_actions_exact_val = generador_policy(n = num, candidates = 10, ruta = 'Test')

        input_images_testeo, best_actions_predict_testeo, best_actions_exact_testeo = generador_policy(n = 100, candidates = 10, ruta = 'Testeo')

        if len(input_images_train.shape) > 2:
            bs, n, dim = input_images_train.shape
            input_images_train = np.reshape(input_images_train, (bs, n*dim))

        if len(input_images_val.shape) > 2:
            bs, n, dim = input_images_val.shape
            input_images_val = np.reshape(input_images_val, (bs, n*dim))

        if len(input_images_testeo.shape) > 2:
            bs, n, dim = input_images_testeo.shape
            input_images_testeo = np.reshape(input_images_testeo, (bs, n*dim))

    carga = False
    if carga:
        X_train =load_images("C:/Users/pablo/Desktop/CodigoTFM/disentangled_experiments-main/disentangled_experiments-main/generador_policy_copia/Train")

        obj_size = (64,64)
        for i in X_train:

            i = cv2.resize(i, obj_size)

            if len(i.shape) < 4:
                n1, n2, dim = i.shape
                i = np.reshape(i, (1, n1, n2, dim))

            lat_actual_train = encoder.predict(i)

            image_train_list.append(lat_actual_train)
            input_images_train = np.array(image_train_list)

        #input_images_train = cv2.resize(input_images_train, (batch, dim) )
        if len(input_images_train.shape) > 2:
            bs, n, dim = input_images_train.shape
            input_images_train = np.reshape(input_images_train, (bs, n*dim))


        X_test =load_images("C:/Users/pablo/Desktop/CodigoTFM/disentangled_experiments-main/disentangled_experiments-main/generador_policy_copia/Test")

        obj_size = (64,64)
        for j in X_test:

            j = cv2.resize(j, obj_size)

            if len(j.shape) < 4:
                n1, n2, dim = j.shape
                j = np.reshape(j, (1, n1, n2, dim))

            lat_actual_test = encoder.predict(j)

            image_test_list.append(lat_actual_test)
            input_images_val = np.array(image_test_list)

        batch, num, dim = input_images_val.shape
        if len(input_images_val.shape) > 2:
            bs, n, dim = input_images_val.shape
            input_images_val = np.reshape(input_images_val, (bs, n*dim))


        df_train = pd.read_csv("C:/Users/pablo/Desktop/CodigoTFM/disentangled_experiments-main/disentangled_experiments-main/generador_policy_copia/Best_actions/Train/Best_action_batch.csv", sep = ';')
        df_test = pd.read_csv("C:/Users/pablo/Desktop/CodigoTFM/disentangled_experiments-main/disentangled_experiments-main/generador_policy_copia/Best_actions/Test/Best_action_batch.csv", sep = ';')




    metodo_exacto = True
    if metodo_exacto == True:
        if generador:
            best_actions_train = best_actions_exact_train
            best_actions_val = best_actions_exact_val

        if carga:
            best_actions_exact_train = df_train['Best Action exacto']
            best_actions_exact_val = df_test['Best Action exacto']
            best_actions_train = best_actions_exact_train
            best_actions_val = best_actions_exact_val


    if metodo_exacto == False:
        if generador:
            best_actions_train = best_actions_predict_train
            best_actions_val = best_actions_predict_val

        if carga:
            best_actions_predict_train = df_train['Best Action deliberativo']
            best_actions_predict_val = df_test['Best Action deliberativo']
            best_actions_train = best_actions_predict_train
            best_actions_val = best_actions_predict_val



    print(best_actions_train[0:10])
    print(best_actions_val[0:10])


    doit = True
    if doit:

        neurons = [64, 16, 6, 2, 1]
        func = [tanh, relu, relu, relu, relu, tanh]
        pat = 20
        # Create Policy
        #TODO Crear red neuronal parecida a la value function que de input tenga el espacio latente y de salida una neurona que de la mejor acción
        policy = KerasNN(input_neurons= latent_space_manual, neurons_by_hidden_layer= neurons, act_func= func, patience = pat, verbose = 1)
        model = policy.get_model()
        print(model.summary())
        print(policy.act_func)
        print(policy.patience)
        num_epochs = 2*1000#100

        #TRAIN
        train = True
        if train:
            print(best_actions_predict_train[0:10]*360)
            print(best_actions_exact_train[0:10]*360)
            #input('\nModelo listo para Entrenar\n\nPulsa una tecla para continuar...\n')
            history = policy.train(input_data=input_images_train, output_data= best_actions_train, batch_size=50, epochs=num_epochs,
                        validation_data=(input_images_val, best_actions_val))
            model_name = "Policy_model_{}.h5".format(experiment_label)
            #df.to_csv("trained_models/Policy/"+str(latent_space_manual)+"/loss_"+model_name+".csv", index=False)


            policy.save_model("C:/Users/pablo/Desktop/Policy_model_{}.h5".format(experiment_label))
            fig = plt.figure()
            plt.plot(history.history['loss'], color='red', label='mse')
            plt.plot(history.history['val_loss'], color='blue', label='val_mse')
            #plt.yticks(np.arange(0, 1, step=0.05)) 
            plt.ylim(0,1) 
            plt.legend()
            plt.xlabel('Epochs')  
            plt.ylabel('Mse')
            plt.show()
            fig.savefig(
                    "C:/Users/pablo/Desktop/Policy_loss_model_{}.png".format(experiment_label),
                    bbox_inches='tight')



        test = False
        if test:
            path_model= "C:/Users/pablo/Desktop/Policy_model_{}.h5".format(experiment_label)
            model = policy.load_model(path_model)

            #predicciones ricas
            exitos, intentos, steps, fracasos = simulador_exitos(perfection = False)
            print(exitos, intentos, fracasos, steps)
            print('\n\n')
            print(f'Porcentaje de Exito -> {(exitos/intentos)*100}%')
            print(f'Porcentaje de Fracaso -> {(fracasos/intentos)*100}%')

            #Esperado VS Real
            #input('\nEsperado VS Real\n\nPulsa una tecla para continuar...\n')

            # policy_prediction = policy.predict(input_images_train)

            # fig = plt.figure()
            # # sort freq
            # best_actions_test = np.sort(best_actions_exact_train, axis = 0)
            # policy_prediction = np.sort(policy_prediction, axis = 0)
            # #print(best_actions_val)
            # # plt.plot(policy_prediction, 'o', color='red', label='Prediction')
            # # plt.plot(best_actions_predict_val, 'o', color='blue', label='Real')
            # plt.scatter(range(0,len(policy_prediction)),policy_prediction, marker='+',label='obtenidos')
            # plt.scatter(range(0,len(best_actions_test)),best_actions_test, marker='+',label='esperados')
            # #plt.yticks(np.arange(0, 1, step=0.05))
            # plt.ylim(-1,1)
            # plt.legend()
            # plt.xlabel('Images')
            # plt.ylabel('Best Action')
            # #plt.show()
            # fig.savefig('C:/Users/pablo/Desktop/Policy_predictions.png', bbox_inches='tight')






