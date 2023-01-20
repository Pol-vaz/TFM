# candidates = 10
# n = 100
# from simulator import Sim
# import numpy as np

# latent_size = 64
# sim = Sim(max_iter=n)
# # sim.restart_scenario()
# # sim.show_image(0)
# # sim.apply_action(90)
# # sim.show_image(1)
# batch_inputs = np.zeros((1, sim.h, sim.w, sim.ch), dtype=np.float32)

# # For n iteraciones
# steps = 0
# intentos = 0
# exitos = 0
# for i in range(1, n):
#     # Pos. actual
#     actual = sim.images_history[sim.iter - 1]
#     np.copyto(batch_inputs, actual)
#     #lat_actual = encoder.predict(batch_inputs)
#     # Genero acciones candidatas
#     candidate_actions = np.zeros((candidates, 1), dtype=np.float32)
#     print(candidate_actions.shape)
#     print('\n')
#     for j in range(candidates):
#         action = np.random.uniform(-180, 180)
#         print(action)
#         candidate_actions[j] = action / 180.0
#         print(candidate_actions)
#     candidate_actions = np.repeat(candidate_actions, latent_size, axis=1)
#     print(candidate_actions.shape)

import os
import numpy as np
import cv2

import matplotlib as mpl
import matplotlib.pyplot as plt
# include demo dataset mnist
#from keras.datasets import mnist

# from scipy.misc import imresize, imsave #, imread, imsave
# from scipy.misc import imsave #, imread, imsave
from imageio import imwrite, imread

# from scipy.ndimage import imread
from skimage.transform import resize, rotate

from utils import trim, read_files, psnr, log10, preprocess_size, \
     loadAndResizeImages2, preprocess_enhance_edges
from utils import load_parameters

try:
    import imgaug as ia
    from imgaug import augmenters as iaa
except:
    ia = None
    iaa = None
from pathlib import Path
from matplotlib import pyplot as plt
from tqdm import tqdm

import tensorflow
import numpy as np
import math

from skimage.transform import resize

from data_generators import brownian_data_generator, vf_data_generator, random_data_generator
from train_ae import prepare_optimizer_object
from models import make_forward_model
from world_models_vae_arch import build_vae_world_model

from utils import load_parameters

from keras_neural_network import KerasNN


def cdist(a, b, keepdims=False):
    d = np.sqrt(np.sum(np.square(a - b), axis=2))
    if keepdims:
        d = np.repeat(d[:,:,np.newaxis], 3, axis=2)
    return d
    
max_cdist = cdist(np.zeros((1, 1, 3)), np.ones((1, 1, 3)))

#def cdist_average(avg_color, b, keepdims=False):
#    avg_img = np.ones(b.shape) * avg_color
#    return cdist(avg_img, b, keepdims=False)

def random_rotation(image_array: np.ndarray, angle=None):
    # pick a random degree of rotation between 25% on the left and 25% on the right
    if angle is None:
        angle = np.random.uniform(-180, 180)
    return rotate(image_array, angle)


def vf_data_generator(dir_with_src_images, base_image_filename, object_image_list, img_shape=(28, 28, 1),
                            batch_size=100, p_goal=1.0, resized_objects=None):
    h, w, ch = img_shape

    # define inputs
    batch_inputs = np.zeros((batch_size, h, w, ch), dtype=np.float32)
    # define outputs
    batch_outputs = np.zeros((batch_size, h, w, ch), dtype=np.float32)
    # define attention masks (by default ones as everything has the same importance)
    batch_masks = np.ones((batch_size, h, w, ch), dtype=np.float32)

    def preprocess_size_helper(new_dim=(h, w)):
        return lambda image: preprocess_size(image, new_dim)

    preprocessors = [preprocess_size_helper(new_dim=(h, w)), preprocess_enhance_edges]

    # load images
    base_image = loadAndResizeImages2(dir_with_src_images, [base_image_filename])[0]
    objects = loadAndResizeImages2(dir_with_src_images, object_image_list, load_alpha=True)

    # load params, since some of them are needed to generate data:
    parameters_filepath = "config.ini"
    parameters = load_parameters(parameters_filepath)

    size_factor = float(parameters["synthetic"]["size_factor"])
    save_batch = eval(parameters["synthetic"]["save_batch"])
    calc_masks = eval(parameters["synthetic"]["calc_masks"])
    dilate_masks = int(parameters["synthetic"]["dilate_masks"])
    blur_masks = eval(parameters["synthetic"]["blur_masks"])
    blur_kernel = eval(parameters["synthetic"]["blur_kernel"])

    obj_attention = float(parameters["synthetic"]["obj_attention"])
    back_attention = float(parameters["synthetic"]["back_attention"])

    subtract_median = eval(parameters["synthetic"]["subtract_median"])

    add_noise = eval(parameters["synthetic"]["add_noise"])
    noise_amnt = float(parameters["synthetic"]["noise_amnt"])

    loss = (parameters["hyperparam"]["loss"])

    # median threshold
    threshold = .5

    # resize to desired size
    orig_h, orig_w, _ = base_image.shape
    ratio_h = orig_h / h
    ratio_w = orig_w / w
    print(ratio_h, ratio_w)
    print(orig_h, orig_w)
    input('\nPulsa alguna tecla para continuar...\n')
    
    base_image = preprocess_size(base_image, (h, w))
    # imwrite("tmp/{}.png".format("imagen_base"), base_image)
    if resized_objects is None:
        resized_objects = []
    objects_pos = []
    rotated_objs = [None] * len(objects)
    for o in objects:
        ho, wo, cho = o.shape
        print(ho,wo,cho)
        input('\nPulsa alguna tecla para continuar...\n')
        if ho == wo:
            hn = int((ho / ratio_w) * size_factor) #10
            wn = int((wo / ratio_w) * size_factor) #10
            print(hn, wn)
            input('\nPulsa alguna tecla para continuar...\n')
        else:
            hn = int((ho / ratio_h) * size_factor)
            wn = int((wo / ratio_w) * size_factor)
        resized_o = preprocess_size(o, (hn, wn))
        # imwrite("tmp/{}.png".format("resized_object"), resized_o)
        resized_objects.append(resized_o)
        print(resized_o.shape)
        input('\nPulsa alguna tecla para continuar...\n')

        #
        # print(w, " - ", wo)
        x = np.random.randint(low=0, high=w - wn)  # +wo
        y = np.random.randint(low=(60 / ratio_h), high=h - hn - (30 / ratio_h))
        # y = np.random.randint(18, high=h - hn - (30 / ratio_h))
        objects_pos.append((x, y))
        print(x,y)
        input('\nPulsa alguna tecla para continuar...\n')

    L = wn // 3 # wn // 2
    # print("L: ", L)
    a = 0
    replace_robots = True
    P_greed = p_goal  # 1. / 2. #1 / 20. #1 / 33. # Probabilidad de ir directo al goal

    iteration = -1

    # serve randomly generated images
    while True:
        iteration += 1
        # go through the entire dataset, using batch_sized chunks each time

        batch_actions = np.zeros((batch_size, 1), dtype=np.float32)
        batch_rewards = np.zeros((batch_size, 1), dtype=np.float32)
        batch_new_rewards = np.zeros((batch_size, 1), dtype=np.float32)
        ultimo_reward = 0
        for i in range(0, batch_size):

            np.copyto(batch_inputs[i], base_image)
            print(i)
            print(batch_size, i%5)
            #input('\nPulsa alguna tecla para continuar...\n')

            # TODO: randomly place the objects:
            if replace_robots:
                if i%5 == 0:
                    for ix, o in enumerate(resized_objects):
                        a = np.random.uniform(-180, 180)
                        print(f'a -> {a}')
                        o_rot = random_rotation(o, angle=a - 90)  # +90 since robot is sideways
                        print(f'o_rot -> {o_rot.shape}')
                        # if ix == 1:
                        #    batch_actions[i] = a / 180
                        ho, wo, cho = o_rot.shape
                        print(f'ho,wo,cho -> {ho} {wo} {cho}')
                        x = np.random.randint(low=0, high=w - wo)  # +wo
                        # print((100 / ratio_h))
                        # 30 is the magic number to limit the random placement of objects inside image
                        y = np.random.randint(low=(60 / ratio_h), high=h - ho - (30 / ratio_h))
                        # y = np.random.randint(18, high=h - ho - (30 / ratio_h))
                        print(f'x -> {x} and y-> {y}')
                        #input('\nPulsa alguna tecla para continuar...\n')

                        xg = x - (wo // 2)
                        yg = y - (ho // 2)

                        if xg + wo > w:
                            xg = w - wo
                        if yg + ho > h:
                            yg = h - ho
                        if xg < 0:
                            xg = 0
                        if yg < 0:
                            yg = 0

                        x = xg + (wo // 2)
                        y = yg + (ho // 2)

                        objects_pos[ix] = (x, y)
                        rotated_objs[ix] = o_rot
                    print(f'object_pos 5 -> {objects_pos} and rotated_objs-> {o_rot.shape} {len(rotated_objs)}')
                    #input('\nPulsa alguna tecla para continuar...\n')

                    # replace_robots = False
                else:
                    for ix, o in enumerate(resized_objects):
                        if ix == 1:
                            a = np.random.uniform(-180, 180)
                            o_rot = random_rotation(o, angle=a - 90)  # +90 since robot is sideways
                            # if ix == 1:
                            #    batch_actions[i] = a / 180
                            ho, wo, cho = o_rot.shape

                            # x = np.random.randint(low=0, high=w - wo)  # +wo
                            x = np.random.randint(low=0, high=60)  # +wo
                            # print((100 / ratio_h))
                            # 30 is the magic number to limit the random placement of objects inside image
                            # y = np.random.randint(low=(60 / ratio_h), high=h - ho - (30 / ratio_h))
                            y = np.random.randint(8, high=60)

                            xg = x - (wo // 2)
                            yg = y - (ho // 2)

                            if xg + wo > w:
                                xg = w - wo
                            if yg + ho > h:
                                yg = h - ho
                            if xg < 0:
                                xg = 0
                            if yg < 0:
                                yg = 0

                            x = xg + (wo // 2)
                            y = yg + (ho // 2)

                            objects_pos[ix] = (x, y)
                            rotated_objs[ix] = o_rot
                    print(f'object_pos -> {objects_pos} and rotated_objs-> {o_rot.shape} {len(rotated_objs)}')
                    #input('\nPulsa alguna tecla para continuar...\n')                       
            else:
                print('replace_robots ya no es true\n')
                # move the robot into random orientation with fixed direction
                # imwrite("tmp/{}.png".format("obj_generated_" + str(i)),  o_rot)
                robot_object_distance = h + w
                for ix, o in enumerate(resized_objects):
                    if ix == 1:  # only do this for the robot and not the object
                    # if ix == 2:  # only do this for the robot and not the object Para 2 objetos

                        x, y = objects_pos[ix]
                        x_t, y_t = objects_pos[ix - 1]
                        # x_t, y_t = objects_pos[ix - 2] # Para objeto rojo con 2 objetos
                        # select angle towards the object in 10% of the time
                        if np.random.random() <= P_greed:
                            a = np.arctan2(x_t - x, y_t - y)
                            a = np.degrees(a)  # -180 <> 180
                            # print("target angle: ", a)
                        else:  # select angle randomly in 90% of the time
                            a = np.random.uniform(-180, 180)
                            # a = a * 360 # -180 <> 180

                        action = a  # action in degrees

                        o_rot = random_rotation(o, a - 90)  # +90 since robot is sideways
                        ho, wo, cho = o_rot.shape

                        # this is new position of the robot
                        a = np.radians(a)  # need to convert to radians before using sin and cos
                        batch_actions[i] = action / 180  # np.sin(a), np.cos(a) # action / 180

                        x = x + L * np.sin(a)
                        y = y + L * np.cos(a)

                        x = int(np.round(x))
                        y = int(np.round(y))

                        xg = x - (wo // 2)
                        yg = y - (ho // 2)

                        if xg + wo > w:
                            xg = w - wo
                        if yg + ho > h:
                            yg = h - ho
                        if xg < 0:
                            xg = 0
                        if yg < 0:
                            yg = 0

                        x = xg + (wo // 2)
                        y = yg + (ho // 2)

                        objects_pos[ix] = (x, y)
                        rotated_objs[ix] = o_rot

                        robot_object_distance = np.sqrt((x - x_t) ** 2 + (y - y_t) ** 2)
                        # print("robot dist / L: {} / {}".format(robot_object_distance, L))

                if robot_object_distance < L:
                    reward = 1
                    batch_rewards[i] = reward
                    # Retropropago el reward hacia atras
                    # for j in range(i, ultimo_reward, -1):
                    reward -= 0.15
                    if ultimo_reward == 0:
                        for j in reversed(range(ultimo_reward, i)):
                            batch_rewards[j] = max(0, reward)
                            reward -= 0.15
                    else:
                        for j in reversed(range(ultimo_reward+1,i)):
                            batch_rewards[j] = max(0, reward)
                            reward -= 0.15
                    # batch_rewards[ultimo_reward+1:i]=

                    ultimo_reward = i
                    replace_robots = True

            for ix, o in enumerate(resized_objects):
                x, y = objects_pos[ix]
                o_rot = rotated_objs[ix]
                ho, wo, cho = o_rot.shape
                mask = o_rot[:, :, 3]  # / 255.0
                # print(mask.max(), mask.min())
                # print(x, y, ho, wo, cho)
                xg = x - (wo // 2)
                yg = y - (ho // 2)

                batch_inputs[i][yg:yg + ho, xg:xg + wo, 0] = batch_inputs[i][yg:yg + ho, xg:xg + wo, 0] * (
                            1 - mask) + mask * o_rot[:, :, 0]  # *255.0
                batch_inputs[i][yg:yg + ho, xg:xg + wo, 1] = batch_inputs[i][yg:yg + ho, xg:xg + wo, 1] * (
                            1 - mask) + mask * o_rot[:, :, 1]  # *255.0
                batch_inputs[i][yg:yg + ho, xg:xg + wo, 2] = batch_inputs[i][yg:yg + ho, xg:xg + wo, 2] * (
                            1 - mask) + mask * o_rot[:, :, 2]  # *255.0

                x, y = objects_pos[0]

                # x_t, y_t = objects_pos[1]
                x_t, y_t = objects_pos[ix - 2] # Para 2 objetos

                robot_object_distance = np.sqrt((x - x_t) ** 2 + (y - y_t) ** 2)
                print(f'robot_object_distance -> {robot_object_distance}')
                print(x,y)
                print(x_t,y_t)
                
                batch_new_rewards[i] = max(min(1.0, (50-robot_object_distance)/(50-L)), 0)
                print(f'batch_new_rewards -> {batch_new_rewards[i]}')
                

            # imwrite("tmp/{}.png".format("in_generated_" + str(i)),  batch_inputs[i])
            np.copyto(batch_outputs[(i - 1) % batch_size], batch_inputs[i])
            # imwrite("tmp/{}.png".format("out_generated_" + str(i)),  batch_outputs[i])
            # print(batch_outputs[i].max(), batch_outputs[i].min())

            # fig = plt.figure()  # 20,4 if 10 imgs
            # ax = plt.subplot()
            # plt.imshow(batch_inputs[i])  # .reshape(img_dim, img_dim)

        batch_median = np.median(batch_outputs, axis=0, keepdims=True)

        # print("Batch median shape: ", batch_median.shape)
        # print("Batch median shape: ", batch_outputs.shape)
        if calc_masks:
            median_min = batch_median[0].min()
            median_max = batch_median[0].max()
            for i in range(0, batch_size):

                tmp = cdist(batch_median[0], batch_inputs[i], keepdims=True)  # color distance between images
                mask = (tmp > threshold * max_cdist).astype(float)
                batch_masks[i] = mask * obj_attention
                # back_mask = ( tmp <= 0 ).astype(float) + back_attention

                # batch_masks[i][batch_masks[i] > 0.5] += 0.1
                # uncomment to blur the images (soft attention)
                if dilate_masks > 0:
                    # print("dilating masks...")
                    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                    batch_masks[i] = cv2.dilate(batch_masks[i], kernel, iterations=dilate_masks)
                if back_attention > 0.0:
                    # print("Setting background weights...")
                    #    back_mask = ( tmp <= 0 ).astype(float) + back_attention
                    batch_masks[i] += ((1 - (mask).astype(int)).astype(float) * back_attention)

                if blur_masks:
                    # print("Blurring masks....")
                    batch_masks[i] = cv2.blur(batch_masks[i], blur_kernel)  # add blur if needed

                if save_batch:  # save generated images to tmp folder
                    me_min = batch_actions[i][0]  # batch_masks[i].min()
                    me_max = batch_rewards[i][0]  # batch_masks[i].max()
                    label = str(i)  # str(np.random.randint(0, 1000))
                    imwrite("tmp/{}.png".format("m_{}_{}_{:01.2f}_{}".format(iteration, label, me_min, me_max)),
                            batch_masks[i])
                    imwrite("tmp/{}.png".format("A_{}_{}_{:01.2f}_{}".format(iteration, label, me_min, me_max)),
                            batch_inputs[i])
                    imwrite("tmp/{}.png".format("B_{}_{}_{:01.2f}_{}".format(iteration, label, me_min, me_max)),
                            batch_outputs[i])

            if save_batch:  # save only first batch
                save_batch = False

        # batch_percentile = np.percentile(batch_outputs, 99.9, axis=0, keepdims=True)
        # label = str(np.random.randint(0, 1000))
        # imwrite("tmp/{}.png".format("percentile_99.9_" + str(label)), batch_percentile[0])
        if subtract_median:
            # batch_mean = batch_outputs.mean(axis=0, keepdims=True)
            # careful - batch_size must be greater than 1!!!
            # batch_median = np.median(batch_outputs, axis=0, keepdims=True)

            # imwrite("tmp/{}.png".format("median_" + str(i)), batch_median[0])
            batch_outputs = batch_median - batch_outputs
            # imwrite("tmp/{}.png".format("out1_" + str(i)), batch_outputs[0])

        if add_noise:
            batch_inputs += noise_amnt * np.random.normal(loc=0.0, scale=1.0, size=batch_inputs.shape)

            # label = str(np.random.randint(0, 1000))
        # imwrite("tmp/{}.png".format(label + "_in_generated_" + str(i)),  batch_inputs[0])
        # imwrite("tmp/{}.png".format(label + "_out_generated_" + str(i)),  batch_median[0] - batch_outputs[0])
        # print(batch_median.shape)
        # if 'wmse' in loss and 'out-median' in mode:
        #    yield [batch_inputs, np.repeat(np.array([batch_median]), batch_size, axis=0).reshape((batch_size, h, w, 3))], batch_outputs
        # print(batch_actions.shape)
        #input('\nPulsa alguna tecla para continuar...\n')
        if 'wmse' in loss:
            # yield [batch_inputs, batch_masks, batch_actions, batch_rewards], batch_outputs  # , batch_actions, batch_masks
            # yield batch_inputs, batch_rewards  # , batch_actions, batch_masks
            yield batch_inputs, batch_new_rewards
        else:
            yield [batch_inputs, batch_actions], batch_outputs
    
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

    load_vf = False

    # Load VAE and FM models:
    experiment_label = 'simple_vaewm.osize-3.0.oatt-0.8.e50.bs32.lat64.c4.opt-adamw.loss-wmse'
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

    # Create VF
    vf = KerasNN(input_neurons= 64, verbose = 1)

    # Train VF
    #   -  Image -> VAE -> Latent space -> VF -> Evaluation
    batch_size = 2*1000
   
    fitting_generator = vf_data_generator(dir_with_src_images, base_image, object_images, img_shape=img_shape, batch_size=batch_size)
    
    for (batch_inputs, batch_rewards) in fitting_generator:

        print(batch_inputs.shape, batch_rewards.shape)
        input('\nTecla....\n')
        batch_latent = encoder.predict(batch_inputs) 

        latent = encoder.predict(batch_inputs)
        if len(latent.shape) > 2:
            bs, h, w, ch = latent.shape
            latent = np.reshape(latent, (bs, h * w * ch))
        
        # print(batch_rewards[0:10])
        # lista = batch_rewards[0:10]
        # posi_max = np.where(lista == np.amax(lista)) 
        # posi_min = np.where(lista == np.amin(lista))               
        # for num_max, num_min in zip(posi_max[0], posi_min[0]):
        #     print(num_max, num_min) 
        #     imwrite("C:/Users/pablo/Desktop/prueba_input_max.png",batch_inputs[num_max])
        #     imwrite("C:/Users/pablo/Desktop/prueba_input_min.png",batch_inputs[num_min])
        # input('\nDale WEY\n')
        #Mostrar imagen
        # img_width  = batch_inputs.shape[1]
        # img_height = batch_inputs.shape[2]
        # num_channels = 3 #--> grey scale so 1 channel

        # x_train = batch_inputs.reshape( img_width, img_height,  num_channels)
        # plt.imshow(x_train)
        # plt.show()
        break
