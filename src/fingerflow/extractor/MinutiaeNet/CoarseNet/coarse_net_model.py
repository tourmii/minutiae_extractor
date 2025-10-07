"""Code for FineNet in paper "Robust Minutiae Extractor: Integrating Deep Networks and Fingerprint Domain Knowledge" at ICB 2018
  https://arxiv.org/pdf/1712.09401.pdf

  If you use whole or partial function in this code, please cite paper:

  @inproceedings{Nguyen_MinutiaeNet,
    author    = {Dinh-Luan Nguyen and Kai Cao and Anil K. Jain},
    title     = {Robust Minutiae Extractor: Integrating Deep Networks and Fingerprint Domain Knowledge},
    booktitle = {The 11th International Conference on Biometrics, 2018},
    year      = {2018},
    }
"""
from time import time
from datetime import datetime

import cv2
import tensorflow as tf
import numpy as np
from scipy import ndimage
from tensorflow.keras import layers, models, regularizers, optimizers, utils

# using as submodule
from ..FineNet import fine_net_model
from . import coarse_net_utils, minutiae_net_utils, loss_function


# def conv_bn(bottom, w_size, name, strides=(1, 1), dilation_rate=(1, 1)):
#     top = Conv2D(w_size[0], (w_size[1], w_size[2]),
#                  kernel_regularizer=regularizers.l2(5e-5),
#                  padding='same',
#                  strides=strides,
#                  dilation_rate=dilation_rate,
#                  name='conv-'+name)(bottom)
#     top = BatchNormalization(name='bn-'+name)(top)
#     return top


def conv_bn_prelu(bottom, w_size, name, strides=(1, 1), dilation_rate=(1, 1)):
    if dilation_rate == (1, 1):
        conv_type = 'conv'
    else:
        conv_type = 'atrousconv'

    top = layers.Conv2D(w_size[0], (w_size[1], w_size[2]),
                        kernel_regularizer=regularizers.l2(5e-5),
                        padding='same',
                        strides=strides,
                        dilation_rate=dilation_rate,
                        name=conv_type + name)(bottom)
    top = layers.BatchNormalization(name='bn-' + name)(top)
    top = layers.PReLU(alpha_initializer='zero', shared_axes=[
        1, 2], name='prelu-' + name)(top)
    # top = Dropout(0.25)(top)
    return top


def get_coarse_net_model(input_shape=(400, 400, 1), weights_path=None, mode='train'):
    # Change network architecture here!!
    img_input = layers.Input(input_shape)
    bn_img = layers.Lambda(coarse_net_utils.img_normalization,
                           output_shape=lambda input_shape: input_shape,
                           name='img_normalized')(img_input)

    # Main part
    conv = conv_bn_prelu(bn_img, (64, 5, 5), '1_0')
    conv = conv_bn_prelu(conv, (64, 3, 3), '1_1')
    conv = conv_bn_prelu(conv, (64, 3, 3), '1_2')
    conv = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)

    # =======Block 1 ========
    conv1 = conv_bn_prelu(conv, (128, 3, 3), '2_1')
    conv = conv_bn_prelu(conv1, (128, 3, 3), '2_2')
    conv = conv_bn_prelu(conv, (128, 3, 3), '2_3')
    conv = layers.add([conv, conv1])

    conv1 = conv_bn_prelu(conv, (128, 3, 3), '2_1b')
    conv = conv_bn_prelu(conv1, (128, 3, 3), '2_2b')
    conv = conv_bn_prelu(conv, (128, 3, 3), '2_3b')
    conv = layers.add([conv, conv1])

    conv1 = conv_bn_prelu(conv, (128, 3, 3), '2_1c')
    conv = conv_bn_prelu(conv1, (128, 3, 3), '2_2c')
    conv = conv_bn_prelu(conv, (128, 3, 3), '2_3c')
    conv = layers.add([conv, conv1])

    conv_block1 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)
    # ==========================

    # =======Block 2 ========
    conv1 = conv_bn_prelu(conv_block1, (256, 3, 3), '3_1')
    conv = conv_bn_prelu(conv1, (256, 3, 3), '3_2')
    conv = conv_bn_prelu(conv, (256, 3, 3), '3_3')
    conv = layers.add([conv, conv1])

    conv1 = conv_bn_prelu(conv, (256, 3, 3), '3_1b')
    conv = conv_bn_prelu(conv1, (256, 3, 3), '3_2b')
    conv = conv_bn_prelu(conv, (256, 3, 3), '3_3b')
    conv = layers.add([conv, conv1])

    conv_block2 = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(conv)
    # ==========================

    # =======Block 3 ========
    conv1 = conv_bn_prelu(conv_block2, (512, 3, 3), '3_1c')
    conv = conv_bn_prelu(conv1, (512, 3, 3), '3_2c')
    conv = conv_bn_prelu(conv, (512, 3, 3), '3_3c')
    conv = layers.add([conv, conv1])
    conv_block3 = conv_bn_prelu(conv, (256, 3, 3), '3_4c')

    #conv_block3 = MaxPooling2D(pool_size=(2,2),strides=(2,2))(conv)
    # ==========================

    # multi-scale ASPP
    level_2 = conv_bn_prelu(conv_block3, (256, 3, 3),
                            '4_1', dilation_rate=(1, 1))
    ori_1 = conv_bn_prelu(level_2, (128, 1, 1), 'ori_1_1')
    ori_1 = layers.Conv2D(90, (1, 1), padding='same', name='ori_1_2')(ori_1)
    seg_1 = conv_bn_prelu(level_2, (128, 1, 1), 'seg_1_1')
    seg_1 = layers.Conv2D(1, (1, 1), padding='same', name='seg_1_2')(seg_1)

    level_3 = conv_bn_prelu(conv_block2, (256, 3, 3),
                            '4_2', dilation_rate=(4, 4))
    ori_2 = conv_bn_prelu(level_3, (128, 1, 1), 'ori_2_1')
    ori_2 = layers.Conv2D(90, (1, 1), padding='same', name='ori_2_2')(ori_2)
    seg_2 = conv_bn_prelu(level_3, (128, 1, 1), 'seg_2_1')
    seg_2 = layers.Conv2D(1, (1, 1), padding='same', name='seg_2_2')(seg_2)

    level_4 = conv_bn_prelu(conv_block2, (256, 3, 3),
                            '4_3', dilation_rate=(8, 8))
    ori_3 = conv_bn_prelu(level_4, (128, 1, 1), 'ori_3_1')
    ori_3 = layers.Conv2D(90, (1, 1), padding='same', name='ori_3_2')(ori_3)
    seg_3 = conv_bn_prelu(level_4, (128, 1, 1), 'seg_3_1')
    seg_3 = layers.Conv2D(1, (1, 1), padding='same', name='seg_3_2')(seg_3)

    # sum fusion for ori
    ori_out = layers.Lambda(coarse_net_utils.merge_sum,
                           output_shape=lambda input_shape: input_shape[0])([ori_1, ori_2, ori_3])
    ori_out_1 = layers.Activation('sigmoid', name='ori_out_1')(ori_out)
    ori_out_2 = layers.Activation('sigmoid', name='ori_out_2')(ori_out)

    # sum fusion for segmentation
    seg_out = layers.Lambda(coarse_net_utils.merge_sum,
                           output_shape=lambda input_shape: input_shape[0])([seg_1, seg_2, seg_3])
    seg_out = layers.Activation('sigmoid', name='seg_out')(seg_out)

    # ----------------------------------------------------------------------------
    # enhance part
    filters_cos, filters_sin = minutiae_net_utils.gabor_bank(stride=2, lambda_value=8)

    # Create Conv2D layers without weights parameter (Keras 3.x compatibility)
    filter_img_real = layers.Conv2D(
        filters_cos.shape[3],
        (filters_cos.shape[0],
         filters_cos.shape[1]),
        kernel_initializer='zeros',
        bias_initializer='zeros',
        trainable=False,
        padding='same', name='enh_img_real_1')(img_input)
    
    filter_img_imag = layers.Conv2D(
        filters_sin.shape[3],
        (filters_sin.shape[0],
         filters_sin.shape[1]),
        kernel_initializer='zeros',
        bias_initializer='zeros',
        trainable=False,
        padding='same', name='enh_img_imag_1')(img_input)

    ori_peak = layers.Lambda(coarse_net_utils.ori_highest_peak, 
                            output_shape=lambda input_shape: input_shape)(ori_out_1)
    ori_peak = layers.Lambda(coarse_net_utils.select_max,
                            output_shape=lambda input_shape: input_shape)(
        ori_peak)  # select max ori and set it to 1

    # Use this function to upsample image
    upsample_ori = layers.UpSampling2D(size=(8, 8))(ori_peak)
    seg_round = layers.Activation('softsign')(seg_out)

    upsample_seg = layers.UpSampling2D(size=(8, 8))(seg_round)
    mul_mask_real = layers.Lambda(coarse_net_utils.merge_mul,
                                 output_shape=lambda input_shape: input_shape[0])(
        [filter_img_real, upsample_ori])

    enh_img_real = layers.Lambda(coarse_net_utils.reduce_sum,
                                output_shape=lambda input_shape: input_shape[:-1] + (1,),
                                name='enh_img_real_2')(mul_mask_real)
    mul_mask_imag = layers.Lambda(coarse_net_utils.merge_mul,
                                 output_shape=lambda input_shape: input_shape[0])(
        [filter_img_imag, upsample_ori])

    enh_img_imag = layers.Lambda(coarse_net_utils.reduce_sum,
                                output_shape=lambda input_shape: input_shape[:-1] + (1,),
                                name='enh_img_imag_2')(mul_mask_imag)
    enh_img = layers.Lambda(coarse_net_utils.atan2, 
                           output_shape=lambda input_shape: input_shape[0][:-1] + (1,),
                           name='phase_img')(
        [enh_img_imag, enh_img_real])

    enh_seg_img = layers.Lambda(coarse_net_utils.merge_concat,
                               output_shape=lambda input_shape: input_shape[0][:-1] + (input_shape[0][-1] + input_shape[1][-1],),
                               name='phase_seg_img')(
        [enh_img, upsample_seg])
    # ----------------------------------------------------------------------------
    # mnt part
    # =======Block 1 ========
    mnt_conv1 = conv_bn_prelu(enh_seg_img, (64, 9, 9), 'mnt_1_1')
    mnt_conv = conv_bn_prelu(mnt_conv1, (64, 9, 9), 'mnt_1_2')
    mnt_conv = conv_bn_prelu(mnt_conv, (64, 9, 9), 'mnt_1_3')
    mnt_conv = layers.add([mnt_conv, mnt_conv1])

    mnt_conv1 = conv_bn_prelu(mnt_conv, (64, 9, 9), 'mnt_1_1b')
    mnt_conv = conv_bn_prelu(mnt_conv1, (64, 9, 9), 'mnt_1_2b')
    mnt_conv = conv_bn_prelu(mnt_conv, (64, 9, 9), 'mnt_1_3b')
    mnt_conv = layers.add([mnt_conv, mnt_conv1])

    mnt_conv = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(mnt_conv)
    # ==========================

    # =======Block 2 ========
    mnt_conv1 = conv_bn_prelu(mnt_conv, (128, 5, 5), 'mnt_2_1')
    mnt_conv = conv_bn_prelu(mnt_conv1, (128, 5, 5), 'mnt_2_2')
    mnt_conv = conv_bn_prelu(mnt_conv, (128, 5, 5), 'mnt_2_3')
    mnt_conv = layers.add([mnt_conv, mnt_conv1])

    mnt_conv1 = conv_bn_prelu(mnt_conv, (128, 5, 5), 'mnt_2_1b')
    mnt_conv = conv_bn_prelu(mnt_conv1, (128, 5, 5), 'mnt_2_2b')
    mnt_conv = conv_bn_prelu(mnt_conv, (128, 5, 5), 'mnt_2_3b')
    mnt_conv = layers.add([mnt_conv, mnt_conv1])

    mnt_conv = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(mnt_conv)
    # ==========================

    # =======Block 3 ========
    mnt_conv1 = conv_bn_prelu(mnt_conv, (256, 3, 3), 'mnt_3_1')
    mnt_conv2 = conv_bn_prelu(mnt_conv1, (256, 3, 3), 'mnt_3_2')
    mnt_conv3 = conv_bn_prelu(mnt_conv2, (256, 3, 3), 'mnt_3_3')
    mnt_conv3 = layers.add([mnt_conv3, mnt_conv1])
    mnt_conv4 = conv_bn_prelu(mnt_conv3, (256, 3, 3), 'mnt_3_4')
    mnt_conv4 = layers.add([mnt_conv4, mnt_conv2])

    mnt_conv = layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(mnt_conv4)
    # ==========================

    mnt_o_1 = layers.Lambda(coarse_net_utils.merge_concat,
                           output_shape=lambda input_shape: input_shape[0][:-1] + (input_shape[0][-1] + input_shape[1][-1],))([mnt_conv, ori_out_1])
    mnt_o_2 = conv_bn_prelu(mnt_o_1, (256, 1, 1), 'mnt_o_1_1')
    mnt_o_3 = layers.Conv2D(180, (1, 1), padding='same', name='mnt_o_1_2')(mnt_o_2)
    mnt_o_out = layers.Activation('sigmoid', name='mnt_o_out')(mnt_o_3)

    mnt_w_1 = conv_bn_prelu(mnt_conv, (256, 1, 1), 'mnt_w_1_1')
    mnt_w_2 = layers.Conv2D(8, (1, 1), padding='same', name='mnt_w_1_2')(mnt_w_1)
    mnt_w_out = layers.Activation('sigmoid', name='mnt_w_out')(mnt_w_2)

    mnt_h_1 = conv_bn_prelu(mnt_conv, (256, 1, 1), 'mnt_h_1_1')
    mnt_h_2 = layers.Conv2D(8, (1, 1), padding='same', name='mnt_h_1_2')(mnt_h_1)
    mnt_h_out = layers.Activation('sigmoid', name='mnt_h_out')(mnt_h_2)

    mnt_s_1 = conv_bn_prelu(mnt_conv, (256, 1, 1), 'mnt_s_1_1')
    mnt_s_2 = layers.Conv2D(1, (1, 1), padding='same', name='mnt_s_1_2')(mnt_s_1)
    mnt_s_out = layers.Activation('sigmoid', name='mnt_s_out')(mnt_s_2)

    if mode == 'deploy':
        model = models.Model(
            inputs=[
                img_input,
            ],
            outputs=[
                enh_img,
                enh_img_imag,
                enh_img_real,
                ori_out_1,
                ori_out_2,
                seg_out,
                mnt_o_out,
                mnt_w_out,
                mnt_h_out,
                mnt_s_out])
    else:
        model = models.Model(
            inputs=[
                img_input,
            ],
            outputs=[
                ori_out_1,
                ori_out_2,
                seg_out,
                mnt_o_out,
                mnt_w_out,
                mnt_h_out,
                mnt_s_out])

    if weights_path is not None:
        model.load_weights(weights_path, by_name=True)
    
    # Set Gabor filter weights after model is built (Keras 3.x compatibility)
    try:
        # Get the Gabor filters
        filters_cos, filters_sin = minutiae_net_utils.gabor_bank(stride=2, lambda_value=8)
        
        # Set weights for the real part filter
        real_layer = model.get_layer('enh_img_real_1')
        real_layer.set_weights([filters_cos, np.zeros([filters_cos.shape[3]])])
        
        # Set weights for the imaginary part filter  
        imag_layer = model.get_layer('enh_img_imag_1')
        imag_layer.set_weights([filters_sin, np.zeros([filters_sin.shape[3]])])
        
    except Exception as e:
        print(f"Warning: Could not set Gabor filter weights: {e}")
    
    return model


def train(
        train_set=None, output_dir='../output_CoarseNet/' + datetime.now().strftime('%Y%m%d-%H%M%S'),
        pretrain_dir=None, batch_size=1, test_set=None, learning_config=None, logging=None):

    img_name, folder_name, img_size = coarse_net_utils.get_maximum_img_size_and_names(
        train_set, None)

    main_net_model = get_coarse_net_model(
        (img_size[0], img_size[1], 1), pretrain_dir, 'train')
    # Save model architecture
    utils.plot_model(main_net_model, to_file=output_dir +
                     '/model.png', show_shapes=True)

    main_net_model.compile(
        optimizer=learning_config,
        loss={'seg_out': loss_functions.segmentation_loss, 'mnt_o_out': loss_functions.orientation_output_loss,
              'mnt_w_out': loss_functions.orientation_output_loss, 'mnt_h_out': loss_functions.orientation_output_loss,
              'mnt_s_out': loss_functions.minutiae_score_loss},
        loss_weights={'seg_out': .5, 'mnt_w_out': .5, 'mnt_h_out': .5, 'mnt_o_out': 100.,
                      'mnt_s_out': 50.},
        metrics={'seg_out': [coarse_net_utils.seg_acc_pos, coarse_net_utils.seg_acc_neg, coarse_net_utils.seg_acc_all],
                 'mnt_o_out': [coarse_net_utils.mnt_acc_delta_10, ],
                 'mnt_w_out': [coarse_net_utils.mnt_mean_delta, ],
                 'mnt_h_out': [coarse_net_utils.mnt_mean_delta, ],
                 'mnt_s_out': [coarse_net_utils.seg_acc_pos, coarse_net_utils.seg_acc_neg, coarse_net_utils.seg_acc_all]})

    writer = tf.compat.v1.summary.FileWriter(output_dir)

    best_f1_result = 0
    best_loss = 10000000
    for epoch in range(1000):
        outdir = "%s/saved_best_loss/" % (output_dir)
        minutiae_net_utils.mkdir(outdir)

        for i, train_step in enumerate(
            coarse_net_utils.load_data(
                (img_name, folder_name, img_size),
                coarse_net_utils.get_tra_ori, rand=True, aug=0.7, batch_size=batch_size)):
            loss = main_net_model.train_on_batch(
                train_step[0],
                {'seg_out': train_step[3],
                 'mnt_w_out': train_step[4],
                 'mnt_h_out': train_step[5],
                 'mnt_o_out': train_step[6],
                 'mnt_s_out': train_step[7]})
            # Save the lowest loss for easy converge
            if best_loss > loss[0]:
                savedir = "%s%s_%d_%s" % (outdir, str(epoch), i, str(loss[0]))
                main_net_model.save_weights(savedir, True)
                best_loss = loss[0]

            # Write log on screen at every 20 epochs
            if i % (2/batch_size) == 0:
                logging.info("epoch=%d, step=%d", epoch, i)
                # Write details loss
                logging.info("%s", " ".join(
                    ["%s:%.4f\t" % (x) for x in zip(main_net_model.metrics_names, loss)]))
                # logging.info("Loss = %f    Best loss = %f",loss[0],Best_loss)

                # Show in tensorboard
                for name, value in zip(main_net_model.metrics_names, loss):
                    summary = tf.compat.v1.Summary(
                        value=[tf.compat.v1.Summary.Value(tag=name, simple_value=value), ])
                    writer.add_summary(summary, i)

        # Evaluate every 5 epoch: for faster training
        if epoch % 10 == 0:
            outdir = "%s/saved_models/" % (output_dir)
            minutiae_net_utils.mkdir(outdir)
            savedir = "%s%s" % (outdir, str(epoch))
            main_net_model.save_weights(savedir, True)

            for folder in test_set:
                (precision_test, recall_test, f1_test, precision_test_location, recall_test_location,
                 f1_test_location) = evaluate_training(savedir, [folder, ], logging=logging)

            summary = tf.compat.v1.Summary(
                value=[tf.compat.v1.Summary.Value(
                    tag="Precision", simple_value=precision_test),
                    tf.compat.v1.Summary.Value(
                    tag="Recall", simple_value=recall_test),
                    tf.compat.v1.Summary.Value(tag="F1", simple_value=f1_test),
                    tf.compat.v1.Summary.Value(
                    tag="Location Precision",
                    simple_value=precision_test_location),
                    tf.compat.v1.Summary.Value(
                    tag="Location Recall",
                    simple_value=recall_test_location),
                    tf.compat.v1.Summary.Value(
                    tag="Location F1", simple_value=f1_test_location), ])
            writer.add_summary(summary, epoch)

        # Only save the best result
        if f1_test > best_f1_result:
            best_f1_result = f1_test
        # else:
        #     os.remove(savedir)

    writer.close()


def evaluate_training(model_dir, test_set, logging=None, finenet_path=None):
    logging.info("Evaluating %s:", test_set)

    # Prepare input info
    img_name, folder_name, img_size = coarse_net_utils.get_maximum_img_size_and_names(test_set)

    main_net_model = get_coarse_net_model((None, None, 1), model_dir, 'test')

    ave_prf_nms, ave_prf_nms_location = [], []

    if finenet_path is not None:
        # ====== Load FineNet to verify
        model_finenet = fine_net_model.get_fine_net_model(num_classes=2,
                                                          pretrained_path=finenet_path,
                                                          input_shape=(224, 224, 3))

        model_finenet.compile(loss='categorical_crossentropy',
                              optimizer=optimizers.Adam(lr=0),
                              metrics=['accuracy'])

    for _, test in enumerate(
        coarse_net_utils.load_data(
            (img_name, folder_name, img_size),
            coarse_net_utils.get_tra_ori, rand=False, aug=0.0, batch_size=1)):

        # logging.info("%d / %d: %s"%(j+1, len(img_name), img_name[j]))
        _, _, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = main_net_model.predict(
            test[0])
        mnt_gt = coarse_net_utils.label2mnt(test[7], test[4], test[5], test[6])

        original_image = test[0].copy()

        mnt_s_out = mnt_s_out * seg_out

        # Does not useful to use this while training
        final_minutiae_score_threashold = 0.45
        early_minutiae_thres = final_minutiae_score_threashold + 0.05
        is_having_finenet = False

        # In cases of small amount of minutiae given, try adaptive threshold
        while final_minutiae_score_threashold >= 0:
            mnt = coarse_net_utils.label2mnt(
                mnt_s_out, mnt_w_out, mnt_h_out, mnt_o_out, thresh=early_minutiae_thres)
            # Previous exp: 0.2
            mnt_nms_1 = minutiae_net_utils.py_cpu_nms(mnt, 0.5)
            mnt_nms_2 = minutiae_net_utils.nms(mnt)
            # Make sure good result is given
            if mnt_nms_1.shape[0] > 4 and mnt_nms_2.shape[0] > 4:
                break
            else:
                final_minutiae_score_threashold = final_minutiae_score_threashold - 0.05
                early_minutiae_thres = early_minutiae_thres - 0.05

        mnt_nms = minutiae_net_utils.fuse_nms(mnt_nms_1, mnt_nms_2)

        mnt_nms = mnt_nms[mnt_nms[:, 3] > early_minutiae_thres, :]
        mnt_refined = []
        if is_having_finenet is True:
            # ======= Verify using FineNet ============
            patch_minu_radio = 22
            if finenet_path is not None:
                for idx_minu in range(mnt_nms.shape[0]):
                    try:
                        # Extract patch from image
                        x_begin = int(mnt_nms[idx_minu, 1]) - patch_minu_radio
                        y_begin = int(mnt_nms[idx_minu, 0]) - patch_minu_radio
                        patch_minu = original_image[x_begin:x_begin + 2 * patch_minu_radio,
                                                    y_begin:y_begin + 2 * patch_minu_radio]

                        patch_minu = cv2.resize(
                            patch_minu, dsize=(224, 224),
                            interpolation=cv2.INTER_NEAREST)

                        ret = np.empty(
                            (patch_minu.shape[0],
                             patch_minu.shape[1],
                             3),
                            dtype=np.uint8)
                        ret[:, :, 0] = patch_minu
                        ret[:, :, 1] = patch_minu
                        ret[:, :, 2] = patch_minu
                        patch_minu = ret
                        patch_minu = np.expand_dims(patch_minu, axis=0)

                        # # Can use class as hard decision
                        # # 0: minu  1: non-minu
                        # [class_Minutiae] = np.argmax(model_finenet.predict(patch_minu), axis=1)
                        #
                        # if class_Minutiae == 0:
                        #     mnt_refined.append(mnt_nms[idx_minu,:])

                        # Use soft decision: merge FineNet score with CoarseNet score
                        [is_minutiae_prob] = model_finenet.predict(patch_minu)
                        is_minutiae_prob = is_minutiae_prob[0]
                        # print is_minutiae_prob
                        tmp_mnt = mnt_nms[idx_minu, :].copy()
                        tmp_mnt[3] = (4 * tmp_mnt[3] + is_minutiae_prob) / 5
                        mnt_refined.append(tmp_mnt)

                    # TODO : fix exception
                    except:  # pylint: disable=bare-except
                        mnt_refined.append(mnt_nms[idx_minu, :])
        else:
            mnt_refined = mnt_nms

        mnt_nms = np.array(mnt_refined)

        if mnt_nms.shape[0] > 0:
            mnt_nms = mnt_nms[mnt_nms[:, 3] > final_minutiae_score_threashold, :]

        p, r, f, l, o = minutiae_net_utils.metric_p_r_f(mnt_gt, mnt_nms, 16, np.pi/6)
        ave_prf_nms.append([p, r, f, l, o])
        p, r, f, l, o = minutiae_net_utils.metric_p_r_f(mnt_gt, mnt_nms, 16, np.pi)
        ave_prf_nms_location.append([p, r, f, l, o])

    logging.info("Average testing results:")
    ave_prf_nms = np.mean(np.array(ave_prf_nms), 0)
    ave_prf_nms_location = np.mean(np.array(ave_prf_nms_location), 0)
    logging.info(
        "Precision: %f\tRecall: %f\tF1-measure: %f\tLocation_dis: %f\tOrientation_delta:%f\n----------------\n",
        ave_prf_nms[0],
        ave_prf_nms[1],
        ave_prf_nms[2],
        ave_prf_nms[3],
        ave_prf_nms[4])

    return (ave_prf_nms[0], ave_prf_nms[1], ave_prf_nms[2],
            ave_prf_nms_location[0], ave_prf_nms_location[1], ave_prf_nms_location[2])


def fuse_minu_orientation(dir_map, mnt, mode=1, block_size=16):
    # mode is the way to fuse output minutiae with orientation
    # 1: use orientation; 2: use minutiae; 3: fuse average
    # blk_h, blk_w = dir_map.shape
    dir_map = dir_map % (2 * np.pi)
    
    # Get dir_map dimensions for bounds checking
    dir_h, dir_w = dir_map.shape

    if mode == 1:
        for k in range(mnt.shape[0]):
            # Choose nearest orientation with proper rounding instead of truncation
            row_idx = int(np.round(mnt[k, 1] / block_size))
            col_idx = int(np.round(mnt[k, 0] / block_size))
            
            # Ensure indices are within bounds
            row_idx = max(0, min(row_idx, dir_h - 1))
            col_idx = max(0, min(col_idx, dir_w - 1))
            
            ori_value = dir_map[row_idx, col_idx]
            if mnt[k, 2] > 0 and mnt[k, 2] <= np.pi / 2:
                if ori_value > 0 and ori_value <= np.pi / 2:
                    mnt[k, 2] = ori_value
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    if (ori_value - mnt[k, 2]) < (np.pi -
                                                  ori_value + mnt[k, 2]):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    mnt[k, 2] = ori_value - np.pi
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    if (np.pi * 2 - ori_value +
                            mnt[k, 2]) < (ori_value - np.pi - mnt[k, 2]):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value - np.pi
            if np.pi / 2 < mnt[k, 2] and mnt[k, 2] <= np.pi:
                if ori_value > 0 and ori_value <= np.pi / 2:
                    if (mnt[k, 2] - ori_value) < (np.pi -
                                                  ori_value + mnt[k, 2]):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    mnt[k, 2] = ori_value
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    if (ori_value -
                        mnt[k, 2]) < (mnt[k, 2] -
                                      ori_value +
                                      np.pi):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value - np.pi
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    mnt[k, 2] = ori_value - np.pi
            if np.pi < mnt[k, 2] and mnt[k, 2] <= 3 * np.pi / 2:
                if ori_value > 0 and ori_value <= np.pi / 2:
                    mnt[k, 2] = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    if (mnt[k, 2] - ori_value) < (ori_value + np.pi - mnt[k, 2]):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    mnt[k, 2] = ori_value
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    if (ori_value -
                        mnt[k, 2]) < (mnt[k, 2] -
                                      ori_value +
                                      np.pi):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value - np.pi
            if 3 * np.pi / 2 < mnt[k, 2] and mnt[k, 2] <= 2 * np.pi:
                if ori_value > 0 and ori_value <= np.pi / 2:
                    if (np.pi -
                        mnt[k, 2] +
                        ori_value) < (mnt[k, 2] -
                                      np.pi -
                                      ori_value):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    mnt[k, 2] = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    if (mnt[k, 2] - ori_value) < (np.pi *
                                                  2 - mnt[k, 2] + ori_value - np.pi):
                        mnt[k, 2] = ori_value
                    else:
                        mnt[k, 2] = ori_value - np.pi
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    mnt[k, 2] = ori_value

    elif mode == 2:
        return
    elif mode == 3:
        for k in range(mnt.shape[0]):
            # Choose nearest orientation with proper rounding instead of truncation
            row_idx = int(np.round(mnt[k, 1] / block_size))
            col_idx = int(np.round(mnt[k, 0] / block_size))
            
            # Ensure indices are within bounds
            row_idx = max(0, min(row_idx, dir_h - 1))
            col_idx = max(0, min(col_idx, dir_w - 1))
            
            ori_value = dir_map[row_idx, col_idx]
            if mnt[k, 2] > 0 and mnt[k, 2] <= np.pi / 2:
                if ori_value > 0 and ori_value <= np.pi / 2:
                    fixed_ori = ori_value
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    if (ori_value - mnt[k, 2]) < (np.pi -
                                                  ori_value + mnt[k, 2]):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    fixed_ori = ori_value - np.pi
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    if (np.pi * 2 - ori_value +
                            mnt[k, 2]) < (ori_value - np.pi - mnt[k, 2]):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value - np.pi
            if np.pi / 2 < mnt[k, 2] and mnt[k, 2] <= np.pi:
                if ori_value > 0 and ori_value <= np.pi / 2:
                    if (mnt[k, 2] - ori_value) < (np.pi -
                                                  ori_value + mnt[k, 2]):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    fixed_ori = ori_value
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    if (ori_value -
                        mnt[k, 2]) < (mnt[k, 2] -
                                      ori_value +
                                      np.pi):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value - np.pi
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    fixed_ori = ori_value - np.pi
            if np.pi < mnt[k, 2] and mnt[k, 2] <= 3 * np.pi / 2:
                if ori_value > 0 and ori_value <= np.pi / 2:
                    fixed_ori = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    if (mnt[k, 2] - ori_value) < (ori_value + np.pi - mnt[k, 2]):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    fixed_ori = ori_value
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    if (ori_value -
                        mnt[k, 2]) < (mnt[k, 2] -
                                      ori_value +
                                      np.pi):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value - np.pi
            if 3 * np.pi / 2 < mnt[k, 2] and mnt[k, 2] <= 2 * np.pi:
                if ori_value > 0 and ori_value <= np.pi / 2:
                    if (np.pi -
                        mnt[k, 2] +
                        ori_value) < (mnt[k, 2] -
                                      np.pi -
                                      ori_value):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value + np.pi
                if np.pi / 2 < ori_value and ori_value <= np.pi:
                    fixed_ori = ori_value + np.pi
                if np.pi < ori_value and ori_value <= 3 * np.pi / 2:
                    if (mnt[k, 2] - ori_value) < (np.pi *
                                                  2 - mnt[k, 2] + ori_value - np.pi):
                        fixed_ori = ori_value
                    else:
                        fixed_ori = ori_value - np.pi
                if 3 * np.pi / 2 < ori_value and ori_value <= 2 * np.pi:
                    fixed_ori = ori_value

            mnt[k, 2] = (mnt[k, 2] + fixed_ori) / 2.0
    else:
        return


def deploy_with_gt(
        deploy_set, output_dir, model_path, finenet_path=None, set_name=None, logging=None):
    if set_name is None:
        set_name = deploy_set.split('/')[-2]

    # Read image and GT
    img_name, folder_name, img_size = coarse_net_utils.get_maximum_img_size_and_names(deploy_set)

    minutiae_net_utils.mkdir(output_dir + '/' + set_name + '/')
    minutiae_net_utils.mkdir(output_dir + '/' + set_name + '/mnt_results/')
    minutiae_net_utils.mkdir(output_dir + '/' + set_name + '/seg_results/')
    minutiae_net_utils.mkdir(output_dir + '/' + set_name + '/OF_results/')

    logging.info("Predicting %s:", set_name)

    is_having_finenet = False

    main_net_model = get_coarse_net_model((None, None, 1), model_path, mode='deploy')

    if is_having_finenet is True:
        # ====== Load FineNet to verify
        model_finenet = fine_net_model.get_fine_net_model(num_classes=2,
                                                          pretrained_path=finenet_path,
                                                          input_shape=(224, 224, 3))

        model_finenet.compile(loss='categorical_crossentropy',
                              optimizer=optimizers.Adam(lr=0),
                              metrics=['accuracy'])

    time_c = []
    ave_prf_nms = []
    for i, test in enumerate(
        coarse_net_utils.load_data(
            (img_name, folder_name, img_size),
            coarse_net_utils.get_tra_ori(), rand=False, aug=0.0, batch_size=1)):

        print(i, img_name[i])
        logging.info("%s %d / %d: %s", set_name, i + 1, len(img_name), img_name[i])
        time_start = time()

        image = cv2.imread(
            deploy_set + 'img_files/' + img_name[i] + '.bmp', cv2.IMREAD_GRAYSCALE)  # / 255.0
        mask = cv2.imread(
            deploy_set + 'seg_files/' + img_name[i] + '.bmp', cv2.IMREAD_GRAYSCALE) / 255.0

        img_size = image.shape
        img_size = np.array(img_size, dtype=np.int32) // 8 * 8
        image = image[:img_size[0], :img_size[1]]
        mask = mask[:img_size[0], :img_size[1]]

        original_image = image.copy()

        # Generate OF
        texture_img = minutiae_net_utils.fast_enhance_texture(image, sigma=2.5, show=False)
        dir_map, _ = minutiae_net_utils.get_maps_stft(
            texture_img, patch_size=64, block_size=16, preprocess=True)

        image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])
        # enh_img, enh_img_imag, enhance_img, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out
        _, _, _, _, _, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = main_net_model.predict(
            image)

        time_afterconv = time()

        # Use post processing to smooth image
        round_seg = np.round(np.squeeze(seg_out))
        seg_out = 1 - round_seg
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        seg_out = cv2.dilate(seg_out, kernel)

        # If use mask from outside
        # seg_out = cv2.resize(mask, dsize=(seg_out.shape[1], seg_out.shape[0]))

        mnt_gt = coarse_net_utils.label2mnt(test[7], test[4], test[5], test[6])

        final_minutiae_score_threashold = 0.45
        early_minutiae_thres = final_minutiae_score_threashold + 0.05

        # In cases of small amount of minutiae given, try adaptive threshold
        while final_minutiae_score_threashold >= 0:
            mnt = coarse_net_utils.label2mnt(
                np.squeeze(mnt_s_out) * np.round(np.squeeze(seg_out)),
                mnt_w_out, mnt_h_out, mnt_o_out, thresh=early_minutiae_thres)

            # Previous exp: 0.2
            mnt_nms_1 = minutiae_net_utils.py_cpu_nms(mnt, 0.5)
            mnt_nms_2 = minutiae_net_utils.nms(mnt)
            # Make sure good result is given
            if mnt_nms_1.shape[0] > 4 and mnt_nms_2.shape[0] > 4:
                break
            else:
                final_minutiae_score_threashold = final_minutiae_score_threashold - 0.05
                early_minutiae_thres = early_minutiae_thres - 0.05

        mnt_nms = minutiae_net_utils.fuse_nms(mnt_nms_1, mnt_nms_2)

        mnt_nms = mnt_nms[mnt_nms[:, 3] > early_minutiae_thres, :]
        mnt_refined = []
        if is_having_finenet is True:
            # ======= Verify using FineNet ============
            patch_minu_radio = 22
            if finenet_path is not None:
                for idx_minu in range(mnt_nms.shape[0]):
                    try:
                        # Extract patch from image
                        x_begin = int(mnt_nms[idx_minu, 1]) - patch_minu_radio
                        y_begin = int(mnt_nms[idx_minu, 0]) - patch_minu_radio
                        patch_minu = original_image[x_begin:x_begin + 2 * patch_minu_radio,
                                                    y_begin:y_begin + 2 * patch_minu_radio]

                        patch_minu = cv2.resize(
                            patch_minu, dsize=(224, 224),
                            interpolation=cv2.INTER_NEAREST)

                        ret = np.empty(
                            (patch_minu.shape[0],
                             patch_minu.shape[1],
                             3),
                            dtype=np.uint8)
                        ret[:, :, 0] = patch_minu
                        ret[:, :, 1] = patch_minu
                        ret[:, :, 2] = patch_minu
                        patch_minu = ret
                        patch_minu = np.expand_dims(patch_minu, axis=0)

                        # # Can use class as hard decision
                        # # 0: minu  1: non-minu
                        # [class_Minutiae] = np.argmax(model_finenet.predict(patch_minu), axis=1)
                        #
                        # if class_Minutiae == 0:
                        #     mnt_refined.append(mnt_nms[idx_minu,:])

                        # Use soft decision: merge FineNet score with CoarseNet score
                        [is_minutiae_prob] = model_finenet.predict(patch_minu)
                        is_minutiae_prob = is_minutiae_prob[0]
                        # print is_minutiae_prob
                        tmp_mnt = mnt_nms[idx_minu, :].copy()
                        tmp_mnt[3] = (4*tmp_mnt[3] + is_minutiae_prob) / 5
                        mnt_refined.append(tmp_mnt)

                    # TODO : fix exception
                    except:  # pylint: disable=bare-except
                        mnt_refined.append(mnt_nms[idx_minu, :])
        else:
            mnt_refined = mnt_nms

        mnt_nms = np.array(mnt_refined)

        if mnt_nms.shape[0] > 0:
            mnt_nms = mnt_nms[mnt_nms[:, 3] > final_minutiae_score_threashold, :]

        final_mask = ndimage.zoom(np.round(np.squeeze(seg_out)), [8, 8], order=0)

        # Show the orientation
        minutiae_net_utils.show_orientation_field(original_image, dir_map + np.pi, mask=final_mask,
                                                  fname="%s/%s/OF_results/%s_OF.jpg" %
                                                  (output_dir, set_name, img_name[i]))

        fuse_minu_orientation(dir_map, mnt_nms, mode=3)

        time_afterpost = time()
        minutiae_net_utils.mnt_writer(mnt_nms, img_name[i], img_size, "%s/%s/mnt_results/%s.mnt" %
                                      (output_dir, set_name, img_name[i]))
        minutiae_net_utils.draw_minutiae_overlay_with_score(
            image, mnt_nms, mnt_gt[:, : 3],
            "%s/%s/%s_minu.jpg" % (output_dir, set_name, img_name[i]),
            saveimage=True)
        # misc.imsave("%s/%s/%s_score.jpg"%(output_dir, set_name, img_name[i]), np.squeeze(mnt_s_out_upscale))

        cv2.imwrite("%s/%s/seg_results/%s_seg.jpg" %
                    (output_dir, set_name, img_name[i]), final_mask)

        time_afterdraw = time()
        time_c.append([time_afterconv - time_start, time_afterpost - time_afterconv,
                      time_afterdraw - time_afterpost])
        logging.info("load+conv: %.3fs, seg-postpro+nms: %.3f, draw: %.3f",
                     time_c[-1][0], time_c[-1][1], time_c[-1][2])

        # Metrics calculating
        p, r, f, l, o = minutiae_net_utils.metric_p_r_f(mnt_gt, mnt_nms)
        ave_prf_nms.append([p, r, f, l, o])
        print(p, r, f)

    time_c = np.mean(np.array(time_c), axis=0)
    ave_prf_nms = np.mean(np.array(ave_prf_nms), 0)
    print("Precision: %f\tRecall: %f\tF1-measure: %f" %
          (ave_prf_nms[0], ave_prf_nms[1], ave_prf_nms[2]))

    logging.info("Average: load+conv: %.3fs, oir-select+seg-post+nms: %.3f, draw: %.3f",
                 time_c[0], time_c[1], time_c[2])
    return


def inference(
        deploy_set, output_dir, model_path, finenet_path=None, set_name=None, file_ext='.bmp',
        is_having_finenet=False, logging=None):
    if set_name is None:
        set_name = deploy_set.split('/')[-2]

    minutiae_net_utils.mkdir(output_dir + '/' + set_name + '/')
    minutiae_net_utils.mkdir(output_dir + '/' + set_name + '/mnt_results/')
    minutiae_net_utils.mkdir(output_dir + '/' + set_name + '/seg_results/')
    minutiae_net_utils.mkdir(output_dir + '/' + set_name + '/OF_results/')

    logging.info("Predicting %s:", set_name)

    _, img_name = minutiae_net_utils.get_files_in_folder(deploy_set + 'img_files/', file_ext)
    print(deploy_set)

    # ====== Load FineNet to verify
    if is_having_finenet is True:
        model_finenet = fine_net_model.get_fine_net_model(num_classes=2,
                                                          pretrained_path=finenet_path,
                                                          input_shape=(224, 224, 3))

        model_finenet.compile(loss='categorical_crossentropy',
                              optimizer=optimizers.Adam(lr=0),
                              metrics=['accuracy'])

    time_c = []

    main_net_model = get_coarse_net_model((None, None, 1), model_path, mode='deploy')

    for i in enumerate(img_name):
        print(i)

        image = cv2.imread(
            deploy_set + 'img_files/' + img_name[i] + file_ext, cv2.IMREAD_GRAYSCALE)  # / 255.0

        img_size = image.shape
        img_size = np.array(img_size, dtype=np.int32) // 8 * 8

        # read the mask from files
        try:
            mask = cv2.imread(
                deploy_set + 'seg_files/' + img_name[i] + '.jpg', cv2.IMREAD_GRAYSCALE) / 255.0

        except:  # pylint: disable=bare-except
            mask = np.ones((img_size[0], img_size[1]))

        image = image[:img_size[0], :img_size[1]]
        mask = mask[:img_size[0], :img_size[1]]

        original_image = image.copy()

        texture_img = minutiae_net_utils.fast_enhance_texture(image, sigma=2.5, show=False)
        dir_map, _ = minutiae_net_utils.get_maps_stft(
            texture_img, patch_size=64, block_size=16, preprocess=True)

        image = image*mask

        logging.info("%s %d / %d: %s", set_name, i + 1, len(img_name), img_name[i])
        time_start = time()

        image = np.reshape(image, [1, image.shape[0], image.shape[1], 1])
        # enh_img, enh_img_imag, enhance_img, ori_out_1, ori_out_2, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out
        _, _, _, _, _, seg_out, mnt_o_out, mnt_w_out, mnt_h_out, mnt_s_out = main_net_model.predict(
            image)
        time_afterconv = time()

        # If use mask from model
        round_seg = np.round(np.squeeze(seg_out))
        seg_out = 1 - round_seg
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_CLOSE, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
        seg_out = cv2.morphologyEx(seg_out, cv2.MORPH_OPEN, kernel)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        seg_out = cv2.dilate(seg_out, kernel)

        # If use mask from outside
        # seg_out = cv2.resize(mask, dsize=(seg_out.shape[1], seg_out.shape[0]))

        max_num_minu = 20
        min_num_minu = 6

        early_minutiae_thres = 0.5

        # New adaptive threshold
        mnt = coarse_net_utils.label2mnt(np.squeeze(mnt_s_out) * np.round(np.squeeze(seg_out)),
                                         mnt_w_out, mnt_h_out, mnt_o_out, thresh=0)

        # Previous exp: 0.2
        mnt_nms_1 = minutiae_net_utils.py_cpu_nms(mnt, 0.5)
        mnt_nms_2 = minutiae_net_utils.nms(mnt)
        mnt_nms_1.view('f8,f8,f8,f8').sort(order=['f3'], axis=0)
        mnt_nms_1 = mnt_nms_1[::-1]

        mnt_nms_1_copy = mnt_nms_1.copy()
        mnt_nms_2_copy = mnt_nms_2.copy()
        # Adaptive threshold goes here
        # Make sure the maximum number of minutiae is max_num_minu

        # Sort minutiae by score
        while early_minutiae_thres > 0:
            mnt_nms_1 = mnt_nms_1_copy[mnt_nms_1_copy[:, 3] > early_minutiae_thres, :]
            mnt_nms_2 = mnt_nms_2_copy[mnt_nms_2_copy[:, 3] > early_minutiae_thres, :]

            if mnt_nms_1.shape[0] > max_num_minu or mnt_nms_2.shape[0] > max_num_minu:
                mnt_nms_1 = mnt_nms_1[:max_num_minu, :]
                mnt_nms_2 = mnt_nms_2[:max_num_minu, :]
            if mnt_nms_1.shape[0] > min_num_minu and mnt_nms_2.shape[0] > min_num_minu:
                break

            early_minutiae_thres = early_minutiae_thres - 0.05

        mnt_nms = minutiae_net_utils.fuse_nms(mnt_nms_1, mnt_nms_2)

        final_minutiae_score_threashold = early_minutiae_thres - 0.05

        print(early_minutiae_thres, final_minutiae_score_threashold)

        mnt_refined = []
        if is_having_finenet is True:
            # ======= Verify using FineNet ============
            patch_minu_radio = 22
            if finenet_path is not None:
                for idx_minu in range(mnt_nms.shape[0]):
                    try:
                        # Extract patch from image
                        x_begin = int(mnt_nms[idx_minu, 1]) - patch_minu_radio
                        y_begin = int(mnt_nms[idx_minu, 0]) - patch_minu_radio
                        patch_minu = original_image[x_begin:x_begin + 2 * patch_minu_radio,
                                                    y_begin:y_begin + 2 * patch_minu_radio]

                        patch_minu = cv2.resize(
                            patch_minu, dsize=(224, 224),
                            interpolation=cv2.INTER_NEAREST)

                        ret = np.empty(
                            (patch_minu.shape[0],
                             patch_minu.shape[1],
                             3),
                            dtype=np.uint8)
                        ret[:, :, 0] = patch_minu
                        ret[:, :, 1] = patch_minu
                        ret[:, :, 2] = patch_minu
                        patch_minu = ret
                        patch_minu = np.expand_dims(patch_minu, axis=0)

                        # # Can use class as hard decision
                        # # 0: minu  1: non-minu
                        # [class_Minutiae] = np.argmax(model_finenet.predict(patch_minu), axis=1)
                        #
                        # if class_Minutiae == 0:
                        #     mnt_refined.append(mnt_nms[idx_minu,:])

                        # Use soft decision: merge FineNet score with CoarseNet score
                        [is_minutiae_prob] = model_finenet.predict(patch_minu)
                        is_minutiae_prob = is_minutiae_prob[0]
                        # print is_minutiae_prob
                        tmp_mnt = mnt_nms[idx_minu, :].copy()
                        tmp_mnt[3] = (4*tmp_mnt[3] + is_minutiae_prob)/5
                        mnt_refined.append(tmp_mnt)

                    except:  # pylint: disable=bare-except
                        mnt_refined.append(mnt_nms[idx_minu, :])
        else:
            mnt_refined = mnt_nms

        # mnt_nms_backup = mnt_nms.copy()
        mnt_nms = np.array(mnt_refined)

        if mnt_nms.shape[0] > 0:
            mnt_nms = mnt_nms[mnt_nms[:, 3] > final_minutiae_score_threashold, :]

        final_mask = ndimage.zoom(np.round(np.squeeze(seg_out)), [8, 8], order=0)
        # Show the orientation
        minutiae_net_utils.show_orientation_field(original_image, dir_map + np.pi, mask=final_mask,
                                                  fname="%s/%s/OF_results/%s_OF.jpg" %
                                                  (output_dir, set_name, img_name[i]))

        fuse_minu_orientation(dir_map, mnt_nms, mode=3)

        time_afterpost = time()
        minutiae_net_utils.mnt_writer(mnt_nms, img_name[i], img_size, "%s/%s/mnt_results/%s.mnt" %
                                      (output_dir, set_name, img_name[i]))
        minutiae_net_utils.draw_minutiae(original_image, mnt_nms, "%s/%s/%s_minu.jpg" %
                                         (output_dir, set_name, img_name[i]), save_image=True)
        cv2.imwrite("%s/%s/seg_results/%s_seg.jpg" %
                    (output_dir, set_name, img_name[i]), final_mask)
        time_afterdraw = time()
        time_c.append([time_afterconv - time_start, time_afterpost -
                      time_afterconv, time_afterdraw - time_afterpost])
        logging.info("load+conv: %.3fs, seg-postpro+nms: %.3f, draw: %.3f",
                     time_c[-1][0], time_c[-1][1], time_c[-1][2])
    # time_c = np.mean(np.array(time_c), axis=0)
    # logging.info(
    #     "Average: load+conv: %.3fs, oir-select+seg-post+nms: %.3f, draw: %.3f" % (time_c[0], time_c[1], time_c[2]))
    return
