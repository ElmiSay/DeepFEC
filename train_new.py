
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.optimizers import SGD, Adam, Adadelta, Nadam, RMSprop

import os
import pandas as pd
import numpy as np

import time

from __init__ import get_train_test_data
from model.Factory import factory
from model.metrics import rmse, mape, mae, get_model_save_path
import argparse



from keras import backend as K
import tensorflow as tf




config = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1, allow_soft_placement=True, device_count = {'CPU': 1})
session = tf.Session(config=config)
K.set_session(session)



from keras.callbacks import TensorBoard
NAME = "RESNET_BILSTM_model"
tensorboard = TensorBoard(log_dir="logs/{}".format(NAME))


def main():
    from config import Config
    conf = Config("config_fig.yaml")
    print(conf.observe_length)
    # reading Data 
    data, arm_shape, train_xs, train_ys, train_arms, train_xp, train_xt, train_xe,\
    train_vehicle_type, train_engine_config, train_gen_weight,\
     test_xs, test_ys, test_arms, test_xp, test_xt, test_xe,\
     test_vehicle_type, test_engine_config, test_gen_weight = \
        get_train_test_data(conf,need_road_network_structure_matrix=True) # \ 
    print('************** Train - Predict **********')
    print('train_xs:', train_xs.shape,  'test_xs:', test_xs.shape, 'train_xp:', train_xp.shape, 'test_xp:', test_xp.shape, 'test_xe:', test_xe.shape, 'train_ys:', train_ys.shape, 'test_ys:', test_ys.shape)
    if conf.use_loopup:
        train_xs = [train_xs, train_arms]
        test_xs = [test_xs, test_arms]

    train_xs += [train_vehicle_type, train_engine_config, train_gen_weight]
    test_xs += [test_vehicle_type, test_engine_config, test_gen_weight]
    
    if conf.use_externel:
        if conf.observe_p != 0:
            if isinstance(train_xs, list):
                train_xs += [train_xp]
                test_xs += [test_xp]
            else:
                train_xs = [train_xs, train_xp]
                test_xs = [test_xs, test_xp]

        
        if conf.observe_t != 0:
            if isinstance(train_xs, list):
                train_xs += [train_xt]
                test_xs += [test_xt]
            else:
                train_xs = [train_xs, train_xt]
                test_xs = [test_xs, test_xt]
        
        if conf.observe_p != 0 or conf.observe_t != 0:
            train_xs += [train_xe]
            test_xs += [test_xe]
    

    #print('train arms:' )
    #print(test_xs[1])
    #print('train xs:' )
    #print(test_xs[0])

    
    # train_xs = [train_xp, train_xt, train_xe]
    # test_xs = [test_xp, test_xt, test_xe]    k

    # model weights save path
    model_save_path = get_model_save_path(conf)
    # get model
    print('arm shape', arm_shape)


    start = time.time()



    model = factory.RESNET_BILSTM_model(conf,arm_shape)

    if conf.use_cache_model and os.path.exists(model_save_path):
        model.load_weights(model_save_path)
    else:
        adam = Adam(lr=conf.learning_rate)
        model.compile(adam, loss="mean_squared_error", metrics=["mae", "mape", "accuracy"])
        early_stopping = EarlyStopping(monitor="val_loss",
                                       patience=conf.early_stopping)

        check_points = ModelCheckpoint(model_save_path,
                                       monitor="val_loss",
                                       save_best_only=False,
                                       save_weights_only=False)
        model.summary()
        
        with open('model_sumamry.txt','w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            model.summary(print_fn=lambda x: fh.write(x + '\n'))
            
        # training
        if conf.use_externel:
            train_ys = train_ys[:,:-1,:]
            test_ys = test_ys[:,:-1,:]
        else:
            train_ys = train_ys[:,:-1,:]
            test_ys = test_ys[:,:-1,:]
        
        print('train ys', test_ys.shape)
        history = model.fit(train_xs,
                            train_ys,
                            verbose=2,
                            epochs=conf.epochs,
                            batch_size=conf.batch_size,
                            callbacks=[early_stopping, check_points,tensorboard],
                            validation_data=[test_xs, test_ys])

        model.load_weights(model_save_path)

    # test
    predict = model.predict(test_xs, batch_size=conf.batch_size)

    predict = data.min_max_scala.inverse_transform(predict)
    y_true = data.min_max_scala.inverse_transform(test_ys)
    

    end = time.time()
    print("TRAINING TIME COST", (end - start))

    print("RMSE:", rmse(predict, y_true))
    print("MAE:", mae(predict, y_true))
    print("MAPE:", mape(predict, y_true))

    print("test")
    print(" predict")
    print(predict.shape)
    print(predict)
    print(" real")
    print(y_true.shape)
    print(y_true)

    

    '''Acc = abs(predict - y_true)
    print('Acc Shape', Acc.shape)
    
    
    
   
    with open('ACC.csv', 'w') as outfile:
        for data_slice in Acc:
            outfile.write('# New slice\n')
            np.savetxt(outfile, data_slice)
    with open('predicted.csv', 'w') as outfile:
        for data_slice in predict:
            outfile.write('# New slice\n')
            np.savetxt(outfile, data_slice)
    with open('real.csv', 'w') as outfile:
        for data_slice in y_true:
            outfile.write('# New slice\n')
            np.savetxt(outfile, data_slice)'''
            


if __name__ == '__main__':
    main()
