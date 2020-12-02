import sys
sys.path.insert(0, r'C:\Users\dcssaeb\Desktop\speed_prediction_in_road_network\pro\dataset')
    
import numpy as np
from SG_DATA import SG_DATA


def get_multiple_equal_batch_size(datas, batch_size):
    n = datas[0].shape[0]
    print('n = datas[0].shape[0]= ',n )
    n = n / batch_size
    n = int(n * batch_size)
    datas = [_d[-n:] for _d in datas]
    return datas


def get_train_test_data(conf, need_road_network_structure_matrix):
    
    data = SG_DATA(conf.observe_length, conf.predict_length, conf)
    xs, ys, arm, vehicle_type, engine_config, gen_weight = data.get_data()
    
    arm_shape = arm.shape

    xe = xs[3] #external factors
    xp = xs[1] #daily periodicity
    xt = xs[2] #not used
    xs = xs[0]
    
    
    #xs = xs.reshape(xs.shape[0], xs.shape[1], xs.shape[2],1)
    print('xs after reshape', xs.shape)


    arms = np.tile(arm, (xs.shape[0], 1, 1))
    train_xs, test_xs, train_ys, test_ys, train_arms, test_arms, \
    train_xp, test_xp, train_xt, test_xt, train_xe, test_xe, \
    train_vehicle_type, test_vehicle_type, train_engine_config, test_engine_config,\
    train_gen_weight, test_gen_weight = data.split(conf.test_ratio, [xs, ys, arms, xp, xt, xe, vehicle_type, engine_config, gen_weight]) #  #
    print('************* split  **********************')
    print('arms:', arms.shape, 'train_xs:',train_xs.shape,'test_xs:',test_xs.shape ,'train_ys:',train_ys.shape,'test_ys:',test_ys.shape, 'train and test _xp:', train_xp.shape, test_xp.shape, 'train and test _xt:', train_xt.shape, test_xt.shape, 'train and test _xe:', train_xe.shape, test_xe.shape)

    train_xs, train_ys, train_arms, train_xp, train_xt, train_xe, \
    train_vehicle_type, train_engine_config, train_gen_weight = get_multiple_equal_batch_size([train_xs, #
                                                                                        train_ys,
                                                                                        train_arms,
                                                                                        train_xp,
                                                                                        train_xt,
                                                                                        train_xe,
                                                                                        train_vehicle_type,
                                                                                        train_engine_config,
                                                                                        train_gen_weight],
                                                                                        conf.batch_size) #conf.batch_size

    test_xs, test_ys, test_arms, test_xp, test_xt, test_xe,\
    test_vehicle_type, test_engine_config, test_gen_weight = get_multiple_equal_batch_size([test_xs,   #test_xt,
                                                                                   test_ys,
                                                                                   test_arms,
                                                                                   test_xp,
                                                                                   test_xt,
                                                                                   test_xe,
                                                                                   test_vehicle_type,
                                                                                   test_engine_config,
                                                                                   test_gen_weight],
                                                                                   conf.batch_size) #conf.batch_size
    
    print('************* batch_size  **********************')
    print('arms:', train_arms.shape, 'train_xs:',train_xs.shape,'test_xs:',test_xs.shape ,'train_ys:',train_ys.shape,'test_ys:',test_ys.shape, 'train and test _xp:', train_xp.shape, test_xp.shape, 'train and test _xt:', train_xt.shape, test_xt.shape, 'train and test _xe:', train_xe.shape, test_xe.shape)
    
    
    if need_road_network_structure_matrix: #
        return data, arm_shape, train_xs, train_ys, train_arms, train_xp, train_xt, train_xe,\
         train_vehicle_type, train_engine_config, train_gen_weight,\
          test_xs, test_ys, test_arms, test_xp, test_xt, test_xe,\
          test_vehicle_type, test_engine_config, test_gen_weight
    else: #
        return data, arm_shape, train_xs, train_ys, None, train_xp, train_xt, train_xe,\
        train_vehicle_type, train_engine_config, train_gen_weight,\
         test_xs, test_ys, None, test_xp, test_xt, test_xe,\
         test_vehicle_type, test_engine_config, test_gen_weight



from config import Config
conf = Config("config_fig.yaml")
if __name__ == '__main__':
    data, arm_shape, train_xs, train_ys, train_arms, train_xp, train_xt, train_xe, \
    train_vehicle_type, train_engine_config, train_gen_weight,\
    test_xs, test_ys, test_arms, test_xp, test_xt, test_xe,\
    test_vehicle_type, test_engine_config, test_gen_weight = get_train_test_data(conf,need_road_network_structure_matrix=True)   #
