
from util import *


class MinMaxScalar(object):
    def __init__(self, _min=-1, _max=1):
        self._min = _min
        self._max = _max
        assert (self._max > self._min)
        self.min = 0
        self.max = 0
        self.s = 0
        self.x = 0
        self.is_fit = False

    def fit(self, x):
        if self.is_fit is False:
            self.min = np.min(x)
            self.max = np.max(x)
            self.z = (self.max - self.min) / float((self._max - self._min))
            self.x = (self.min + self.max - self.z * (self._min + self._max)) / 2
            self.is_fit = True

    def transform(self, x):
        if not self.is_fit:
            print("please fit first")
            exit(1)
        _x = x.copy()
        _x = (_x - self.x).astype(float) / self.z
        return _x

    def fit_transform(self, x):
        self.fit(x)
        return self.transform(x)

    def inverse_transform(self, x):
        if not self.is_fit:
            print("please fit first")
            exit(1)
        _x = x.copy()
        _x = _x.astype(float) * self.z + self.x
        return _x


class SG_DATA(object):
    def __init__(self, observe_length, predict_length, conf):
        self.data_name = "SG"
        self.observe_length = observe_length
        self.predict_length = predict_length
        self.min_max_scala = MinMaxScalar()
        self.conf = conf
        self.observe_p = self.conf.observe_p
        self.observe_t = self.conf.observe_t

    @performance
    def get_data(self, start_hour=8, end_hour=20, time_fill_split=0.5, road_fill_split=0.2, no_adjacent_fill_zero=True,stride_sparse=False, stride_edges=1, fix_adjacent_road_num=-1):
        
        
        stm, arm, t, speed, vehicle_type, engine_config, gen_weight = completion_data(conf.data_path)
        
        self.stm = stm
        self.arm = arm
        self.t = t
        stm = stm[:] 
        stm = self.min_max_scala.fit_transform(stm)

        #speed = self.min_max_scala.fit_transform(speed)

        stm = np.dstack((stm, speed))
        _i = 0
        _start = 0
        current = ""

        # externel data
        holiday = load_holiday(t, "SG_Holiday.txt")
        print('****************** External Information ****************************')
        if conf.use_externel:
            meteorol = load_meteorol(t, "SG_WEATHER.h5")
            print('meteorol (Sunny, Rainy, cloudy, etc..)', meteorol.shape)

        
        
        
        print('holiday shape {}'.format(holiday.shape))
        vec = timestamp2vec(t)
        print('vec shape (day of week 7, weekend/weekday 1, hour of day 13 {}'.format(vec.shape))
        
        

        if conf.use_externel:
            externel_data = np.hstack([holiday,vec,meteorol]) #,meteorol
        else:
            externel_data = np.hstack([holiday,vec])
            
        print('External data shape {}'.format(externel_data.shape))
        print('*********************************************************************')

        tt = []
        for _t in self.t:
            #_t = pd.to_datetime(_t, format='%Y-%m-%d %H:%M:%S').dt.strftime('%Y-%m-%d %H:%M:%S')
            tt.append(pd.to_datetime(_t, format='%d/%m/%Y %H:%M'))
        #print(tt)
        

        time_dict = dict(zip(tt, range(len(tt))))
        #print('time dic', time_dict)
        T = 24 * 60 / self.conf.time_window #T=96
        offset_frame = pd.DateOffset(minutes=self.conf.time_window)

        XC = []
        XP = []
        XT = []
        YS = []
        vehicle = []
        engine = []
        weight = []

        E = []
        for _t in tt:
            not_it = False
            indexs = []
            for _i in range(self.predict_length):
                
                _tt = _t + _i * offset_frame
                
                if (_tt in time_dict):
                    
                    indexs.append(time_dict[_tt])
                    #print(indexs)
                else:
                    not_it = True
            if not_it:
                continue
            y = stm[:, indexs, 0] #index energy out for y 
            YS.append(y)
            E.append(externel_data[indexs])
            # print indexs
        #should add here to change shape
        for _t in tt:
            indexs = []
            not_it = False
            for _i in range(self.observe_length, 0, -1):
                _tt = _t - _i * offset_frame
                if (_tt in time_dict):
                    indexs.append(time_dict[_tt])
                    # print('_tt in time_dict', _tt, _t, indexs)
                else:
                    not_it = True
            if not_it:
                continue
            xc = stm[:, indexs]
            XC.append(xc)
            vehicle.append(vehicle_type[:, indexs, :])
            engine.append(engine_config[:, indexs, :])
            weight.append(gen_weight[:, indexs, :])

            
            

        for _t in tt:
            indexs = []
            not_it = False
            for _i in range(self.observe_p, 0, -1):
                
                _tt = _t - _i * T * offset_frame
                
                if (_tt in time_dict):
                    indexs.append(time_dict[_tt])
                else:
                    not_it = True
            if not_it:
                continue
            xp = stm[:, indexs]
            XP.append(xp)
            # print indexs
        
        for _t in tt:
            not_it = False
            indexs = []
            for _i in range(self.observe_t, 0, -1):
                _tt = _t - _i * T * 7 * offset_frame 
                if (_tt in time_dict):
                    indexs.append(time_dict[_tt])
                    #print(indexs)
                else:
                    not_it = True
            if not_it:
                continue
            # print indexs
            xt = stm[:, indexs]
            XT.append(xt)
            
        
        
            
        
            
        
            
        
            
            
        
            
            
            
        YS = np.stack(YS, axis=0)
        print('YS shape(,edges,predict_length)', YS.shape)
        E = np.stack(E,axis=0)
        print('externel_data weather and holidays E.shape(,predict_length,)', E.shape)
        print('****************** Periodicity  Information ****************************')
        XC = np.stack(XC, axis=0)
        print('In-day periodicity XC.shape(nb_days,edges,timeslots_in_day)', XC.shape)
        XP = np.stack(XP, axis=0)
        print('Weekly periodicity XP.shape(49* 24 days because the first 7 days have not previous hitorical data,edges,7)', XP.shape)

        vehicle_type = np.stack(vehicle, axis=0)
        engine_config = np.stack(engine, axis=0)

        gen_weight = np.stack(weight, axis=0)
                    
        XT = np.stack(XT, axis=0) #weekly periodicity
        #print('periodicity XT.shape(,edges,)', XT.shape)
        print('*********************************************************************')
        if not no_adjacent_fill_zero:
            for _i in range(arm.shape[0]):
                _a = arm[_i]
                _a[_a[:] == arm.shape[0]] = _i
    
        #print(arm)
        
        return [XC, XP, XT, E], YS, arm, vehicle_type, engine_config, gen_weight #  XT, #

    def split(self, test_ratio, datas):
        n = datas[0].shape[0]
        return_datas = []
        test_size = int(n * test_ratio)
        for _d in datas:
            return_datas.append(_d[:-test_size])
            return_datas.append(_d[-test_size:])

        return return_datas



from config import Config
conf = Config("config_fig.yaml")
if __name__ == '__main__':
    data = SG_DATA(conf.observe_length,conf.predict_length, conf) # observe_length must be less than 12 or 49 (we have 12 (3 hours*(60/15) timeslot in a day), predict_length (4 = 1 hour, maximum possible 3 hours=12), conf
    xs, ys, arm, vehicle_type, engine_config, gen_weight = data.get_data() # ys,
