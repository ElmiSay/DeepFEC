


import pandas as pd
import numpy as np
import os
import re
import scipy.sparse as sp
import _pickle as cPickle
import time
import threading
import queue
import h5py


CACHE = "cache"
REGEX = re.compile(".*")


def performance(f):
    def fn(*args, **kwargs):
        start = time.time()
        r = f(*args, **kwargs)
        end = time.time()
        print("function {} cost {} s".format(f.__name__, (end - start)))
        print
        return r

    return fn


def load_all_RG(path="R_G.txt"):
    rg = pd.read_csv('R_G.txt', sep="\t", dtype={"link_id": int})
    return rg

def load_all_RG_node(path="data/R_G_node.csv"):
    rg_node = pd.read_csv("R_G_node.csv", dtype={"node_id": int}, usecols=['node_id', 'longitude', 'latitude'])
    return rg_node


'''
def load_part_RG(path, suffix="part"):
    rg_path = os.path.join(path, "R_G.csv")
    return pd.read_csv(rg_path, index_col=0)


def load_part_RG_node(path, suffix="part"):
    rg_node_path = os.path.join(path, "R_G_node_{}.csv".format(suffix))
    return pd.read_csv(rg_node_path, index_col=0)

'''


def get_node_id2id_dict(rg_node):
    node_ids = rg_node.node_id.unique()
    node_id2id_dict = dict(zip(node_ids, range(len(node_ids))))
    return node_id2id_dict


def get_id2node_id_dict(node_id2id_dict):
    return dict(zip(node_id2id_dict.values(), node_id2id_dict.keys()))


def get_s_e_id_of_edge(link_id, rg):
    nodes_list = rg[rg.link_id == link_id]["nodes"].str.split(',')
    s_id = nodes_list.str[0]
    e_id = nodes_list.str[-1]
    
    if len(nodes_list) == 0:
        return None, None
    
    return int(s_id), int(e_id)


def split_one_link_avg_speed(l, nodeid2id_dict, r, rg):
    items = l.strip().split(",") #eliminer les espaces et diviser le link (road)
    one_time = items[0] #on a un fichier contenant time|link_id|avg_speed qui s appele link_avg_speed.csv
    edgeid = int(items[1])
    avg_speed = float(items[2])
    
    rows = []
    datas = []
    cs = []
    
    
    s_id, e_id = get_s_e_id_of_edge(edgeid, rg)
    rows.append(nodeid2id_dict.get(s_id))
    cs.append(nodeid2id_dict.get(e_id))
    datas.append(avg_speed)
    '''
    for _i in range(1, len(items)):
        _v = items[_i]
        g = r.match(_v.strip()) #
        edgeid = int(g.group(1))
        avg_speed = float(g.group(2))
        #traj_num = float(g.group(3))
        s_id, e_id = get_s_e_id_of_edge(edgeid, rg)
        if s_id is None or e_id is None:
            continue
        rows.append(nodeid2id_dict.get(s_id))
        cs.append(nodeid2id_dict.get(e_id))
        datas.append(avg_speed)
    '''
    print('one_time, datas, rows, cs', one_time, datas, rows, cs)
    return one_time, datas, rows, cs 

'''
def in_time(c, start_hour, end_hour):
    t = pd.to_datetime(c, errors ='coerce')
    if t.hour >= start_hour and t.hour < end_hour:
        return True
    else:
        return False
'''

def get_time_day(c):
    return c[:10]


def load_raw_link_speed(path, n_jobs=1):
    
    link_path = os.path.join(path, "link_avg_speed.txt")
    rg = load_all_RG(path)
    rg_node = load_all_RG_node(path)
    nodeid2id_dict = get_node_id2id_dict(rg_node)
    time_list = []
    coo_matrix_list = []
    datass = []
    rowss = []
    columnss = []
    

    r = REGEX
    s = time.time()
    
    
    if n_jobs == 1:  # Single thread
        s = time.time()
        with open(link_path, "r") as f:
            for _index, l in enumerate(f):
                
                one_time, datas, rows, cs = split_one_link_avg_speed(l, nodeid2id_dict, r, rg)
                time_list.append(one_time)
                datass.append(datas)
                rowss.append(rows)
                columnss.append(cs)
    '''
    else:  # Multithreading
        queue = queue.queue(500)
        result = Queue.queue(n_jobs)

        class worker(threading.Thread):
            def __init__(self, queue, result):
                threading.Thread.__init__(self)
                self.queue = queue
                self.thread_stop = False
                self.result = result
                self.datass = []
                self.rowss = []
                self.times = []
                self.columnss = []

            def run(self):
                while not self.thread_stop:
                    try:
                        l = self.queue.get(block=True, timeout=20)
                    except queue.Empty:
                        self.thread_stop = True
                        self.result.put([self.times, self.datass, self.rowss, self.columnss])
                        break
                    one_time, datas, rows, cs = split_one_link_avg_speed(l, nodeid2id_dict, r, rg)
                    self.datass.append(datas)
                    self.times.append(one_time)
                    self.rowss.append(rows)
                    self.columnss.append(cs)
                    self.queue.task_done()

        ws = []
        for _ in range(n_jobs):
            w = worker(queue, result)
            w.start()
            ws.append(w)
        with open(link_path, "r") as f:
            for l in f:
                queue.put(l)
        for w in ws:  # waii all thread finish
            w.join()
        while not result.empty():
            try:
                item = result.get(block=False)
                time_list += item[0]
                datass += item[1]
                rowss += item[2]
                columnss += item[3]
            except Queue.Empty:
                break
        result.join()
    '''
    print("load from raw finish, spend {} s".format(time.time() - s))
    

    for _t, _d, _r, _c in zip(time_list, datass, rowss, columnss):
        coo_matrix = sp.coo_matrix((_d, (_r, _c)), shape=(len(nodeid2id_dict), len(nodeid2id_dict)))# une seule matrice pour t, on dirait graphe de s id rows et e id colum contenant speed comme data
        coo_matrix_list.append(coo_matrix) # coo matrix pour chaque t
    print('raw link speed;: ', time_list, coo_matrix_list)
    return time_list, coo_matrix_list


def load_raw_link_speed_in_time(path,
                                start_hour=8, end_hour=20,
                                remove_complete_day=True,
                                complete_ratio=0.9):
    time_interval = 15 # chaque 15 min, une heure est divisee par 4
    size = (end_hour - start_hour) + 1 * 60 / time_interval
    rg_node = load_all_RG_node(path)

    nodeid2id_dict = get_node_id2id_dict(rg_node)
    assert isinstance(start_hour, int)
    assert isinstance(end_hour, int)
    print("load_raw_link_speed_in_time ing..")
    

    nt = []
    ncoom = []
    
    s = time.time()
    t, coo_matrix = load_raw_link_speed(path) #cache eliminer et suffix aussi
    for _t, _c in zip(t, coo_matrix):
        #if in_time(_t, start_hour, end_hour):
        nt.append(_t)
        ncoom.append(_c)
    l = sorted(zip(nt, ncoom), key=lambda x: x[0])
    #print('this is l in load_raw_link_speed_in_time', l.shape, l)
    
    current_day = ""
    _nt = [] #time (in time)
    _ncoom = [] # matrices de vitesse pour tous les temps (in time)
    l += [("", "")]  # Can make the last day judge in the loop
    for _t, _m in l:
        _cd = get_time_day(_t)
        if _cd != current_day:
            # Determine if the previous data is relatively complete
            if remove_complete_day:
                if len(_nt) > size * complete_ratio:
                    nt += _nt
                    ncoom += _ncoom
                    # print "stay {}, {}/{}".format(current_day,len(_nt),size)
                else:
                    print("remove {}, just {}/{}".format(current_day, len(_nt), size))
            else:  # No need to remove
                nt += _nt
                ncoom += _ncoom
            _nt = [_t]
            _ncoom = [_m]
            current_day = _cd
        else:
            _ncoom.append(_m)
            _nt.append(_t)

    print("preprocess finish, spend {} s".format(time.time() - s)) 
    return nt, ncoom #time and whole matrice apres verification de complete day


def get_edgeid2id(rg):
    edgeids = sorted(list(rg["link_id"].unique()))
    # Map edgeid to a continuous key starting at 0
    keys = range(len(edgeids))
    edgeid2id = dict(zip(edgeids, keys))
    return edgeid2id


def get_id2edgeid(rg):
    #edgeids = sorted(list(rg["link_id"].unique()))
    edgeids = list(rg["link_id"].unique())
    # Map edgeid to a continuous key starting at 0
    keys = range(len(edgeids))
    id2edgeid = dict(zip(keys, edgeids))
    return id2edgeid


def split_one_link_avg_speed_by_road(l, speed_items, edgeid2id_dict, all_road_num):
    #veh id first loaded here
    one_time = l.strip() 
    datas = np.zeros((all_road_num,))
    vehicle_ids = np.zeros((all_road_num,), dtype=object)
    vehicle_ids.fill('') #initialize as empty string
    speeds = np.zeros((all_road_num,))
    for items in speed_items:
        if one_time == items[0]:

            edgeid = int(items[1])
            avg_speed = float(items[2])
            speed = float(items[3])
            vehicle_id = str(items[4])

            if edgeid in edgeid2id_dict:
                key = edgeid2id_dict[edgeid]
                datas[key] = avg_speed
                speeds[key] = speed
                vehicle_ids[key] = vehicle_id
    return one_time, datas, speeds, vehicle_ids#traj_nums


@performance
def load_raw_link_speed_by_road(path): #cache eliminer
    #2nd loaded veh id here
    link_path = os.path.join(path, "SG_time.txt")
    # rg = load_part_RG(path)
    rg = load_all_RG(path)
    
    edgeids = sorted(list(rg["link_id"].unique()))
    all_road_num = len(edgeids)
    # Map edgeid to a continuous key starting at 0
    edgeid2id = get_edgeid2id(rg)
    id2edgeid = get_id2edgeid(rg)

    # edgeid,speed,num regular

    time_list = []
    data_list = []
    vehicle_id_list = []
    speed_list = []
    link_path_speed = os.path.join(conf.data_path, "link_avg_speed.txt")
    #counter = 0
    df = pd.read_csv(link_path_speed, header=None)
    df[3] = df[3].fillna(df[3].mean()) #fill na by mean
    df[0] = df[0].map(str.strip)
    leng = len(df)
    groups = df.groupby(0, sort=False)

    print('speed items {}'.format(leng))

    with open(link_path, "r") as f:
        ls = list(map(str.strip, f.readlines()))

    for l, group in groups:#this cause problem, veh id not in groupby object.
        if l in ls:
            _speed_items = group.values
            one_time, datas, speed, vehicle_ids = split_one_link_avg_speed_by_road(l, _speed_items, edgeid2id, all_road_num) #, traj_nums
            time_list.append(one_time)
            data_list.append(datas)
            speed_list.append(speed)
            vehicle_id_list.append(vehicle_ids)

    print("load from raw finish")
        
    return time_list, data_list, speed_list, vehicle_id_list#    #, traj_num_list




@performance
def load_raw_link_speed_by_road_in_time(path,
                                        start_hour=8, end_hour=20,
                                        remove_complete_day=True,
                                        complete_ratio=0.9): #suffix, cache=True, sont elimine
    time_interval = 15
    size = (end_hour - start_hour) + 1 * 60 / time_interval  # How many time intervals are divided into
    assert isinstance(start_hour, int)
    assert isinstance(end_hour, int)
    print("Start load_raw_link_speed_by_road_in_time ing..")
    
    nt = []
    nd = []
    ns = []
    nv = []
    #nn = []
    
    t, datas, speed, vehicle = load_raw_link_speed_by_road(path) #, traj_nums
    #print('check if this empty or not from load_raw_link_speed_by_road ', t, datas)
    #print(" load_raw_link_speed_by_road_in_time ing DONE ..")

    
    for _t, _d, _s, _v in zip(t, datas, speed, vehicle): #, _n, traj_nums
        #if in_time(_t, start_hour, end_hour):
        nt.append(_t)
        nd.append(_d)
        ns.append(_s)
        nv.append(_v)
    #print("Sorting..")       
    l = sorted(zip(nt, nd, ns, nv), key=lambda x: x[0]) #, nn sort by time
    #print('this is t:     **** ',t)
    #print('this is datas:     **** ',datas)
    #print('nt et nd',  nt,nd)# nt ; toutes les dates et nd ; vector speed correspond a chaque date
    #print('sorted l zip(nt, nd):  ', l) # sorted by time, i.e., for 7am, we have speed vector of all roads
    #print(len(l))
    
    
    current_day = ""
    _nt = []
    _nd = []
    _ns = []
    _nv = []
    #_nn = []
    l += [("", "", "", "")]  #, "" #  Can make the last day judge in the loop 
    '''
    for _t, _m in l:  #, _n
        _cd = get_time_day(_t)
        if _cd != current_day:
            # Determine if the previous data is relatively complete
            if remove_complete_day:
                if len(_nt) > size * complete_ratio:
                    nt += _nt
                    nd += _nd
                    #nn += _nn
                    # print("stay {}, {}/{}".format(current_day,len(_nt),size))
                else:
                    print("remove {}, just {}/{}".format(current_day, len(_nt), size))
            else:  # No need to remove
                nt += _nt
                nd += _nd
                #nn += _nn
            _nt = [_t]
            _nd = [_m]
            #_nn = [_n]
            current_day = _cd
        else:
            _nd.append(_m)
            _nt.append(_t)
            #_nn.append(_n)
    '''
    for _t, _m, _s, _v in l:  #, _n
        _cd = get_time_day(_t)
        if _cd != current_day:
            # Determine if the previous data is relatively complete
            _nd.append(_m)
            _nt.append(_t)
            _ns.append(_s)
            _nv.append(_v)
    print("preprocess finish")
    
    return _nt, _nd, _ns, _nv   #, nn


def fill_by_time(_d):
    _n = _d.shape[0]
    start_index = 0
    _c = 0
    while _c < _n:
        fill_num = 0
        if _d[_c] == 0:  # Need to fill
            _c += 1
            fill_num += 1
            start_index = start_index - 1
            if start_index == -1:
                start_index = _n - 1
                while (_d[start_index] == 0):
                    fill_num += 1
                    start_index -= 1
                while _d[_c] == 0:
                    fill_num += 1
                    _c += 1

                _x = (_d[_c] - _d[start_index]) / float(fill_num + 1)
                _k = 1
                for _z in range(start_index + 1, _n):
                    _d[_z] = _d[start_index] + (_k * _x)
                    _k += 1
                for _z in range(_c):
                    _d[_z] = _d[start_index] + (_k * _x)
                    _k += 1
            else:
                while _c < _n and _d[_c] == 0:
                    fill_num += 1
                    _c += 1
                _x = (_d[_c % _n] - _d[start_index]) / (fill_num + 1)
                _k = 1
                for _z in range(start_index + 1, _c):
                    _d[_z] = _d[start_index] + _k * _x
                    _k += 1

            start_index = _c
        else:
            _c += 1
            start_index = _c
    return _d


def fill_by_road(d, _other):
    _n = d.shape[0]
    for _i in range(_n):
        _o = _other[:, _i]
        _ds = []
        for _d in _o:
            if _d != 0:
                _ds.append(_d)
        if len(_ds) != 0:
            _v = np.mean(_ds)
        else:
            _v = 0
        d[_i] = _v


def get_adjacent_edge_ids(rg, _choose_edge_id, start=True, end=True):  #
    s = rg["nodes"][rg.link_id.searchsorted(_choose_edge_id)].split(',')
    s_id = int(s[0])
    e_id = int(s[-1])

    adjacent_edge_ids = None
    
    
    #b= rg[int(rg['nodes'].str.split(',').str[0]) > 2]


    #result = result[(result['var']>0.25) or (result['var']<-0.25)]
    #if int(rg.nodes.str.split(',').str[-1]) == s_id :
    # adjacent_edge_ids = rg["link_id"].values
    
    if start and end:
        adjacent_edge_ids = rg[(rg.e_id == s_id) | (rg.s_id == e_id)]["link_id"].values
    elif start:
        adjacent_edge_ids = rg[(rg.e_id == s_id)]["link_id"].values
    elif end:
        adjacent_edge_ids = rg[(rg.s_id == e_id)]["link_id"].values
    
    return adjacent_edge_ids


@performance
def completion_data(path,
                    start_hour=8, end_hour=20,
                    time_fill_split=0.01, road_fill_split=0.2,
                    stride_sparse=False, stride_edges=1,
                    A=-1): #suffix, cache=True, eliminer
    if A != -1:
        fix_A = True
    else:
        fix_A = False
        A = 0

    
    #time_window = 15

    print("complete start")
    # Data completion
    t, d, speed, v = load_raw_link_speed_by_road_in_time(path, start_hour=start_hour, end_hour=end_hour)


    d = np.vstack(d).T #equal to d[np.newaxis, :]
    d = d.astype(float)
    
    speed = np.vstack(speed).T
    speed = speed.astype(float)
    #print('oooooooooooooooooo d apres vstack (only links having full data) :',  d.shape) #(73689, 84)
    time_num = d.shape[1] #84
    
    

    rg = load_all_RG(path).sort_values('link_id')
    s = rg['nodes'].str.split(',').str
    rg['s_id'] = pd.to_numeric(s[0])
    rg['e_id'] = pd.to_numeric(s[-1])
    id2edgeid = get_id2edgeid(rg)
    #edgeid2id = get_edgeid2id(rg)

    #print('id2edgeid',edgeid2id)
    
    
    dr = []
    choose = []

    
    for _d in d:
        
        dr.append((_d != 0).sum() / float(time_num))
    dr = np.asarray(dr)
    #print('dr shape',dr, dr.shape) # 
    for _i, _dr in enumerate(dr):
        
        if _dr >= time_fill_split:  # When the missing rate of this road is less than 0.3, fill by time
            choose.append(_i)
            _d = d[_i]
            fill_by_time(_d)
            #print('  DR  akber men time_fill_split hhhhhhhhhhhhhhhhh',  _dr) # all rows are not empty

    
    # Construct the final spatial-temporal matrix and adjacent road matrix
    choose = sorted(list(set(choose)))
    
    #print('sorted(list(set(choose)))', choose)
    choose_edge_ids = [id2edgeid[_c] for _c in choose] 
    edgeid2newid = dict(zip(choose_edge_ids, range(len(choose))))
    #newid2edgeid = dict(zip(range(len(choose)), choose_edge_ids))
    choose_edge_ids_set = set(choose_edge_ids)
    choose_edge_ids_adjacent_ids = []
    if not stride_sparse:  # Not crossing
        for _choose_edge_id in choose_edge_ids:
            adjacent_edge_ids = get_adjacent_edge_ids(rg, _choose_edge_id)
            #print('adjacent_edge_ids for the edge: ',_choose_edge_id, adjacent_edge_ids) #empty adjacent egde ids
            temp = set()
            for _choose_adjacent_edgeid in adjacent_edge_ids:
                if _choose_adjacent_edgeid in choose_edge_ids_set:
                    temp.add(edgeid2newid[_choose_adjacent_edgeid])
            temp.add(edgeid2newid[_choose_edge_id])
            temp = sorted(list(temp))
            #print('temp  ###### ', temp)
            if len(temp) > A:
                A = len(temp)
                #print('********** A', A)
            choose_edge_ids_adjacent_ids.append(sorted(temp))

    v = pd.Series(np.array(v).T.ravel(), name='VehId')
    v = v.values.reshape((1000, 364))

    v = v[choose]
    v = pd.Series(v.ravel(), name='VehId')#955, 364
    vehicle_stat = pd.read_excel('VED_Static_Data_ICE&HEV&PHEV&EV.xlsx',
                                usecols=['VehId', 'Vehicle Type', 'Engine Configuration & Displacement',
                                        'Generalized_Weight'])

    vehicle_stat['VehId'] = vehicle_stat['VehId'].astype(str)
    v = vehicle_stat.merge(v, how='right', on='VehId')
    v = v.fillna('NO DATA')

    vehicle_type = v['Vehicle Type'].astype('category').cat.codes
    engine_config = v['Engine Configuration & Displacement'].astype('category').cat.codes
    gen_weight = v['Generalized_Weight'].astype('category').cat.codes

    vehicle_type = vehicle_type.values.reshape((len(choose), -1, 1))
    engine_config = engine_config.values.reshape((len(choose), 364, 1))
    gen_weight = gen_weight.values.reshape((len(choose), -1, 1))

    
    speed = speed[choose]
    
    stm = d[choose]  # spatial-temporal matrix 

    arm = np.zeros((len(choose), A))  # adjacent road matrix
    arm[:] = len(choose) - 1
    for _index, _temp in enumerate(choose_edge_ids_adjacent_ids):
        for _j, _v in enumerate(_temp):
            arm[_index, _j] = _v
            
    arm = arm.astype(int)
    #print(arm)
    print("complete finish")
    print('STM  SHAPE(edges, timeslots)  13*28=  ####### ',  stm.shape)
    print('ARM  SHAPE(edges, A) #######',  arm.shape)
    print('T  SHAPE #######', len(t))
    return stm, arm, t, speed, vehicle_type, engine_config, gen_weight 


def timestamp2vec(timestamps):
    # tm_wday range [0, 6], Monday is 0
    vec = [time.strptime(str(t[:10]), '%d/%m/%Y').tm_wday for t in timestamps]  # python3
    
    
    hours = [int(t[11:13]) for t in timestamps]
    #print('hours', hours)
    hour_min = np.min(hours)
    hour_max = np.max(hours)
    ret = []
    for i in vec:
        v = [0 for _ in range(7)]
        v[i] = 1
        if i >= 5:
            v.append(0)  # weekend
        else:
            v.append(1)  # weekday
        ret.append(v)
        v2 = [0 for _ in range(hour_max - hour_min + 1)]
        v2[hours[i] - hour_min] = 1
        v+=v2
    return np.asarray(ret)

def load_meteorol(timeslots, fname):
    '''
    timeslots: the predicted timeslots
    In real-world, we dont have the meteorol data in the predicted timeslot, instead, we use the meteoral at previous timeslots, i.e., slot = predicted_slot - timeslot (you can use predicted meteorol data as well)
    '''
    #print('meteorol',timeslots)
    f = h5py.File(fname, 'r')
    Timeslot = f['date'].value
    WindSpeed = f['windspeeds'].value
    Weather = f['weathers'].value
    maxTs = f['maxTs'].value
    minTs = f['minTs'].value
    # Temperature = f['Temperature'].value
    Temperature = maxTs
    f.close()

    M = dict()  # map timeslot to index
    for i, slot in enumerate(Timeslot):
        M[slot] = i


    WS = []  # WindSpeed
    WR = []  # Weather
    #maxTE = []  # maxTs
    #minTE = []
    TMPR = []

    for slot in timeslots:
        #print(slot)
        predicted_id = M[int(slot[6:10]+slot[3:5]+slot[:2])]
        cur_id = predicted_id - 1
        WS.append(WindSpeed[cur_id])
        WR.append(Weather[cur_id])
        #maxTE.append(maxTs[cur_id])
        #minTE.append(minTs[cur_id])
        TMPR.append(Temperature[cur_id])

    WS = np.asarray(WS)
    WR = np.asarray(WR)
    #maxTE = np.asarray(maxTE)
    #minTE = np.asarray(minTE)
    TMPR = np.asarray(TMPR)
    # 0-1 scale
    if WS.max() - WS.min() != 0:
        WS = 1. * (WS - WS.min()) / (WS.max() - WS.min())
    else:
        WS[:] = 0
    #maxTE = 1. * (maxTE - maxTE.min()) / (maxTE.max() - maxTE.min())
    #minTE = 1. * (minTE - minTE.min()) / (minTE.max() - minTE.min())
    TMPR = 1. * (TMPR - TMPR.min()) / (TMPR.max() - TMPR.min())
    #print("shape: ", WS.shape, WR.shape, TMPR.shape) #maxTE.shape, minTE.shape

    # concatenate all these attributes
    merge_data = np.hstack([WR, WS[:, None], TMPR[:, None]])

    # print('meger shape:', merge_data.shape)
    return merge_data


def load_holiday(timeslots, fname):
    f = open(fname, 'r')
    holidays = f.readlines()
    holidays = set([h.strip() for h in holidays])
    H = np.zeros(len(timeslots))
    for i, slot in enumerate(timeslots):
        if slot[:8] in holidays:
            H[i] = 1
    # print(timeslots[H==1])
    return H[:, None]

def load_ved_static(timeslots, fname):
    #get cols 1,3,6
    f = pd.read_excel(fname)




from config import Config
conf = Config("config_fig.yaml")
#conf.data_path = "."
if __name__ == '__main__':
    completion_data(conf.data_path)
    '''with open("output", "wb") as f:
        cPickle.dump(completion_data(conf.data_path), f)'''

# if __name__ == "__main__":
#     out = timestamp2vec(['01/20/2018 7:00','01/20/2018 8:15','01/21/2018 9:00'])
#     print(out)
