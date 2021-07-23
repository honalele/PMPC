import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Verdana']
from scipy.signal import savgol_filter

plt.style.use('bmh')               # ggplot風
plt.style.use('seaborn-colorblind')
import random
random.seed(1)
type_color = plt.rcParams['axes.prop_cycle'].by_key()['color']

N_COUNT = 4
data_dir = '/Users/narenbao/workspace/second_journal/data/driving_data'

YOUNG_IDS = [2,3,24,25]
NORMAL_IDS = [6,9,10,11,12,14,15,17,19,20,21,22,23,26,29]
ELDER_IDS = [0,1,7,8,13,27,28]
EXPERT_IDS = [4,5,16,18]
NO_DATA_IDS = [15,16,17,18,19,20,21,22,23,24]

def get_type(i_drv):
    color = type_color[5]
    if i_drv in YOUNG_IDS:
        type = 'Young'
        color = type_color[0]
    elif i_drv in NORMAL_IDS:
        type = 'Normal'
        color = type_color[1]
    elif  i_drv in ELDER_IDS:
        type = 'Elderly'
        color = type_color[2]
    elif i_drv in EXPERT_IDS:
        type = 'Expert'
        color = type_color[3]
    else:
        type = 'None'
        color = type_color[5]

    return type,color


def compare_v(drv_data_all):
    drv_num = len(drv_data_all)
    #fig = plt.figure(figsize=plt.figaspect(0.8))
    fig = plt.figure(figsize=(20.0, 10.0))

    for i_drv in range(drv_num):
        #z =[x,y,v,yaw,ay,ax] #ay:横方向加速度, ax:進行方向加速度
        #u = [steer,throttle,brake]
        z, u, i_driver = drv_data_all[i_drv]
        v = z[2]
        steer = u[1]
        ax = fig.add_subplot(5,6,i_drv+1)
        type,color = get_type(i_drv)
        ax.plot(v,color=color)
        ax.set_ylim(0,70)
        ax.set_ylabel('$v$ [km/h]')
        ax.set_xlabel('time [10hz]')
        ax.set_title('{}, ID:{}'.format(type,i_drv+1))

    fig.tight_layout()
    plt.savefig('carla_v.png')
    plt.show()


def compare_v_type(drv_data_all):
    drv_num = len(drv_data_all)
    #fig = plt.figure(figsize=plt.figaspect(0.8))
    fig = plt.figure(figsize=(10.0, 6.0))
    V_MAX = 120
    DELTA_MAX = 0.05

    for i_drv in range(drv_num):
        #z =[x,y,v,yaw,ay,ax] #ay:横方向加速度, ax:進行方向加速度
        #u = [steer,throttle,brake]
        z, u, i_driver = drv_data_all[i_drv]
        v = z[2]
        steer = u[0]
        type,color = get_type(i_drv)
        
        if type == 'Young':
            ax = fig.add_subplot(4,2,1)
            ax.plot(v,color=color)
            #ax.plot(steer,color=color)
            ax.set_ylim(0,V_MAX)
            ax.set_ylabel('$v$ [km/h]')
            #ax.set_xlabel('time [10hz]')
            ax.set_title('{} drivers'.format(type))
            ax = fig.add_subplot(4,2,3)
            ax.plot(steer,color=color)
            #ax.plot(steer,color=color)
            ax.set_ylim(-DELTA_MAX,DELTA_MAX)
            ax.set_ylabel('$\delta$')
            #ax.set_xlabel('time [10hz]')            
        elif type == 'Normal':
            ax = fig.add_subplot(4,2,2)
            ax.plot(v,color=color)
            #ax.plot(steer,color=color)
            ax.set_ylim(0,V_MAX)
            ax.set_ylabel('$v$ [km/h]')
            #ax.set_xlabel('time [10hz]')
            ax.set_title('{} drivers'.format(type))
            ax = fig.add_subplot(4,2,4)
            ax.plot(steer,color=color)
            #ax.plot(steer,color=color)
            ax.set_ylim(-DELTA_MAX,DELTA_MAX)
            ax.set_ylabel('$\delta$')
            #ax.set_xlabel('time [10hz]')            
        elif type == 'Elderly':
            ax = fig.add_subplot(4,2,5)
            ax.plot(v,color=color)
            #ax.plot(steer,color=color)
            ax.set_ylim(0,V_MAX)
            ax.set_ylabel('$v$ [km/h]')
            #ax.set_xlabel('time [10hz]')
            ax.set_title('{} drivers'.format(type))
            ax = fig.add_subplot(4,2,7)
            ax.plot(steer,color=color)
            #ax.plot(steer,color=color)
            ax.set_ylim(-DELTA_MAX,DELTA_MAX)
            ax.set_ylabel('$\delta$')
            #ax.set_xlabel('time [10hz]')            
        elif type == 'Expert':
            ax = fig.add_subplot(4,2,6)
            ax.plot(v,color=color)
            #ax.plot(steer,color=color)
            ax.set_ylim(0,V_MAX)
            ax.set_ylabel('$v$ [km/h]')
            #ax.set_xlabel('time [10hz]')
            ax.set_title('{} drivers'.format(type))
            ax = fig.add_subplot(4,2,8)
            ax.plot(steer,color=color)
            #ax.plot(steer,color=color)
            ax.set_ylim(-DELTA_MAX,DELTA_MAX)
            ax.set_ylabel('$\delta$')
            #ax.set_xlabel('time [10hz]')            
        else:
            print('Something wrong')
    fig.tight_layout()
    plt.savefig('carla_v_steer_type_s5.png')
    plt.show()


def get_v(data):
    features_name = data.columns
    vx = data['vx'].to_numpy()
    vy = data['vy'].to_numpy()
    #vz = data['vz'].to_numpy()
    
    v = np.sqrt(vx**2 + vy**2) * 3.6
    #print(v)
    v = savgol_filter(v, 51, 3)
    return(v)


def get_profile(i_driver, drv_data):
    cnt_num = len(drv_data)
    v, ax, ay =  [], [], []
    longest_cnt = 0
    max_frame = 0
    for i in range(cnt_num):
        
        frame, _ = drv_data[i].shape
        if frame >= max_frame:
            longest_cnt = i
    drv_data_selected = drv_data[longest_cnt]
    features_name = drv_data_selected.columns
    v = get_v(drv_data_selected) #km/h
    
    x = drv_data_selected['x'] #m
    y = drv_data_selected['y'] #m
    yaw = drv_data_selected['yaw'] #degrees:rotation angle.
    throttle = drv_data_selected['throttle']#[0,1]
    steer = drv_data_selected['steer']#[-1,1]
    brake = drv_data_selected['brake']#[0,1]

    if 'ax' in features_name:
        ax = drv_data_selected['ax'] #進行方向加速度
        ay = drv_data_selected['ay'] #横方向加速度
        z =[x,y,v,yaw,ay,ax] #ay:横方向加速度, ax:進行方向加速度
    else:
        z = [x,y,v,yaw]
    u = [steer,throttle,brake]

    return z, u, i_driver



def get_driver_profile(i_driver, i_scenario):
    #print('\n\nDriver profile loaded \n')
    flag_drv_data = False
    drv_data_all = []
    
    if i_driver in NO_DATA_IDS:
        i_driver = random.randint(0, 14)

    for i_cnt in range(N_COUNT):
        filename = 'drv_p' + str(i_driver) + '_s' + str(i_scenario+1) + '_c' + str(i_cnt+1) + '.csv' 
        filename = os.path.join(data_dir, filename)

        filename2 = 'p' + str(i_driver) + '_s' + str(i_scenario+1) + '_c' + str(i_cnt+1) + '.csv' 
        filename2 = os.path.join(data_dir, filename2)

        if os.path.exists(filename):
            drv_data = pd.read_csv(filename)
            flag_drv_data = True
            drv_data_all.append(drv_data)
        elif os.path.exists(filename2):
            drv_data = pd.read_csv(filename2)
            drv_data_all.append(drv_data)
            flag_drv_data = True
        #else:
            #print(filename2)
            #print('\nDriver id {} have no data \n'.format(i_driver))
    
    flag_surr_data = False
    surr_data_all = []
    for i_cnt in range(N_COUNT):
        filename = 'sur_p' + str(i_driver) + '_s' + str(i_scenario+1) + '_c' + str(i_cnt+1) + '.csv' 
        filename = os.path.join(data_dir, filename)

        if os.path.exists(filename):
            surr_data = pd.read_csv(filename)
            surr_data_all.append(surr_data)
            flag_surr_data = True
        #else:
            #print(filename2)
            #print('\nDriver id {} have no data \n'.format(i_driver))
    
    flag_drv_data = check_data_length(flag_drv_data,drv_data_all)
    flag_surr_data = check_data_length(flag_surr_data,surr_data_all)


    return flag_drv_data, flag_surr_data, drv_data_all, surr_data_all


def check_data_length(flag, data):
    if flag == True:
        cnt_n = len(data)
        if cnt_n == 1:
            time_steps = data[0].shape[0]
            if time_steps < 100:
                flag = False
        else:
            no_data = True 
            for i in range(len(data)):
                time_steps = data[i].shape[0]
                if time_steps > 100:
                    no_data = False
            if no_data == False:
                flag = True
    return flag






