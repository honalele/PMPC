#from . import utils
import csv
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Verdana']
#plt.rcParams["font.size"] = 30

plt.style.use('bmh')               # ggplot風
plt.style.use('seaborn-colorblind')

from tools.data_process.get_driver_profile import get_driver_profile, get_profile, compare_v, compare_v_type
from tools.data_process.get_course import get_course, visualize_course
from MPC.MPC import generateMPC
N_DRIVRES = 30
N_SCENARIOS = 1

YOUNG_IDS = [2,3,24,25]
NORMAL_IDS = [6,9,10,11,12,14,15,17,19,20,21,22,23,26,29]
ELDER_IDS = [0,1,7,8,13,27,28]
EXPERT_IDS = [4,5,16,18]


def main():

    print('Yay, here is our RSC implementation. We are going to rocknroll')
    # Load driver and scenerio id
    #print('\nwe need to select driver')

    drv_data_all = []
    for i_driver in range(N_DRIVRES):
        for i_scenario in range(N_SCENARIOS):
            flag_drv, flag_surr, drv_data, surr_data = get_driver_profile(i_driver, i_scenario)

            if flag_drv:
                #print('Driver {} have both data for scenario {}'.format(i_driver+1, i_scenario+4))
                # Load driver profile for specific scenario
                # print('Here is the driver profile for specific scenario')
                #z =[x,y,v,yaw,ay,ax] #ay:横方向加速度, ax:進行方向加速度
                #u = [steer,throttle,brake]
                drv_data_all.append(get_profile(i_driver, drv_data)) #z, u, i_driver
    
            #elif flag_drv and (not flag_surr):
                #print('Driver {} only has driver data for scenario {}'.format(i_driver, i_scenario+1))
            #else:
                #print('Driver {} have no data for scenario {}'.format(i_driver, i_scenario+1))

    # Compare the normal vs expert driver profile
    #print('We compare the normal and expert driver')
    #compare_v(drv_data_all)
    #compare_v_type(drv_data_all)

    # Genenarate MPC
    #visualize_course()
    for i_scenario in range(1): 
        num_scenario = 3
        map, course = get_course(num_scenario)
        [t, x, y, yaw, v, d, a] = generateMPC(i_driver, num_scenario, map, course, drv_data_all)
        #np.savetxt("numpy_test.csv", list_rows, delimiter =",",fmt ='% s')
        
        v_new = []
        fig = plt.figure(figsize=(5,3))
        for vi in v:
            v_new.append(abs(vi)*3.6)    
        plt.plot(v_new,label='MPC')
        plt.xlabel('Time [10hz]')
        plt.ylabel('$v$ [km/h]')
        plt.tight_layout()
        plt.savefig('pmpc_v_s3.png')

        fig = plt.figure(figsize=(5,3))
        plt.plot(d)
        plt.xlabel('Time [10hz]')
        plt.ylabel('$\delta$')
        plt.tight_layout()
        plt.savefig('pmpc_delta_s3.png')

        fig = plt.figure(figsize=(5,3))
        plt.plot(a)
        plt.xlabel('Time [10hz]')
        plt.ylabel('$a [m/{s^2}]$')
        plt.tight_layout()
        plt.savefig('pmpc_a_s3.png')





    # Visualized the cost function
    print('Here is how your cost function look liks')

    # How about the subjective risk
    print('Here is how the subjective risk look like for the expert drivers')
    print('Here is how the subjective risk analysis result')

    # Genenarate RSC
    print('Here are generations for RSC')

    # Visualized the cost function
    print('Here is how RSC cost function look liks')

    # Performance of RSC in each scenario for each individual
    print('Performance of RSC in each scenario for each individual')
    #(general profile -> scenario specific profile -> personalized scenario specific profile)


    # Visualize the map, intervention timing

    # Visualization of which driving features cause intervention, and timing

    # Prepare statistical test

    # Open personalized drivingdataset

if __name__ == '__main__':
    main()