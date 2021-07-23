import os
from preprocess_data import NUM_DRIVER, NUM_SCENARIO
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import csv

import matplotlib.pyplot as plt
import autograd.numpy as np
import json

data_dir = '/Users/narenbao/workspace/second_journal/driving_data' 

def load_scenario_course(i_scenario):
    scenario_csv = pd.read_csv(os.path.join(data_dir, ('scenario_'+str(i_scenario)+'.csv')))
    course_x = scenario_csv.x
    course_y = scenario_csv.y
    return [course_x, course_y]

def check_goal(course, current_position):

    goal =[course[0].tail(1).iloc[-1], course[1].tail(1).iloc[-1]]
    #print(goal)

    distance = np.sqrt((current_position[0]-goal[0])**2+(current_position[1]-goal[1])**2)
    if distance <= 15:
        is_goal = True
    else:
        is_goal = False
    return goal, is_goal

def get_distance(point1, point2):
    distance = np.sqrt((point1[0]-point2[0])**2+(point1[1]-point2[1])**2)
    return distance


def get_nearest_index(course, current_position):
    idx = len(course[0]) #goal index
    nearset_distance = 100000
    if not check_goal(course, current_position)[1]:
        for i in range(len(course[0])):
            distance = get_distance(current_position, [course[0][i],course[1][i]])
            if distance < nearset_distance:
                nearset_distance = distance
                idx = i
    return idx
            
def get_v_a(x, y):
    value = np.sqrt(x**2 + y**2)
    return value



q_t = [0.5,1,1,1,1,1]


def get_cost_function(scenario_drv_selected, scenario_sur_data):  

    for i_scenario in range(1):
        drv_data = scenario_drv_selected[i_scenario]
        sur_data = scenario_sur_data[i_scenario]
        #print(drv_data)
        #print('Scenario {} has {} driving data.'.format(i_scenario+1, len(drv_data)))
        #print('Scenario {} has {} surrounding vehicle data.'.format(i_scenario+1, len(sur_data)))
        J_all_drivers =[]
        if i_scenario == 0: #scenario 1 
            course = load_scenario_course(i_scenario+1)

            #plt.plot(course[0],course[1])
            #plt.show()
            surr_detail = sur_data[0]
            #print(surr_detail)
            for i_driver in range(NUM_DRIVER):
                
                J_save = []
                current_position = []
                n_time_steps, _ = drv_data[i_driver].shape
                # 'x', 'y', 'z', 'yaw', 'vx', 'vy', 'vz', 'ax', 'ay', 'az', 'throttle', 'steer', 'brake'

                #print(drv_data[i_driver].columns)
                for i_data in range(n_time_steps):
                    J = 0.0 # initial cost function
                    drv_data_t = drv_data[i_driver].iloc[i_data,:]
                    current_position=[drv_data_t.x, drv_data_t.y]
                    v_t = 3.6*get_v_a(drv_data_t.vx, drv_data_t.vy)
                    a_t = get_v_a(drv_data_t.ax, drv_data_t.ay)
                    throttle_t = drv_data_t.throttle
                    steer_t = drv_data_t.steer

                    #print(current_position)
                    #print(v_t)
                    goal, is_goal = check_goal(course, current_position)
                    if not is_goal:
                        distance = get_distance(current_position, goal)
                        J += distance*q_t[0]
                        nearest_idx = get_nearest_index(course, current_position)
                        for i_t in range(5):
                            if nearest_idx + 5 < len(course[0]):
                                predicted_position = [course[0][nearest_idx+i_t], course[1][nearest_idx+i_t]]
                                J += get_distance(predicted_position, current_position) * q_t[1]
                                J += throttle_t **2 * q_t[2]
                                J += steer_t **2 * q_t[3]

                                if i_data + i_t < len(surr_detail):
                                    current_position_sur = [surr_detail.x[i_data+i_t], surr_detail.y[i_data+i_t]]
                                J += q_t[4]/(get_distance(predicted_position, current_position_sur)+0.01)
                                #print(current_position_sur)
                                #J += 
                    J_save.append(J)
                    cost_file_name = 'csv/cost_value_' + str(i_driver+1) + '.csv'
                    #with open(cost_file_name, 'w') as myfile:
                    #    wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,delimiter='n')
                    #    wr.writerow(J_save)
                J_all_drivers.append((J_save))
            return J_all_drivers
                
            
def visualize_cost_function(scenario_drv_selected, J_all_drivers):
     for i_scenario in range(1):
        drv_data = scenario_drv_selected[i_scenario]
        if i_scenario == 0: #scenario 1 
            course = load_scenario_course(i_scenario+1)

            #plt.plot(course[0],course[1])
            #plt.show()
            

            for i_driver in range(NUM_DRIVER):
                J_each_driver = J_all_drivers[i_driver]

                n_time_steps, _ = drv_data[i_driver].shape
                xmin = min(drv_data[i_driver].x)
                xmax = max(drv_data[i_driver].x)
                ymin = min(drv_data[i_driver].y)
                ymax = max(drv_data[i_driver].y)
                xstep = (xmax-xmin)/n_time_steps 
                ystep = (ymax-ymin)/n_time_steps 
                x,y = np.meshgrid(np.arange(xmin, xmax+xstep, xstep), np.arange(ymin, ymax+ystep,  ystep))
                #print(len(x))
                #print(len(J_each_driver))
                if len(J_each_driver) < len(x):
                    for i in range(len(x)-len(J_each_driver)):
                        J_each_driver.append(J_each_driver[-1])
                #print(len(x))
                #print(len(J_each_driver))


                fig = plt.figure(figsize=(8, 5))
                ax = plt.axes(projection='3d', elev=50, azim=-50)
                ax.plot_surface(x, y, np.array(J_each_driver), norm=LogNorm(), rstride=1, cstride=1,
                 edgecolor='none', alpha=.8, cmap=plt.cm.jet)
                #ax.plot(*minima_, f(*minima_), 'r*', markersize=10)
                ax.set_xlabel('$x$')
                ax.set_ylabel('$y$')
                ax.set_zlabel('$z$')
                ax.set_xlim((xmin, xmax))
                ax.set_ylim((ymin, ymax))
                plt.show()

linestyles = [ "-", "--",  ":", '-.', 'solid']

def visualize_cost_temp(J_all_drivers):
    plt.figure(figsize=(10,5))
    plt.rcParams["font.size"] = 25
    for i_j in range(len(J_all_drivers)):
        print('working on {}'.format(i_j+1))
        J_i = J_all_drivers[i_j]
        #plt.plot(J_i,  linestyle=linestyles[i_j], color=[0.1*i_j, 0.1*i_j, 0.1*i_j],label=str('Driver_{}'.format(i_j+1)))
        plt.plot(J_i,label=str('Driver_{}'.format(i_j+1)))

    plt.grid(True)
    plt.ylabel('Cost $J$',fontsize=18)
    plt.xlabel('Time steps [10hz]',fontsize=18)
    plt.xticks(fontsize=18)
    plt.yticks(fontsize=18)
    #plt.legend(fontsize=18)
    plt.tight_layout()
    plt.savefig('cost_function_s1_30.png')
    plt.show()


def visualize_cost_summary():
    #plt.figure(figsize=(10,5))
    #plt.rcParams["font.size"] = 25
    #print('Prepare for visualization')
    #print(J_all_drivers)
    cost_file_name = 'csv/cost_value.csv'
    cost_pd = pd.read_csv(cost_file_name)
    """
    with open(cost_file_name, 'w') as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL,delimiter='\n')
        wr.writerow(J_all_drivers)
        """
    
    #DRIVER_NUM, _ = cost_pd.shape()
    time_step = 200000000
    for index, row in cost_pd.iterrows():
        print('Driver {} cost data time steps : {}'.format(index+1, len(row[0])))
        if len(row[0]) < time_step:
            time_step = len(row[0])

    for index, row in cost_pd.iterrows():
        data = row[0].split(',')[0:time_step]
        plt.plot(data)
    plt.show()




    #sns.lineplot(data=cost_pd)



    






