import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Verdana']
#plt.rcParams["font.size"] = 30

plt.style.use('bmh')               # ggplot風
plt.style.use('seaborn-colorblind')
import random
random.seed(1)
type_color = plt.rcParams['axes.prop_cycle'].by_key()['color']

data_dir = '/Users/narenbao/workspace/second_journal/data/driving_data'
map_dir = '/Users/narenbao/workspace/second_journal/data/road_topology'

map_file = pd.read_csv(os.path.join(map_dir,'points_carla.csv'))

def visualize_course():
    fig = plt.figure(figsize=(15,15))
    ax = fig.add_subplot(111)
    map, course = get_course(0)
    plt.scatter(map.iloc[:,0],-map.iloc[:,1],color='black',alpha=0.83,s=0.5)
    
    for i in range(5):
        map, course = get_course(i)
        if i != 2:
            plt.scatter(course.iloc[:,0], -course.iloc[:,1],color=type_color[i],label='Scenario'+str(i+1),s=50)
            plt.scatter(course.iloc[0,0], -course.iloc[0,1],color=type_color[i],marker='*',edgecolors='black',s=400)
            plt.scatter(course.iloc[-1,0], -course.iloc[-1,1],color=type_color[i],marker='^',edgecolors='black',s=400)
            plt.legend()
    map, course = get_course(2)
    plt.scatter(course.iloc[:,0], -course.iloc[:,1],color=type_color[2],label='Scenario'+str(3),s=50)
    plt.scatter(course.iloc[0,0], -course.iloc[0,1],color='white',label='Start', marker='*',edgecolors='black',s=400)
    plt.scatter(course.iloc[-1,0], -course.iloc[-1,1],color='white',label='End', marker='^',edgecolors='black',s=400)
    plt.legend()
    plt.xlabel('Trajectory-$x$ [m]')
    plt.ylabel('Trajectory-$y$ [m]')
    ax.set_aspect('equal')

    plt.savefig('Scenarios.png')
    plt.axis()
    plt.show()

def get_course(i_scenario):
    course = 'course'
    filename = 'scenario_'+str(i_scenario+1) + '.csv'
    scenario_csv = pd.read_csv(os.path.join(data_dir, filename))

    #print(map_file.head(5))
    #print(scenario_csv.head(5))

    return map_file.iloc[:,1:3],  scenario_csv.iloc[:,1:3]
