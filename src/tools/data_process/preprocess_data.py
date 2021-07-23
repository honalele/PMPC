import os
import numpy as np
import pandas as pd
import scipy
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats
import csv
sns.set_style("darkgrid")

data_dir = '/Users/narenbao/workspace/second_journal/driving_data'
NUM_SCENARIO = 5
NUM_DRIVER = 30
CND_ID = [2,3,4]

def get_driving_data():
    #drv_data, surr_data = 'drv_data', 'surr_data'

    drv_data = []
    surr_data = []
    
    for i_scenario in range(NUM_SCENARIO):
    	cnt = 0
    	for i_driver in range(NUM_DRIVER):
    		for i_cnt in CND_ID:
    			drv_filename = 'drv_p' + str(i_driver+1) + '_s' +  str(i_scenario+1) + '_c' + str(i_cnt) + '.csv'

    			file = os.path.join(data_dir, drv_filename)
    			if os.path.exists(file):
    				drv_csv = pd.read_csv(file)
    				cnt += 1
    				#print(drv_csv.head(5))
    				#print('----------------')
    				drv_data_each = {'driver_id': i_driver+1, 
    				'scenario_i': i_scenario+1, 
    				'cnt_i': i_cnt,
    				'data': drv_csv,
    				'cnt_i': cnt}
    				drv_data.append(drv_data_each)



    	#print('We have {} driving data files for scenario #{}'.format(cnt, i_scenario+1))

    for i_scenario in range(NUM_SCENARIO):
    	cnt = 0
    	for i_driver in range(NUM_DRIVER):
    		for i_cnt in CND_ID:
    			drv_filename = 'sur_p' + str(i_driver+1) + '_s' +  str(i_scenario+1) + '_c' + str(i_cnt) + '.csv'

    			file = os.path.join(data_dir, drv_filename)
    			if os.path.exists(file):
    				sur_csv = pd.read_csv(file)
    				cnt += 1
    				sur_data_each = {'driver_id': i_driver+1, 
    				'scenario_i': i_scenario+1, 
    				'cnt_i': i_cnt,
    				'data': sur_csv,
    				'cnt_i': cnt}
    				#print(sur_csv.head(5))
    				surr_data.append(sur_data_each)
    				
    	#print('We have {} surrounding vehicle files for scenario #{}'.format(cnt, i_scenario+1))

    return drv_data, surr_data

def select_scenario(drv_data_all, surr_data_all):

	cnt = 0
	scenario_1 = []
	scenario_2 = []
	scenario_3 = []
	scenario_4 = []
	scenario_5 = []
	#print(len(surr_data_all))
	
	for i_lc in range(len(drv_data_all)):

		drv_data = drv_data_all[i_lc]['data']
		
		#surr_data = surr_data_all[i_lc]['data']
		driver_id = drv_data_all[i_lc]['driver_id']
		scenario_i = drv_data_all[i_lc]['scenario_i']
		
		if not drv_data.empty:
			cnt += 1
			if scenario_i == 1:
				scenario_1.append(drv_data)
			elif scenario_i == 2:
				scenario_2.append(drv_data)
			elif scenario_i == 3:
				scenario_3.append(drv_data)
			elif scenario_i == 4:
				scenario_4.append(drv_data)
			elif scenario_i == 5:
				scenario_5.append(drv_data)
			else:
				print('Something go wrong.')
	
	scenario_drv_data = [scenario_1, scenario_2, scenario_3, scenario_4, scenario_5]

	scenario_drv_selected = []
	for i_scenario in range(len(scenario_drv_data)):
		cnt = 0
		i_scenario_selected = []
		for i_lc in range(len(scenario_drv_data[i_scenario])):
			lc_drv_data = scenario_drv_data[i_scenario][i_lc]
			if len(lc_drv_data) > 300:
				cnt += 1
				i_scenario_selected.append(lc_drv_data)
			elif len(lc_drv_data) > 250 and  i_scenario == 1:
				cnt += 1
				i_scenario_selected.append(lc_drv_data)
		#print('Scenario {} has {} selected driving data'.format(i_scenario, cnt))	
		scenario_drv_selected.append(i_scenario_selected)

	for i_lc in range(len(surr_data_all)):

		sur_data = surr_data_all[i_lc]['data']
		
		#surr_data = surr_data_all[i_lc]['data']
		driver_id = surr_data_all[i_lc]['driver_id']
		scenario_i = surr_data_all[i_lc]['scenario_i']
		
		if not sur_data.empty:
			cnt += 1
			if scenario_i == 1:
				scenario_1.append(sur_data)
			elif scenario_i == 2:
				scenario_2.append(sur_data)
			elif scenario_i == 3:
				scenario_3.append(sur_data)
			elif scenario_i == 4:
				scenario_4.append(sur_data)
			elif scenario_i == 5:
				scenario_5.append(sur_data)
			else:
				print('Something go wrong.')
	
	scenario_surr_data = [scenario_1, scenario_2, scenario_3, scenario_4, scenario_5]

	scenario_surr_selected = []
	for i_scenario in range(len(scenario_surr_data)):
		cnt = 0
		i_scenario_selected = []
		for i_lc in range(len(scenario_surr_data[i_scenario])):
			lc_sur_data = scenario_surr_data[i_scenario][i_lc]
			if len(lc_sur_data) > 300:
				cnt += 1
				i_scenario_selected.append(lc_sur_data)
			elif len(lc_sur_data) > 250 and  i_scenario == 1:
				cnt += 1
				i_scenario_selected.append(lc_sur_data)
		#print('Scenario {} has {} selected surrrounding vehicle data'.format(i_scenario, cnt))	
		scenario_surr_selected.append(i_scenario_selected)

	return scenario_drv_selected, scenario_surr_selected
    



def select_scenario_drv(drv_data_all, surr_data_all):
    cnt = 0
    Scenario_1 = []
    Scenario_2 = []
    Scenario_3 = []
    Scenario_4 = []
    Scenario_5 = []
    scenario_data = []

    for i_lc in range(len(drv_data_all)):

    	drv_data = drv_data_all[i_lc]['data']
    	driver_id = drv_data_all[i_lc]['driver_id']
    	scenario_i = drv_data_all[i_lc]['scenario_i']
    	cnt_i = drv_data_all[i_lc]['cnt_i']
    	if not drv_data.empty:
    		cnt += 1
    		
    		#print('Driver {} of Scenario #{} in Cnt {}.'.format(driver_id, scenario_i, cnt_i))
    		if scenario_i == 1:
    			Scenario_1.append(drv_data)
    		elif scenario_i == 2:
    			Scenario_2.append(drv_data)
    		elif scenario_i == 3:
    			Scenario_3.append(drv_data)
    		elif scenario_i == 4:
    			Scenario_4.append(drv_data)
    		elif scenario_i == 5:
    			Scenario_5.append(drv_data)
    		else:
    			print('Something go wrong.')
    scenario_data = [Scenario_1, Scenario_2, Scenario_3, Scenario_4, Scenario_5]

    scenario_selected = []
    for i_scenario in range(len(scenario_data)):
    	cnt = 0
    	i_scenario_selected = []
    	for i_lc in range(len(scenario_data[i_scenario])):
    		lc_drv_data = scenario_data[i_scenario][i_lc]
    		#print(len(lc_drv_data))
    		if len(lc_drv_data) > 300:
    			cnt += 1
    			i_scenario_selected.append(lc_drv_data)
    		elif len(lc_drv_data) > 250 and  i_scenario == 1:
    			cnt += 1
    			i_scenario_selected.append(lc_drv_data)
    	#print('Scenario {} has {} lane changes'.format(i_scenario+1, cnt))
    	scenario_selected.append(i_scenario_selected)
    return scenario_selected


def print_scenario_data(scenario_selected):
	fig = plt.figure(figsize=(20,10))
	plt.rcParams["font.size"] = 18

	
	#for i_scenario in range(len(scenario_selected)):
	for i_scenario in range(5):

		scenario_data = scenario_selected[i_scenario]
		#print(drv_data[0].columns)
		for i_lc in range(len(scenario_data)):
			drv_data = scenario_data[i_lc]

			ax = fig.add_subplot(5, 1, i_scenario+1)
			ax.set_xlabel("x [m]")
			ax.set_ylabel("y [m]")
			ax.plot(drv_data.x, drv_data.y)
			
			xmin = min(drv_data.x)
			ymin = min(drv_data.y)
			xmax = max(drv_data.x)
			ymax = max(drv_data.y)

			ax.set_ylim([ymin-5,ymax+5])
			ax.grid(True)
			ax.set_xlim([xmin-5, xmax+5])
			ax.set_title("Trajectory of Scenario # {}".format(i_scenario+1))

			
		ax.plot(drv_data.x[0], drv_data.y[0],label='Start position', marker="*", markersize=20)
		#ax.legend()
		plt.rcParams["font.size"] = 18
	plt.tight_layout()
	plt.savefig('Trajectorty.png')
	plt.show()


def print_scenario_data2(scenario_selected):
	fig = plt.figure(figsize=(10,15))
	plt.rcParams["font.size"] = 20

	scenario_data = scenario_selected[0]
	for i_lc in range(len(scenario_data)):
		drv_data = scenario_data[i_lc]

		ax = fig.add_subplot(5, 1, 1)
		#ax.set_xlabel("x [m]")
		ax.set_ylabel("y [m]")
		ax.plot(drv_data.x, drv_data.y)
			
		xmin = -350
		ymin = 425
		xmax = -100
		ymax = 445

		ax.set_ylim([ymin,ymax])
		ax.grid(True)
		ax.set_xlim([xmin, xmax])
		ax.set_title("Trajectory of Scenario # {}".format(1))
		ax.plot(drv_data.x[0], drv_data.y[0], marker="*", markersize=10, markerfacecolor='black')

	scenario_data = scenario_selected[1]
	for i_lc in range(len(scenario_data)):
		drv_data = scenario_data[i_lc]

		ax = fig.add_subplot(5, 1, 2)
		#ax.set_xlabel("x [m]")
		ax.set_ylabel("y [m]")
		ax.plot(drv_data.x, drv_data.y)
			
		xmin = 80
		ymin = 220
		xmax = 140
		ymax = 250

		ax.set_ylim([ymin,ymax])
		ax.grid(True)
		ax.set_xlim([xmin, xmax])
		ax.set_title("Trajectory of Scenario # {}".format(2))
		ax.plot(drv_data.x[0], drv_data.y[0], marker="*", markersize=10, markerfacecolor='black')

	for i_scenario in range(3):

		scenario_data = scenario_selected[i_scenario+2]
		#print(drv_data[0].columns)
		for i_lc in range(len(scenario_data)):
			drv_data = scenario_data[i_lc]

			ax = fig.add_subplot(5, 1, i_scenario+3)
			#ax.set_xlabel("x [m]")
			ax.set_ylabel("y [m]")
			ax.plot(drv_data.x, drv_data.y)
			
			xmin = min(drv_data.x)
			ymin = min(drv_data.y)
			xmax = max(drv_data.x)
			ymax = max(drv_data.y)

			ax.set_ylim([ymin-5,ymax+5])
			ax.grid(True)
			ax.set_xlim([xmin-5, xmax+5])
			ax.set_title("Trajectory of Scenario # {}".format(i_scenario+3))

		ax.plot(drv_data.x[0], drv_data.y[0], marker="*", markersize=10, markerfacecolor='black')
	plt.xlabel('x [m]')
	plt.tight_layout()
	plt.rcParams["font.size"] = 20

	plt.savefig('show_trajectorty1.png')
	plt.show()



def print_scenario_data3(scenario_selected):
	fig = plt.figure(figsize=(10,4))
	plt.rcParams["font.size"] = 18

	scenario_data = scenario_selected[0]
	for i_lc in range(len(scenario_data)):
		drv_data = scenario_data[i_lc]

		#ax.set_xlabel("x [m]")
		plt.ylabel("y [m]", fontsize=18)
		plt.plot(drv_data.x, drv_data.y)
			
		xmin = -330
		ymin = 425
		xmax = -100
		ymax = 438

		plt.ylim([ymin,ymax])
		plt.grid(True)
		plt.xlim([xmin, xmax])
		plt.title("Trajectory of Scenario # {}".format(1), fontsize=18)
		plt.plot(drv_data.x[0], drv_data.y[0], marker="*", markersize=10, markerfacecolor='black')

	plt.xlabel('x [m]', fontsize=18)
	plt.tight_layout()
	plt.rcParams["font.size"] = 18

	plt.savefig('Trajectorty_s1.png')
	plt.show()



def test_individual_difference(scenario_selected):
	p_all = []
	for i_scenario in range(5):
		#plt.figure(figsize=(10,4))

		scenario_data = scenario_selected[i_scenario]
		
		#print(drv_data[0].columns)
		v = []
		a = []
		for i_driver in range(30):
			drv_data = scenario_data[i_driver]
			vx = drv_data.vx
			vy = drv_data.vx
			vi = []
			ai = []

			ax = drv_data.ax
			ay = drv_data.ay
			for i in range(len(vx)):
				vi.append(np.sqrt(vx[i]**2 + vy[i]**2))
				ai.append(np.sqrt(ax[i]**2 + ay[i]**2))

			v.append(vi)
			a.append(ai)

			#plt.plot(vi)
		#pltplt.show()
		p_each_scenaro = []
		for i_driver in range(30):
			if i_driver == 29:
				v1 =v[i_driver]
				v2 =v[i_driver-1]

			else:
				v1 =v[i_driver]
				v2 =v[i_driver+1]

			n1 = len(v1)
			n2 = len(v2)

			t,p = stats.ttest_ind(v1,v2, equal_var=False)
			df = n1 + n2 - 2
			print('t (%g) = %g, p =%g'%(df,t,p))
			p_each_scenaro.append("%.3f" % p)

		for i_driver in range(30):
			if i_driver == 29:
				v1 =a[i_driver]
				v2 =a[i_driver-1]

			else:
				v1 =a[i_driver]
				v2 =a[i_driver+1]

			n1 = len(v1)
			n2 = len(v2)

			t,p = stats.ttest_ind(v1,v2, equal_var=False)
			df = n1 + n2 - 2
			print('t (%g) = %g, p =%g'%(df,t,p))
			p_each_scenaro.append("%.3f" % p)
			

		
		for i_driver in range(30):
			"""
			steer1 = scenario_data[i_driver].steer
			#print(steer1)
			steer2 = scenario_data[i_driver+1].steer
			n1 = len(steer1)
			n2 = len(steer2)
			t,p = stats.ttest_ind(steer1,steer2, equal_var=False)
			df = n1 + n2 - 2
			print('t (%g) = %g, p =%g'%(df,t,p))
			p_each_scenaro.append(p)

			"""
			if i_driver == 29:
				steer1 = scenario_data[i_driver].steer
				steer2 = scenario_data[i_driver-1].steer
			else:
				steer1 = scenario_data[i_driver].steer
				steer2 = scenario_data[i_driver+1].steer

			n1 = len(steer1)
			n2 = len(steer2)
			t,p = stats.ttest_ind(steer1,steer2, equal_var=False)
			df = n1 + n2 - 2
			print('t (%g) = %g, p =%g'%(df,t,p))
			p_each_scenaro.append("%.3f" % p)

		print(p_each_scenaro)
		filename = 'csv/new_i_scenario_' + str(i_scenario+1) + '.csv'
		f = open(filename, 'w')

		writer = csv.writer(f)
		writer.writerows([p_each_scenaro])
		f.close()
		





		




