# -*- coding: utf-8 -*-
"""
Created on Wed Feb 19 10:02:55 2020

@author: hona
"""

import cutils
import numpy as np

class Controller2D(object):
    def __init__(self, waypoints):
        self.vars = cutils.CUtils()
        self._current_x = 0
        self._current_y = 0
        self._current_yaw = 0
        self._current_speed = 0
        self._desired_speed = 0
        self._current_timestamp = 0
        self._start_control_loop = False
        self._set_throttle = 0
        self._set_brake = 0
        self._set_steer = 0
        self._waypoints = waypoints
        self._conv_rad_to_steer = 180.0 / 70.0 / np.pi
        self._pi = np.pi
        self._2pi = 2.0 * np.pi
        
        
    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x = x
        self._current_y = y
        self._current_yaw = yaw
        self._current_speed = speed
        self._current_timestamp = timestamp
        self._current_frame = frame
        
        if self._current_frame:
            self._start_control_loop = True
            
        
    def update_desired_speed(self):
        min_idx = 0
        min_dist = float("inf")
        desired_speed = 0
        
        for i in range(len(self._waypoints)):
            dist = np.linalg.norm(np.array([
                self._waypoints[i][0] - self._current_x,
                self._waypoints[i][1] - self._current_y]))

            if dist < min_dist:
                min_dist = dist
                min_idx = i

        if min_idx < len(self._waypoints)-1:
            desired_speed = self._waypoints[min_idx][2]
        else:
            desired_speed = self._waypoints[-1][2]
            
        
    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints
        
    
    def get_commands(self):
        return self._set_throttle, self._set_brake, self._set_steer
        
        
    def set_throttle(self, input_throttle):
        #Clamp the throttle command to valid bounds
        throttle = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle
        
    
    def set_steer(self, input_steer_in_rad):
        #Convert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad
        
        #Clamp the steering command to valid bounds
        steer = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer
        
    
    def set_brake(self, input_brake):
        #clamp the steering command to valid bounds
        brake = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake
        
        
    def update_control(self):
        # Retrieve simulator feedback
        x = self._current_x
        y = self._current_y
        yaw = self._current_yaw
        v = self._current_speed
        self.update_desired_speed()
        v_desired = self._desired_speed
        t = self._current_timestamp
        waypoints = self._waypoints
        throttle_output = 0
        steer_output = 0
        brake_output = 0
        
        #declare usage variables here
        self.vars.create_var('v_previous', 0.0)
        
        # skip the first frame to store previous values properly
        
        if self._start_control_loop:
            
            
            # Longitudinal control        
            self.vars.v_previous = 1.0            
            throttle_output = 0.5 * self.vars.v_previous
            brake_output = 0.5 * self.vars.v_previous
            
            # Latarel control
            steer_output = 0.5
            
            self.set_throttle(throttle_output)
            self.set_brake(brake_output)
            self.set_steer(steer_output)
            
        #store old values
        self.vars.v_previous = v
            
            
        
        
        
        
        
        
        
        
        






            

        