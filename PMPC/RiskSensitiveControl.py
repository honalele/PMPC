#!/usr/bin/env python

"""
Welcome to CARLA RSC of lane change scenarios
@Author: bao.naren@g.sp.m.is.nagoya-u.ac.jp

You can also control with steering wheel Logitech G923.

To drive start by preshing the accel pedal.
Change your wheel_config.ini according to your steering wheel.

To find out the values of your steering wheel use jstest-gtk in Ubuntu.

"""
from __future__ import print_function

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import rcParams
import matplotlib.font_manager
rcParams['font.family'] = 'sans-serif'
rcParams['font.sans-serif'] = ['Verdana']
rcParams["font.size"] = 15
from scipy.optimize import minimize

import glob
import sys
from carla import ColorConverter as cc
import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import time

try:
    import pygame
    from pygame.locals import KMOD_CTRL
    from pygame.locals import KMOD_SHIFT
    from pygame.locals import K_0
    from pygame.locals import K_9
    from pygame.locals import K_BACKQUOTE
    from pygame.locals import K_BACKSPACE
    from pygame.locals import K_COMMA
    from pygame.locals import K_DOWN
    from pygame.locals import K_ESCAPE
    from pygame.locals import K_F1
    from pygame.locals import K_LEFT
    from pygame.locals import K_PERIOD
    from pygame.locals import K_RIGHT
    from pygame.locals import K_SLASH
    from pygame.locals import K_SPACE
    from pygame.locals import K_TAB
    from pygame.locals import K_UP
    from pygame.locals import K_a
    from pygame.locals import K_c
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

if sys.version_info >= (3, 0):
    from configparser import ConfigParser
else:
    from ConfigParser import RawConfigParser as ConfigParser

import carla


ego_vehicle_log = []
ego_vehicle_u_log = []
surr_vehicle_log = []
preds_ego = []
preds_surr = []

class RiskSensitiveController(object):

    def __init__(self, world, target_traj, start_in_autopilot):
        self.dt = 0.1
        self.target_traj = target_traj
        self.horizon = 5
        self._autopilot_enabled = start_in_autopilot
        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            world.player.set_autopilot(self._autopilot_enabled)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

        # initialize steering wheel
        pygame.joystick.init()

        joystick_count = pygame.joystick.get_count()
        if joystick_count > 1:
            raise ValueError("Please Connect Just One Joystick")

        self._joystick = pygame.joystick.Joystick(0)
        self._joystick.init()

        self._parser = ConfigParser()
        self._parser.read('wheel_config.ini')
        self._steer_idx = int(
            self._parser.get('Driving Force GT', 'steering_wheel'))
        self._throttle_idx = int(
            self._parser.get('Driving Force GT', 'throttle'))
        self._brake_idx = int(self._parser.get('Driving Force GT', 'brake'))
        self._reverse_idx = int(self._parser.get('Driving Force GT', 'reverse'))
        self._handbrake_idx = int(
            self._parser.get('Driving Force GT', 'handbrake'))

    def parse_events(self, world, clock):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.JOYBUTTONDOWN:
                if event.button == 0:
                    world.restart()
                elif event.button == 1:
                    world.hud.toggle_info()
                elif event.button == 2:
                    world.camera_manager.toggle_camera()
                elif event.button == 3:
                    world.next_weather()
                elif event.button == self._reverse_idx:
                    self._control.gear = 1 if self._control.reverse else -1
                elif event.button == 23:
                    world.camera_manager.next_sensor()

            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    world.restart()
                elif event.key == K_F1:
                    world.hud.toggle_info()
                elif event.key == K_h or (event.key == K_SLASH and pygame.key.get_mods() & KMOD_SHIFT):
                    world.hud.help.toggle()
                elif event.key == K_TAB:
                    world.camera_manager.toggle_camera()
                elif event.key == K_c and pygame.key.get_mods() & KMOD_SHIFT:
                    world.next_weather(reverse=True)
                elif event.key == K_c:
                    world.next_weather()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r:
                    world.camera_manager.toggle_recording()
                if isinstance(self._control, carla.VehicleControl):
                    if event.key == K_q:
                        self._control.gear = 1 if self._control.reverse else -1
                    elif event.key == K_m:
                        self._control.manual_gear_shift = not self._control.manual_gear_shift
                        self._control.gear = world.player.get_control().gear
                        world.hud.notification('%s Transmission' %
                                               ('Manual' if self._control.manual_gear_shift else 'Automatic'))
                    elif self._control.manual_gear_shift and event.key == K_COMMA:
                        self._control.gear = max(-1, self._control.gear - 1)
                    elif self._control.manual_gear_shift and event.key == K_PERIOD:
                        self._control.gear = self._control.gear + 1
                    elif event.key == K_p:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification('Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._parse_vehicle_wheel()
                self._control.reverse = self._control.gear < 0
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time())
            
            self._parse_rsc_control(world)
            world.player.apply_control(self._control)

    def _parse_rsc_control(self, world):
        [lc_x, lc_y, lc_yaw] = self.target_traj
        print('Here we can get mpc control in each time step\n')
        t = world.player.get_transform()
        v = world.player.get_velocity() #m/d
        c = world.player.get_control()
        a = world.player.get_acceleration() #m/s^2
        w = world.player.get_angular_velocity() #deg/s

        x_t = t.location.x
        y_t = t.location.y
        yaw_t = t.rotation.yaw
        v_t =  math.sqrt(v.x**2 + v.y**2 + v.z**2)
        delta_t = c.steer
        a_t = c.throttle
        b_t = c.brake

        vehicles = world.world.get_actors().filter('vehicle.*')

        surr_states = []
        if len(vehicles) > 1:
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                s = vehicle.get_transform()
                sv = vehicle.get_velocity()
                sc = vehicle.get_control()

                sx_t = s.location.x
                sy_t = s.location.y
                syaw_t = s.rotation.yaw
                sv_t = math.sqrt(sv.x**2 + sv.y**2 + sv.z**2)
                sd_t = sc.steer
                sa = sc.throttle
                sb = sc.brake
                sa_t = self._get_pedal(sa, sb)
                surr_states.append([sx_t, sy_t, syaw_t, sv_t, sd_t, sa_t])
            surr_vehicle_log.append(surr_states)

    
        
        print('Ego position x: {}, y:{}, yaw:{}\n'.format(x_t, y_t, yaw_t))
        idx, goal = self._find_nearest_index(x_t, y_t, lc_x, lc_y)
        print('Target position x: {}, y:{}\n'.format(lc_x[idx], lc_y[idx]))

        if goal:
            self._control.steer = 0
            self._control.brake = 0
            self._control.throttle = 0
            print('Finished lane change!')
        else: 
            print('Index {}'.format(idx))
            ego_state = [x_t, y_t, yaw_t, v_t]
            ego_vehicle_log.append(ego_state)

            pedal_t = self._get_pedal(a_t, b_t)
            ego_control = [delta_t, pedal_t]
            print('ego_control steer: {}, a:{}\n'.format(delta_t, pedal_t))

            lc_x, lc_y, lc_yaw = self.target_traj
            self._control.steer = 0.0
            if v_t < 100:
                self._control.throttle = 0.2
            self._control.brake = 0


     
    def _cost_function(self, ego_control, *args):
        idx = args[2]
        first = args[3]
        lc_x, lc_y, lc_yaw = self.target_traj
        if first:
            ego_state = args[0]
            surr_states = args[1]
            [x_t, y_t, psi_t, v_t] = ego_state
            [lc_x, lc_y, lc_yaw] = self.target_traj
            [pred_ego_state, pred_surr_states] = self._predict_motion(ego_state, ego_control, surr_states)
            x_t, y_t, psi_t, v_t = pred_ego_state
            cost = 0
            cost += np.sqrt((x_t - lc_x[-1])**2 + (y_t - lc_y[-1])**2)
            cost += 1000*np.sqrt((x_t - lc_x[idx])**2 + (y_t - lc_y[idx])**2)
            cost += np.sqrt((v_t*3.6 - 80)**2)
            for pred_surr_state in pred_surr_states:
                [sx_t, sy_t, syaw_t, sv_t, sd, sa] = pred_surr_state
                cost += 1000/np.sqrt((x_t - sx_t)**2 + (y_t - sy_t)**2)
        else:
            cost = 0
            predicted_ego = args[0]
            pred_surr_states = args[1]
            for n in range(len(predicted_ego)):
                print('Horizon {}'.format(n))
                [x_t, y_t, psi_t, v_t] = predicted_ego[n]
                print('Predicted x, y, v {}{}{}'.format(x_t, y_t, v_t*3.6))
                cost += 1000*np.sqrt((x_t - lc_x[-1])**2 + (y_t - lc_y[-1])**2)
                cost += np.sqrt((x_t - lc_x[idx+n])**2 + (y_t - lc_y[idx+n])**2)
                cost += np.sqrt((v_t*3.6 - 80)**2)
                pred_surr_states_n = pred_surr_states[n]
                for pred_surr_state in pred_surr_states_n:
                    [sx_t, sy_t, syaw_t, sv_t, sd, sa] = pred_surr_state
                    cost += 1000/np.sqrt((x_t - sx_t)**2 + (y_t - sy_t)**2)

        return cost


    def _predict_motion(self, state, control, surr_states):
        x_t = state[0]
        y_t = state[1]
        psi_t = state[2]
        v_t = state[3]

        delta_t = control[0]
        a_t = control[1]

        v_t_1 = v_t + a_t*self.dt - v_t/25.0
        x_dot = v_t*np.cos(psi_t) 
        y_dot = v_t*np.sin(psi_t)
        psi_dot = v_t*np.tan(delta_t*180)/2.5

        x_t += x_dot*self.dt
        y_t += y_dot*self.dt
        psi_t +=  psi_dot*self.dt

        pred_ego_state = [x_t, y_t, psi_t, v_t_1]

        pred_surr_states = []
        for surr_i in surr_states:
            [sx_t, sy_t, syaw_t, sv_t, sd_t, sa_t] = surr_i  
            
            sv_t_1 = sv_t + sa_t*self.dt - sv_t/25.0
            sx_dot = sv_t*np.cos(syaw_t) 
            sy_dot = sv_t*np.sin(syaw_t)
            psi_dot = sv_t*np.tan(sd_t*180)/2.5

            sx_t += sx_dot*self.dt
            sy_t += sy_dot*self.dt
            syaw_t +=  psi_dot*self.dt
            pred_surr_states.append([sx_t, sy_t, syaw_t, sv_t_1, sd_t, sa_t])
        return [pred_ego_state, pred_surr_states]


    def _get_pedal(self, a, b):
        if a > b:
            a_t = a
        elif b > a:
            a_t = -b
        else:
            a_t = 0
        return a_t

    def _find_nearest_index(self, x_t, y_t, lc_x, lc_y):
        idx = 0
        dmin = 100000
        goal = False
        for i in range(len(lc_x)):
            if np.sqrt((lc_x[i] - x_t) **2 + (lc_y[i] - y_t)**2) < dmin:
                dmin = np.sqrt((lc_x[i] - x_t) **2 + (lc_y[i] - y_t)**2)
                idx = i
        if np.sqrt((lc_x[-1] - x_t) **2 + (lc_y[-1] - y_t)**2) == 0:
            goal == True
        return idx, goal


    def _parse_vehicle_keys(self, keys, milliseconds):
        self._control.throttle = 1.0 if keys[K_UP] or keys[K_w] else 0.0
        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.brake = 1.0 if keys[K_DOWN] or keys[K_s] else 0.0
        self._control.hand_brake = keys[K_SPACE]

    def _parse_vehicle_wheel(self):
        numAxes = self._joystick.get_numaxes()
        jsInputs = [float(self._joystick.get_axis(i)) for i in range(numAxes)]
        # print (jsInputs)
        jsButtons = [float(self._joystick.get_button(i)) for i in
                     range(self._joystick.get_numbuttons())]

        # Custom function to map range of inputs [1, -1] to outputs [0, 1] i.e 1 from inputs means nothing is pressed
        # For the steering, it seems fine as it is
        K1 = 0.5  # 0.55
        steerCmd = K1 * math.tan(1.1 * jsInputs[self._steer_idx])

        K2 = 1.6  # 1.6
        throttleCmd = K2 + (2.05 * math.log10(
            -0.7 * jsInputs[self._throttle_idx] + 1.4) - 1.2) / 0.92
        if throttleCmd <= 0:
            throttleCmd = 0
        elif throttleCmd > 1:
            throttleCmd = 1

        brakeCmd = 0.1
        if self._brake_idx < len(jsInputs):
        	brakeCmd = 1.6 + (2.05 * math.log10(
            	-0.7 * jsInputs[self._brake_idx] + 1.4) - 1.2) / 0.92
        if brakeCmd <= 0:
            brakeCmd = 0
        elif brakeCmd > 1:
            brakeCmd = 1

        self._control.steer = -steerCmd
        self._control.brake = brakeCmd
        self._control.throttle = throttleCmd

        #toggle = jsButtons[self._reverse_idx]

        self._control.hand_brake = bool(jsButtons[self._handbrake_idx])

    def _parse_walker_keys(self, keys, milliseconds):
        self._control.speed = 0.0
        if keys[K_DOWN] or keys[K_s]:
            self._control.speed = 0.0
        if keys[K_LEFT] or keys[K_a]:
            self._control.speed = .01
            self._rotation.yaw -= 0.08 * milliseconds
        if keys[K_RIGHT] or keys[K_d]:
            self._control.speed = .01
            self._rotation.yaw += 0.08 * milliseconds
        if keys[K_UP] or keys[K_w]:
            self._control.speed = 5.556 if pygame.key.get_mods() & KMOD_SHIFT else 2.778
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)