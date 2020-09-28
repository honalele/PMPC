
from __future__ import print_function


import glob
import os
import sys
import seaborn as sns
sys.path.append("../CubicSpline/")

try:
    import cubic_spline_planner
except:
    raise RuntimeError('cannot cubic_spline_planner')

# ==============================================================================
# -- find carla module ---------------------------------------------------------
# ==============================================================================


try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass


# ==============================================================================
# -- imports -------------------------------------------------------------------
# ==============================================================================


import carla

from carla import ColorConverter as cc

import argparse
import collections
import datetime
import logging
import math
import random
import re
import weakref
import pandas as pd
import cvxpy
import csv
from pylab import rcParams


import matplotlib.pyplot as plt
from scipy.interpolate import splprep, splev

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
    from pygame.locals import K_g
    from pygame.locals import K_d
    from pygame.locals import K_h
    from pygame.locals import K_m
    from pygame.locals import K_n
    from pygame.locals import K_p
    from pygame.locals import K_q
    from pygame.locals import K_r
    from pygame.locals import K_s
    from pygame.locals import K_w
    from pygame.locals import K_l
    from pygame.locals import K_i
    from pygame.locals import K_z
    from pygame.locals import K_x
    from pygame.locals import K_MINUS
    from pygame.locals import K_EQUALS
except ImportError:
    raise RuntimeError('cannot import pygame, make sure pygame package is installed')

try:
    import numpy as np
except ImportError:
    raise RuntimeError('cannot import numpy, make sure numpy package is installed')

# ==============================================================================
# -- Pesonalized Data-driven Control ---------------------------------------
# ========================================-======================================

markers =[',', 'o', 'v', '^', '<', '>', '1', '2', '3','4', '8', 's', 'p', '*', 'h', 'H', '+', 'x']
vvv= 0
ddd = []
show_animation = True
class State:
    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None

d = []
class MPCcontroller(object):
    """Class that handles keyboard input."""
    def __init__(self, world, target_speed, initial_state):
        self._autopilot_enabled = False
        self._targe_speed = target_speed
        self.initial_state = initial_state

        self.NX = 4
        self.NU = 2
        self.T = 5

        self.R = np.diag([0.01, 0.01]) # input cost function
        self.Rd = np.diag([0.01, 1]) # input different cost matrix
        self.Q = np.diag([1.0, 1.0, 0.5, 0.5]) #stage cost function
        self.Qf = self.Q #stage final matrix

        # iterative parameter
        self.MAX_ITER = 3 
        self.DU_TH = 1
        self.DT = 0.2 #s
        self.WB = 2.5 #m
        self.N_IND_SEARCH = 200 # Search index number
        self.dl = 0.2 # course tick
        self.MAX_TIME = 5

        # CONSTRAINTS
        self.MAX_STEER = 1.0
        self.MAX_DSTEER = 0.1
        self.MAX_SPEED = 100  #km/h
        self.MIN_SPEED = 0 # km/h
        self.MAX_ACCEL = 0.9

        if isinstance(world.player, carla.Vehicle):
            self._control = carla.VehicleControl()
            self._lights = carla.VehicleLightState.NONE
            world.player.set_autopilot(self._autopilot_enabled)
            world.player.set_light_state(self._lights)
        elif isinstance(world.player, carla.Walker):
            self._control = carla.WalkerControl()
            self._autopilot_enabled = False
            self._rotation = world.player.get_transform().rotation
        else:
            raise NotImplementedError("Actor type not supported")
        self._steer_cache = 0.0
        world.hud.notification("Press 'H' or '?' for help.", seconds=4.0)

    def control(self, client, world, clock, trajectory, goal, mpc_state):

        if isinstance(self._control, carla.VehicleControl):
            current_lights = self._lights
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return True
            elif event.type == pygame.KEYUP:
                if self._is_quit_shortcut(event.key):
                    return True
                elif event.key == K_BACKSPACE:
                    if self._autopilot_enabled:
                        world.player.set_autopilot(False)
                        world.restart()
                        world.player.set_autopilot(True)
                    else:
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
                elif event.key == K_g:
                    world.toggle_radar()
                elif event.key == K_BACKQUOTE:
                    world.camera_manager.next_sensor()
                elif event.key == K_n:
                    world.camera_manager.next_sensor()
                elif event.key > K_0 and event.key <= K_9:
                    world.camera_manager.set_sensor(event.key - 1 - K_0)
                elif event.key == K_r and not (pygame.key.get_mods() & KMOD_CTRL):
                    world.camera_manager.toggle_recording()
                elif event.key == K_r and (pygame.key.get_mods() & KMOD_CTRL):
                    if (world.recording_enabled):
                        client.stop_recorder()
                        world.recording_enabled = False
                        world.hud.notification("Recorder is OFF")
                    else:
                        client.start_recorder("manual_recording.rec")
                        world.recording_enabled = True
                        world.hud.notification("Recorder is ON")
                elif event.key == K_p and (pygame.key.get_mods() & KMOD_CTRL):
                    # stop recorder
                    client.stop_recorder()
                    world.recording_enabled = False
                    # work around to fix camera at start of replaying
                    current_index = world.camera_manager.index
                    world.destroy_sensors()
                    # disable autopilot
                    self._autopilot_enabled = False
                    world.player.set_autopilot(self._autopilot_enabled)
                    world.hud.notification("Replaying file 'manual_recording.rec'")
                    # replayer
                    client.replay_file("manual_recording.rec", world.recording_start, 0, 0)
                    world.camera_manager.set_sensor(current_index)
                elif event.key == K_MINUS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start -= 10
                    else:
                        world.recording_start -= 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
                elif event.key == K_EQUALS and (pygame.key.get_mods() & KMOD_CTRL):
                    if pygame.key.get_mods() & KMOD_SHIFT:
                        world.recording_start += 10
                    else:
                        world.recording_start += 1
                    world.hud.notification("Recording start time is %d" % (world.recording_start))
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
                    elif event.key == K_p and not pygame.key.get_mods() & KMOD_CTRL:
                        self._autopilot_enabled = not self._autopilot_enabled
                        world.player.set_autopilot(self._autopilot_enabled)
                        world.hud.notification(
                            'Autopilot %s' % ('On' if self._autopilot_enabled else 'Off'))

                    elif event.key == K_l and pygame.key.get_mods() & KMOD_CTRL:
                        current_lights ^= carla.VehicleLightState.Special1
                    elif event.key == K_l and pygame.key.get_mods() & KMOD_SHIFT:
                        current_lights ^= carla.VehicleLightState.HighBeam
                    elif event.key == K_l:
                        # Use 'L' key to switch between lights:
                        # closed -> position -> low beam -> fog
                        if not self._lights & carla.VehicleLightState.Position:
                            world.hud.notification("Position lights")
                            current_lights |= carla.VehicleLightState.Position
                        else:
                            world.hud.notification("Low beam lights")
                            current_lights |= carla.VehicleLightState.LowBeam
                        if self._lights & carla.VehicleLightState.LowBeam:
                            world.hud.notification("Fog lights")
                            current_lights |= carla.VehicleLightState.Fog
                        if self._lights & carla.VehicleLightState.Fog:
                            world.hud.notification("Lights off")
                            current_lights ^= carla.VehicleLightState.Position
                            current_lights ^= carla.VehicleLightState.LowBeam
                            current_lights ^= carla.VehicleLightState.Fog
                    elif event.key == K_i:
                        current_lights ^= carla.VehicleLightState.Interior
                    elif event.key == K_z:
                        current_lights ^= carla.VehicleLightState.LeftBlinker
                    elif event.key == K_x:
                        current_lights ^= carla.VehicleLightState.RightBlinker

        if not self._autopilot_enabled:
            if isinstance(self._control, carla.VehicleControl):
                self._parse_vehicle_keys(pygame.key.get_pressed(), clock.get_time())
                self._control.reverse = self._control.gear < 0
                # Set automatic control-related vehicle lights
                if self._control.brake:
                    current_lights |= carla.VehicleLightState.Brake
                else: # Remove the Brake flag
                    current_lights &= ~carla.VehicleLightState.Brake
                if self._control.reverse:
                    current_lights |= carla.VehicleLightState.Reverse
                else: # Remove the Reverse flag
                    current_lights &= ~carla.VehicleLightState.Reverse
                if current_lights != self._lights: # Change the light state only if necessary
                    self._lights = current_lights
                    world.player.set_light_state(carla.VehicleLightState(self._lights))
            elif isinstance(self._control, carla.WalkerControl):
                self._parse_walker_keys(pygame.key.get_pressed(), clock.get_time(), world)

            #control(self, client, world, clock, trajectory, goal, mpc_state):
        
            t = world.player.get_transform()
            v = world.player.get_velocity()
            c = world.player.get_control()
            surr_vel_num = 0
            surr_vel_dis = []
            surr_vel_v = []

            vehicles = world.world.get_actors().filter('vehicle.*')
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(surr.get_location()), surr) for surr in vehicles if surr.id != world.player.id]
            for d, vehicle in sorted(vehicles, key=lambda vehicles: vehicles[0]):
                if d > 50.0:
                    break
                else:
                    surr_vel_num += 1
                    surr_v = vehicle.get_velocity()
                    surr_vel_dis.append(vehicle.get_location())
                    surr_vel_v.append(3.6 * math.sqrt(surr_v.x**2 + surr_v.y**2 + surr_v.z**2))

            [cx, cy, cyaw, ck, sp, dl, time_step, state]= mpc_state
            # initial yaw compensation

            state.x = t.location.x
            state.y = t.location.y
            state.v = math.sqrt(v.x**2 + v.y**2 + v.z**2)
            state.yaw = t.rotation.yaw

            print(state.yaw)
            if state.yaw - cyaw[0] >= math.pi:
                state.yaw -= math.pi * 2.0
            elif state.yaw - cyaw[0] <= -math.pi:
                state.yaw += math.pi * 2.0

            c_a = c.throttle #0.0, 1.0
            c_delta = c.steer #-1.0, +1.0
            c_b = c.brake #0.0, 1.0
            if np.abs(c_a) >= np.abs(c_b):
                a = c_a 
            else:
                a = c_b

            time = 0
            xx = [state.x]
            yy = [state.y]
            yyaw = [state.yaw]
            vv = [state.v]
            dd = [0]
            aa = [0]
            target_ind, _ = self._calc_nearest_index(state, cx, cy, cyaw, 0)

            odelta, oa = None, None
            cyaw = self._smooth_yaw(cyaw)

            while self.MAX_TIME >= time:
                xref, target_ind, dref = self._calc_ref_trajectory(
                    state, cx, cy, cyaw, ck, sp, dl, target_ind)
                x0 = [state.x, state.y, state.v, state.yaw] 
                oa, odelta, ox, oy, oyaw, ov = self._iterative_linear_mpc_control(
                    xref, x0, dref, oa, odelta)

                if odelta is not None:
                    di, ai = odelta[0], oa[0]
                    state = self._update_state(state, ai, di)
                time = time + self.DT
                xx.append(state.x)
                yy.append(state.y)
                yyaw.append(state.yaw)
                vv.append(state.v)
                dd.append(di/180)
                aa.append(ai)

                if ai > 0:
                    self._control.throttle = np.abs(ai)
                    self._control.brake = 0
                else:
                    self._control.throttle =0 
                    self._control.brake = np.abs(ai)
                """
                if np.abs(di*np.pi/180) > 0.2:
                    self._control.steer = -di
                else:
                    self._control.steer = di*np.pi/180
                #if time > 120 and time <140:
                #    self._control.steer = -0.2 
                """
                self._control.steer = -di/180


                world.player.apply_control(self._control)
                time_step = time_step + time

                if self._check_goal(state, goal, target_ind, len(cx)):
                    self._control.throttle =0 
                    self._control.brake = 1.0

                    print("Goal")
                    break

                if show_animation:  # pragma: no cover
                    selected_x = trajectory['x']
                    selected_y = trajectory['y']
                    plt.rcParams['figure.figsize'] = 4,7
                    plt.rcParams['font.size'] = 14
                    plt.style.use('seaborn-dark')

                    plt.cla()
                    # for stopping simulation with the esc key.
                    plt.gcf().canvas.mpl_connect('key_release_event',
                        lambda event: [exit(0) if event.key == 'escape' else None])
                    if ox is not None:
                        plt.plot(oy, ox, "xr", label="pred_traj.")
                    #plt.plot(selected_x, selected_y, color='gray', label="Lane change trajectory", linewidth=2)
                    plt.plot(cy, cx, color='gray', label="plan_traj.", linewidth=5)
                    #plt.plot(xx, yy, "ob", label="trajectory")
                    plt.plot(xref[1, :], xref[0, :], "ob", label="ref_traj.")
                    plt.plot(cy[target_ind], cx[target_ind], "xg", label="target")
                    #plot_car(state.y, state.x, state.yaw, steer=di)
                    plt.plot(state.y, state.x,  "*r", markersize=20)
                    plt.text(state.y+2, state.x, 'ego')
                    plt.text(state.y+7, state.x, str(state.v*3.6)[:5]+'[km/h]')
                    for j in range(surr_vel_num):
                        surr_vel = surr_vel_dis[j]
                        surr_num = 'surr_{}'.format(j+1)
                        plt.plot(surr_vel.y, surr_vel.x, marker=markers[j], color='k', label=surr_num, markersize=5)
                        plt.text(surr_vel.y+5,surr_vel.x, str(surr_vel_v[j]*3.6)[:5])

                    plt.xlim(-0, 35)
                    plt.ylim(-400, 15)
                    plt.grid(True)
                    plt.legend(loc="upper right")
                    plt.title("Time[s]:" + str(round(time_step, 2))
                        + ", speed[km/h]:" + str(round(state.v * 3.6, 2)))
                    plt.pause(0.0001)

            return xx, yy, yyaw, vv, dd, aa, time_step, state
            

    def _check_goal(self, state, goal, tind, nind):
        # check goal
        dx = state.x - goal[0]
        dy = state.y - goal[1]
        d = math.hypot(dx, dy)
        isgoal = (d <= 1)
        if abs(tind - nind) >= 5:
            isgoal = False
        isstop = (abs(state.v) <= 2)
        if isgoal and isstop:
            return True
        return False

    def _pi_2_pi(self, angle):
        while(angle > math.pi):
            angle = angle - 2.0 * math.pi
        while(angle < -math.pi):
            angle = angle + 2.0 * math.pi
        return angle


    def _smooth_yaw(self, yaw):
        for i in range(len(yaw) - 1):
            dyaw = yaw[i + 1] - yaw[i]
            while dyaw >= math.pi / 2.0:
                yaw[i + 1] -= math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]
            while dyaw <= -math.pi / 2.0:
                yaw[i + 1] += math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]

        return yaw


    def _get_linear_model_matrix(self, v, phi, delta):

        A = np.zeros((self.NX, self.NX))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.DT * math.cos(phi)
        A[0, 3] = - self.DT * v * math.sin(phi)
        A[1, 2] = self.DT * math.sin(phi)
        A[1, 3] = self.DT * v * math.cos(phi)
        A[3, 2] = self.DT * math.tan(delta) / self.WB

        B = np.zeros((self.NX, self.NU))
        B[2, 0] = self.DT
        B[3, 1] = self.DT * v / (self.WB * math.cos(delta) ** 2)

        C = np.zeros(self.NX)
        C[0] = self.DT * v * math.sin(phi) * phi
        C[1] = - self.DT * v * math.cos(phi) * phi
        C[3] = - self.DT * v * delta / (self.WB * math.cos(delta) ** 2)

        return A, B, C


    def _update_state(self, state, a, delta):
        # input check
        if delta >= self.MAX_STEER:
            delta = self.MAX_STEER
        elif delta <= -self.MAX_STEER:
            delta = -self.MAX_STEER

        state.x = state.x + state.v * math.cos(state.yaw) * self.DT
        state.y = state.y + state.v * math.sin(state.yaw) * self.DT
        state.yaw = state.yaw + state.v / self.WB * math.tan(delta) * self.DT
        state.v = state.v + a * self.DT

        if state. v > self.MAX_SPEED:
            state.v = self.MAX_SPEED
        elif state. v < self.MIN_SPEED:
            state.v = self.MIN_SPEED

        return state


    def _get_nparray_from_matrix(self, x):
        return np.array(x).flatten()


    def _calc_nearest_index(self, state, cx, cy, cyaw, pind):

        dx = [state.x - icx for icx in cx[pind:(pind + self.N_IND_SEARCH)]]
        dy = [state.y - icy for icy in cy[pind:(pind + self.N_IND_SEARCH)]]

        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        mind = min(d)
        ind = d.index(mind) + pind
        mind = math.sqrt(mind)

        dxl = cx[ind] - state.x
        dyl = cy[ind] - state.y

        angle = self._pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
        if angle < 0:
            mind *= -1

        return ind, mind


    def _predict_motion(self, x0, oa, od, xref):
        xbar = xref * 0.0
        for i, _ in enumerate(x0):
            xbar[i, 0] = x0[i]

        state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
        for (ai, di, i) in zip(oa, od, range(1, self.T + 1)):
            state = self._update_state(state, ai, di)
            xbar[0, i] = state.x
            xbar[1, i] = state.y
            xbar[2, i] = state.v
            xbar[3, i] = state.yaw

        return xbar


    def _iterative_linear_mpc_control(self, xref, x0, dref, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        """

        if oa is None or od is None:
            oa = [0.0] * self.T
            od = [0.0] * self.T

        for i in range(self.MAX_ITER):
            xbar = self._predict_motion(x0, oa, od, xref)
            poa, pod = oa[:], od[:]
            oa, od, ox, oy, oyaw, ov = self._linear_mpc_control(xref, xbar, x0, dref)
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            if du <= self.DU_TH:
                break
            else:
                print("Iterative is max iter")

        return oa, od, ox, oy, oyaw, ov


    def _linear_mpc_control(self, xref, xbar, x0, dref):
        """
        linear mpc control
        xref: reference point
        xbar: operational point
        x0: initial state
        dref: reference steer angle
        """

        x = cvxpy.Variable((self.NX, self.T + 1))
        u = cvxpy.Variable((self.NU, self.T))

        cost = 0.0
        constraints = []

        for t in range(self.T):
            cost += cvxpy.quad_form(u[:, t], self.R)

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], self.Q)

            A, B, C = self._get_linear_model_matrix(
                xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t] + C]

            if t < (self.T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], self.Rd)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= self.MAX_DSTEER * self.DT]
    
            cost += cvxpy.quad_form(xref[:, self.T] - x[:, self.T], self.Qf )

        constraints += [x[:, 0] == x0]
        constraints += [x[2, :] <= self.MAX_SPEED]
        constraints += [x[2, :] >= self.MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= self.MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) <= self.MAX_STEER]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = self._get_nparray_from_matrix(x.value[0, :])
            oy = self._get_nparray_from_matrix(x.value[1, :])
            ov = self._get_nparray_from_matrix(x.value[2, :])
            oyaw = self._get_nparray_from_matrix(x.value[3, :])
            oa = self._get_nparray_from_matrix(u.value[0, :])
            odelta = self._get_nparray_from_matrix(u.value[1, :])

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov


    def _calc_ref_trajectory(self, state, cx, cy, cyaw, ck, sp, dl, pind):
        xref = np.zeros((self.NX, self.T + 1))
        dref = np.zeros((1, self.T + 1))
        ncourse = len(cx)

        ind, _ = self._calc_nearest_index(state, cx, cy, cyaw, pind)

        if pind >= ind:
            ind = pind

        xref[0, 0] = cx[ind]
        xref[1, 0] = cy[ind]
        xref[2, 0] = sp[ind]
        xref[3, 0] = cyaw[ind]
        dref[0, 0] = 0.0  # steer operational point should be 0

        travel = 0.0

        for i in range(self.T + 1):
            travel += abs(state.v) * self.DT
            dind = int(round(travel / dl))

            if (ind + dind) < ncourse:
                xref[0, i] = cx[ind + dind]
                xref[1, i] = cy[ind + dind]
                xref[2, i] = sp[ind + dind]
                xref[3, i] = cyaw[ind + dind]
                dref[0, i] = 0.0
            else:
                xref[0, i] = cx[ncourse - 1]
                xref[1, i] = cy[ncourse - 1]
                xref[2, i] = sp[ncourse - 1]
                xref[3, i] = cyaw[ncourse - 1]
                dref[0, i] = 0.0

        return xref, ind, dref


    def _parse_vehicle_keys(self, keys, milliseconds):
        if keys[K_UP] or keys[K_w]:
            self._control.throttle = min(self._control.throttle + 0.01, 1)
        else:
            self._control.throttle = 0.0

        if keys[K_DOWN] or keys[K_s]:
            self._control.brake = min(self._control.brake + 0.2, 1)
        else:
            self._control.brake = 0

        steer_increment = 5e-4 * milliseconds
        if keys[K_LEFT] or keys[K_a]:
            if self._steer_cache > 0:
                self._steer_cache = 0
            else:
                self._steer_cache -= steer_increment
        elif keys[K_RIGHT] or keys[K_d]:
            if self._steer_cache < 0:
                self._steer_cache = 0
            else:
                self._steer_cache += steer_increment
        else:
            self._steer_cache = 0.0
        self._steer_cache = min(0.7, max(-0.7, self._steer_cache))
        self._control.steer = round(self._steer_cache, 1)
        self._control.hand_brake = keys[K_SPACE]

    def _parse_walker_keys(self, keys, milliseconds, world):
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
            self._control.speed = world.player_max_speed_fast if pygame.key.get_mods() & KMOD_SHIFT else world.player_max_speed
        self._control.jump = keys[K_SPACE]
        self._rotation.yaw = round(self._rotation.yaw, 1)
        self._control.direction = self._rotation.get_forward_vector()

    @staticmethod
    def _is_quit_shortcut(key):
        return (key == K_ESCAPE) or (key == K_q and pygame.key.get_mods() & KMOD_CTRL)


# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]
def plot_car(y, x, yaw, steer=0.0, cabcolor="-r", truckcolor="-k"):  # pragma: no cover

    outline = np.array([[-BACKTOWHEEL, (LENGTH - BACKTOWHEEL), (LENGTH - BACKTOWHEEL), -BACKTOWHEEL, -BACKTOWHEEL],
                        [WIDTH / 2, WIDTH / 2, - WIDTH / 2, -WIDTH / 2, WIDTH / 2]])

    fr_wheel = np.array([[WHEEL_LEN, -WHEEL_LEN, -WHEEL_LEN, WHEEL_LEN, WHEEL_LEN],
                         [-WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, WHEEL_WIDTH - TREAD, -WHEEL_WIDTH - TREAD]])

    rr_wheel = np.copy(fr_wheel)

    fl_wheel = np.copy(fr_wheel)
    fl_wheel[1, :] *= -1
    rl_wheel = np.copy(rr_wheel)
    rl_wheel[1, :] *= -1

    Rot1 = np.array([[math.cos(yaw), math.sin(yaw)],
                     [-math.sin(yaw), math.cos(yaw)]])
    Rot2 = np.array([[math.cos(steer), math.sin(steer)],
                     [-math.sin(steer), math.cos(steer)]])

    fr_wheel = (fr_wheel.T.dot(Rot2)).T
    fl_wheel = (fl_wheel.T.dot(Rot2)).T
    fr_wheel[0, :] += WB
    fl_wheel[0, :] += WB

    fr_wheel = (fr_wheel.T.dot(Rot1)).T
    fl_wheel = (fl_wheel.T.dot(Rot1)).T

    outline = (outline.T.dot(Rot1)).T
    rr_wheel = (rr_wheel.T.dot(Rot1)).T
    rl_wheel = (rl_wheel.T.dot(Rot1)).T

    outline[0, :] += y
    outline[1, :] += x
    fr_wheel[0, :] += y
    fr_wheel[1, :] += x
    rr_wheel[0, :] += y
    rr_wheel[1, :] += x
    fl_wheel[0, :] += y
    fl_wheel[1, :] += x
    rl_wheel[0, :] += y
    rl_wheel[1, :] += x

    plt.plot(np.array(outline[0, :]).flatten(),
             np.array(outline[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fr_wheel[0, :]).flatten(),
             np.array(fr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rr_wheel[0, :]).flatten(),
             np.array(rr_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(fl_wheel[0, :]).flatten(),
             np.array(fl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(np.array(rl_wheel[0, :]).flatten(),
             np.array(rl_wheel[1, :]).flatten(), truckcolor)
    plt.plot(y, x, "*")