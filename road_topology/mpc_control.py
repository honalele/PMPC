#!/usr/bin/env python

"""
Welcome to CARLA MPC control of lane change scenarios

You can also control with steering wheel Logitech G923.

To drive start by preshing the brake pedal.
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

driving_data_dir = '/home/hona/carla/driving_data/'
trajectory_file = '/home/hona/carla/PythonAPI/road_topology/points_carla.csv'
SHOW_ANIMATION = True 
MAX_TIME = 0
TARGET_SPEED = 80   

# ==============================================================================
# -- HUD -----------------------------------------------------------------------
# ==============================================================================

class HUD(object):
    def __init__(self, width, height):
        self.dim = (width, height)
        font = pygame.font.Font(pygame.font.get_default_font(), 20)
        font_name = 'courier' if os.name == 'nt' else 'mono'
        fonts = [x for x in pygame.font.get_fonts() if font_name in x]
        default_font = 'ubuntumono'
        mono = default_font if default_font in fonts else fonts[0]
        mono = pygame.font.match_font(mono)
        self._font_mono = pygame.font.Font(mono, 12 if os.name == 'nt' else 14)
        self._notifications = FadingText(font, (width, 40), (0, height - 40))
        self.help = HelpText(pygame.font.Font(mono, 24), width, height)
        self.server_fps = 0
        self.frame = 0
        self.simulation_time = 0
        self._show_info = True
        self._info_text = []
        self._server_clock = pygame.time.Clock()

    def on_world_tick(self, timestamp):
        self._server_clock.tick()
        self.server_fps = self._server_clock.get_fps()
        self.frame = timestamp.frame
        self.simulation_time = timestamp.elapsed_seconds

    def tick(self, world, mapname, clock):
        self._notifications.tick(world, clock)
        if not self._show_info:
            return
        t = world.player.get_transform()
        v = world.player.get_velocity()
        c = world.player.get_control()
        heading = 'N' if abs(t.rotation.yaw) < 89.5 else ''
        heading += 'S' if abs(t.rotation.yaw) > 90.5 else ''
        heading += 'E' if 179.5 > t.rotation.yaw > 0.5 else ''
        heading += 'W' if -0.5 > t.rotation.yaw > -179.5 else ''
        colhist = world.collision_sensor.get_collision_history()
        collision = [colhist[x + self.frame - 200] for x in range(0, 200)]
        max_col = max(1.0, max(collision))
        collision = [x / max_col for x in collision]
        vehicles = world.world.get_actors().filter('vehicle.*')
        self._info_text = [
            'Server:  % 16.0f FPS' % self.server_fps,
            'Client:  % 16.0f FPS' % clock.get_fps(),
            '',
            'Vehicle: % 20s' % get_actor_display_name(world.player, truncate=20),
            'Map:     % 20s' % world.map.name,
            'Simulation time: % 12s' % datetime.timedelta(seconds=int(self.simulation_time)),
            '',
            'Speed:   % 15.0f km/h' % (3.6 * math.sqrt(v.x**2 + v.y**2 + v.z**2)),
            u'Heading:% 16.0f\N{DEGREE SIGN} % 2s' % (t.rotation.yaw, heading),
            'Location:% 20s' % ('(% 5.1f, % 5.1f)' % (t.location.x, t.location.y)),
            'GNSS:% 24s' % ('(% 2.6f, % 3.6f)' % (world.gnss_sensor.lat, world.gnss_sensor.lon)),
            'Height:  % 18.0f m' % t.location.z,
            '']
        if isinstance(c, carla.VehicleControl):
            self._info_text += [
                ('Throttle:', c.throttle, 0.0, 1.0),
                ('Steer:', c.steer, -1.0, 1.0),
                ('Brake:', c.brake, 0.0, 1.0),
                ('Reverse:', c.reverse),
                ('Hand brake:', c.hand_brake),
                ('Manual:', c.manual_gear_shift),
                'Gear:        %s' % {-1: 'R', 0: 'N'}.get(c.gear, c.gear)]
        elif isinstance(c, carla.WalkerControl):
            self._info_text += [
                ('Speed:', c.speed, 0.0, 5.556),
                ('Jump:', c.jump)]
        self._info_text += [
            '',
            'Collision:',
            collision,
            '',
            'Number of vehicles: % 8d' % len(vehicles)]
        if len(vehicles) > 1:
            self._info_text += ['Nearby vehicles:']
            distance = lambda l: math.sqrt((l.x - t.location.x)**2 + (l.y - t.location.y)**2 + (l.z - t.location.z)**2)
            vehicles = [(distance(x.get_location()), x) for x in vehicles if x.id != world.player.id]
            for d, vehicle in sorted(vehicles):
                if d > 200.0:
                    break
                vehicle_type = get_actor_display_name(vehicle, truncate=22)
                self._info_text.append('% 4dm %s' % (d, vehicle_type))

    def toggle_info(self):
        self._show_info = not self._show_info

    def notification(self, text, seconds=2.0):
        self._notifications.set_text(text, seconds=seconds)

    def error(self, text):
        self._notifications.set_text('Error: %s' % text, (255, 0, 0))

    def render(self, display):
        if self._show_info:
            info_surface = pygame.Surface((220, self.dim[1]))
            info_surface.set_alpha(100)
            display.blit(info_surface, (0, 0))
            v_offset = 4
            bar_h_offset = 100
            bar_width = 106
            for item in self._info_text:
                if v_offset + 18 > self.dim[1]:
                    break
                if isinstance(item, list):
                    if len(item) > 1:
                        points = [(x + 8, v_offset + 8 + (1.0 - y) * 30) for x, y in enumerate(item)]
                        pygame.draw.lines(display, (255, 136, 0), False, points, 2)
                    item = None
                    v_offset += 18
                elif isinstance(item, tuple):
                    if isinstance(item[1], bool):
                        rect = pygame.Rect((bar_h_offset, v_offset + 8), (6, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect, 0 if item[1] else 1)
                    else:
                        rect_border = pygame.Rect((bar_h_offset, v_offset + 8), (bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect_border, 1)
                        f = (item[1] - item[2]) / (item[3] - item[2])
                        if item[2] < 0.0:
                            rect = pygame.Rect((bar_h_offset + f * (bar_width - 6), v_offset + 8), (6, 6))
                        else:
                            rect = pygame.Rect((bar_h_offset, v_offset + 8), (f * bar_width, 6))
                        pygame.draw.rect(display, (255, 255, 255), rect)
                    item = item[0]
                if item:  # At this point has to be a str.
                    surface = self._font_mono.render(item, True, (255, 255, 255))
                    display.blit(surface, (8, v_offset))
                v_offset += 18
        self._notifications.render(display)
        self.help.render(display)


def find_weather_presets():
    rgx = re.compile('.+?(?:(?<=[a-z])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])|$)')
    name = lambda x: ' '.join(m.group(0) for m in rgx.finditer(x))
    presets = [x for x in dir(carla.WeatherParameters) if re.match('[A-Z].+', x)]
    return [(getattr(carla.WeatherParameters, x), name(x)) for x in presets]


def get_actor_display_name(actor, truncate=250):
    name = ' '.join(actor.type_id.replace('_', '.').title().split('.')[1:])
    return (name[:truncate - 1] + u'\u2026') if len(name) > truncate else name
# ==============================================================================
# -- FadingText ----------------------------------------------------------------
# ==============================================================================
class FadingText(object):
    def __init__(self, font, dim, pos):
        self.font = font
        self.dim = dim
        self.pos = pos
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)

    def set_text(self, text, color=(255, 255, 255), seconds=2.0):
        text_texture = self.font.render(text, True, color)
        self.surface = pygame.Surface(self.dim)
        self.seconds_left = seconds
        self.surface.fill((0, 0, 0, 0))
        self.surface.blit(text_texture, (10, 11))

    def tick(self, _, clock):
        delta_seconds = 1e-3 * clock.get_time()
        self.seconds_left = max(0.0, self.seconds_left - delta_seconds)
        self.surface.set_alpha(500.0 * self.seconds_left)

    def render(self, display):
        display.blit(self.surface, self.pos)


# ==============================================================================
# -- HelpText ------------------------------------------------------------------
# ==============================================================================
class HelpText(object):
    def __init__(self, font, width, height):
        lines = __doc__.split('\n')
        self.font = font
        self.dim = (680, len(lines) * 22 + 12)
        self.pos = (0.5 * width - 0.5 * self.dim[0], 0.5 * height - 0.5 * self.dim[1])
        self.seconds_left = 0
        self.surface = pygame.Surface(self.dim)
        self.surface.fill((0, 0, 0, 0))
        for n, line in enumerate(lines):
            text_texture = self.font.render(line, True, (255, 255, 255))
            self.surface.blit(text_texture, (22, n * 22))
            self._render = False
        self.surface.set_alpha(220)

    def toggle(self):
        self._render = not self._render

    def render(self, display):
        if self._render:
            display.blit(self.surface, self.pos)


# ==============================================================================
# -- CollisionSensor -----------------------------------------------------------
# ==============================================================================
class CollisionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.history = []
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.collision')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: CollisionSensor._on_collision(weak_self, event))

    def get_collision_history(self):
        history = collections.defaultdict(int)
        for frame, intensity in self.history:
            history[frame] += intensity
        return history

    @staticmethod
    def _on_collision(weak_self, event):
        self = weak_self()
        if not self:
            return
        actor_type = get_actor_display_name(event.other_actor)
        self.hud.notification('Collision with %r' % actor_type)
        impulse = event.normal_impulse
        intensity = math.sqrt(impulse.x**2 + impulse.y**2 + impulse.z**2)
        self.history.append((event.frame, intensity))
        if len(self.history) > 4000:
            self.history.pop(0)


# ==============================================================================
# -- LaneInvasionSensor --------------------------------------------------------
# ==============================================================================
class LaneInvasionSensor(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self._parent = parent_actor
        self.hud = hud
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.lane_invasion')
        self.sensor = world.spawn_actor(bp, carla.Transform(), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: LaneInvasionSensor._on_invasion(weak_self, event))

    @staticmethod
    def _on_invasion(weak_self, event):
        self = weak_self()
        if not self:
            return
        lane_types = set(x.type for x in event.crossed_lane_markings)
        text = ['%r' % str(x).split()[-1] for x in lane_types]
        self.hud.notification('Crossed line %s' % ' and '.join(text))

# ==============================================================================
# -- GnssSensor --------------------------------------------------------
# ==============================================================================
class GnssSensor(object):
    def __init__(self, parent_actor):
        self.sensor = None
        self._parent = parent_actor
        self.lat = 0.0
        self.lon = 0.0
        world = self._parent.get_world()
        bp = world.get_blueprint_library().find('sensor.other.gnss')
        self.sensor = world.spawn_actor(bp, carla.Transform(carla.Location(x=1.0, z=2.8)), attach_to=self._parent)
        # We need to pass the lambda a weak reference to self to avoid circular
        # reference.
        weak_self = weakref.ref(self)
        self.sensor.listen(lambda event: GnssSensor._on_gnss_event(weak_self, event))

    @staticmethod
    def _on_gnss_event(weak_self, event):
        self = weak_self()
        if not self:
            return
        self.lat = event.latitude
        self.lon = event.longitude


# ==============================================================================
# -- CameraManager -------------------------------------------------------------
# ==============================================================================
class CameraManager(object):
    def __init__(self, parent_actor, hud):
        self.sensor = None
        self.surface = None
        self._parent = parent_actor
        self.hud = hud
        self.recording = False
        self._camera_transforms = [
            carla.Transform(carla.Location(x=-5.5, z=2.8), carla.Rotation(pitch=-15)),
            carla.Transform(carla.Location(x=1.6, z=1.7))]
        self.transform_index = 1
        self.sensors = [
            ['sensor.camera.rgb', cc.Raw, 'Camera RGB'],
            ['sensor.camera.depth', cc.Raw, 'Camera Depth (Raw)'],
            ['sensor.camera.depth', cc.Depth, 'Camera Depth (Gray Scale)'],
            ['sensor.camera.depth', cc.LogarithmicDepth, 'Camera Depth (Logarithmic Gray Scale)'],
            ['sensor.camera.semantic_segmentation', cc.Raw, 'Camera Semantic Segmentation (Raw)'],
            ['sensor.camera.semantic_segmentation', cc.CityScapesPalette,
                'Camera Semantic Segmentation (CityScapes Palette)'],
            ['sensor.lidar.ray_cast', None, 'Lidar (Ray-Cast)']]
        world = self._parent.get_world()
        bp_library = world.get_blueprint_library()
        for item in self.sensors:
            bp = bp_library.find(item[0])
            if item[0].startswith('sensor.camera'):
                bp.set_attribute('image_size_x', str(hud.dim[0]))
                bp.set_attribute('image_size_y', str(hud.dim[1]))
            elif item[0].startswith('sensor.lidar'):
                bp.set_attribute('range', '50')
            item.append(bp)
        self.index = None

    def toggle_camera(self):
        self.transform_index = (self.transform_index + 1) % len(self._camera_transforms)
        self.sensor.set_transform(self._camera_transforms[self.transform_index])

    def set_sensor(self, index, notify=True):
        index = index % len(self.sensors)
        needs_respawn = True if self.index is None \
            else self.sensors[index][0] != self.sensors[self.index][0]
        if needs_respawn:
            if self.sensor is not None:
                self.sensor.destroy()
                self.surface = None
            self.sensor = self._parent.get_world().spawn_actor(
                self.sensors[index][-1],
                self._camera_transforms[self.transform_index],
                attach_to=self._parent)
            # We need to pass the lambda a weak reference to self to avoid
            # circular reference.
            weak_self = weakref.ref(self)
            self.sensor.listen(lambda image: CameraManager._parse_image(weak_self, image))
        if notify:
            self.hud.notification(self.sensors[index][2])
        self.index = index

    def next_sensor(self):
        self.set_sensor(self.index + 1)

    def toggle_recording(self):
        self.recording = not self.recording
        self.hud.notification('Recording %s' % ('On' if self.recording else 'Off'))

    def render(self, display):
        if self.surface is not None:
            display.blit(self.surface, (0, 0))

    @staticmethod
    def _parse_image(weak_self, image):
        self = weak_self()
        if not self:
            return
        if self.sensors[self.index][0].startswith('sensor.lidar'):
            points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
            points = np.reshape(points, (int(points.shape[0] / 3), 3))
            lidar_data = np.array(points[:, :2])
            lidar_data *= min(self.hud.dim) / 100.0
            lidar_data += (0.5 * self.hud.dim[0], 0.5 * self.hud.dim[1])
            lidar_data = np.fabs(lidar_data) # pylint: disable=E1111
            lidar_data = lidar_data.astype(np.int32)
            lidar_data = np.reshape(lidar_data, (-1, 2))
            lidar_img_size = (self.hud.dim[0], self.hud.dim[1], 3)
            lidar_img = np.zeros(lidar_img_size)
            lidar_img[tuple(lidar_data.T)] = (255, 255, 255)
            self.surface = pygame.surfarray.make_surface(lidar_img)
        else:
            image.convert(self.sensors[self.index][1])
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            self.surface = pygame.surfarray.make_surface(array.swapaxes(0, 1))
        if self.recording:
            image.save_to_disk('_out/%08d' % image.frame)


# ==============================================================================
# -- World ---------------------------------------------------------------------
# ==============================================================================
class World(object):
    def __init__(self, carla_world, hud):
        self.world = carla_world
        self.mapname = carla_world.get_map().name
        self.hud = hud
        self.world.on_tick(hud.on_world_tick)
        self.world.wait_for_tick(10.0)
        self.player = None
        try:
            self.map = self.world.get_map()
        except RuntimeError as error:
            print('RuntimeError: {}'.format(error))
            print('  The server could not send the OpenDRIVE (.xodr) file:')
            print('  Make sure it exists, has the same name of your town, and is correct.')
        
        while self.player is None:
            print("Scenario not yet ready")
            time.sleep(1)
            possible_vehicles = self.world.get_actors().filter('vehicle.*')
            for vehicle in possible_vehicles:
                if vehicle.attributes['role_name'] == "hero":
                    self.player = vehicle
        self.vehicle_name = self.player.type_id
        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager.set_sensor(0, notify=False)
        self.controller = None
        self.gnss_sensor = GnssSensor(self.player)
        self._weather_presets = find_weather_presets()
        self._weather_index = 0

    def restart(self):
        cam_index = self.camera_manager._index
        cam_pos_index = self.camera_manager._transform_index

        start_pose = self.player.get_transform()
        start_pose.location.z += 2.0
        start_pose.rotation.roll = 0.0
        start_pose.rotation.pitch = 0.0
        blueprint = self._get_random_blueprint()
        self.destroy()
        self.player = self.world.spawn_actor(blueprint, start_pose)

        self.collision_sensor = CollisionSensor(self.player, self.hud)
        self.lane_invasion_sensor = LaneInvasionSensor(self.player, self.hud)
        self.gnss_sensor = GnssSensor(self.player)
        self.camera_manager = CameraManager(self.player, self.hud)
        self.camera_manager._transform_index = cam_pos_index
        self.camera_manager.set_sensor(cam_index, notify=False)
        actor_type = get_actor_display_name(self.player)
        self.hud.notification(actor_type)

    def next_weather(self, reverse=False):
        self._weather_index += -1 if reverse else 1
        self._weather_index %= len(self._weather_presets)
        preset = self._weather_presets[self._weather_index]
        self.hud.notification('Weather: %s' % preset[1])
        self.player.get_world().set_weather(preset[0])

    def tick(self, clock):
        if len(self.world.get_actors().filter(self.vehicle_name)) < 1:
            print("Scenario ended -- Terminating")
            return False

        self.hud.tick(self, self.mapname, clock)
        return True

    def render(self, display):
        self.camera_manager.render(display)
        self.hud.render(display)

    def destroy(self):
        actors = [
            self.camera_manager.sensor,
            self.collision_sensor.sensor,
            self.lane_invasion_sensor.sensor,
            self.player]
        for actor in actors:
            if actor is not None:
                actor.destroy()




# ==============================================================================
# -- ModelPredictiveControl -------------------------------------------------------------
# ==============================================================================
ego_vehicle_log = []
ego_vehicle_u_log = []
surr_vehicle_log = []
preds_ego = []
preds_surr = []

class ModelPredictiveControl(object):
    def __init__(self, world, target_traj, start_in_autopilot, args):
        self.dt = 0.1
        self.target_traj = target_traj
        self.horizon = 5
        self._autopilot_enabled = start_in_autopilot
        self.args = args
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
            
            self._parse_naren_control(world)
            world.player.apply_control(self._control)

    def _parse_naren_control(self, world):
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

        #print('Ego_vehicle x:{} y:{} z:{} yaw:{} vx{} vy{} delta{} a{} b{}' .format(x_t, y_t, t.location.z, yaw_t, vx_t, vy_t, delta_t, a_t, b_t))
        #print('lc_x:{}, lc_y:{}, lc_yaw:{}'.format(lc_x, lc_y, lc_yaw))
        
        
        idx, goal = self._find_nearest_index(x_t, y_t, lc_x, lc_y)
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

            lc_x, lc_y, lc_yaw = self.target_traj

            LANE_CHANGE_TIMING = 150
            LANE_CHANGE_TIMING_END = 160
            TAEGET_V = 35
            MAX_V = 60
            SAFE_DISTANCE = 10

            d_min = 10000
            for su_vehice in surr_states:
                sx_t, sy_t, syaw_t, sv_t, sd_t, sa_t = su_vehice
                if np.sqrt((sx_t - x_t)**2 + ((sy_t - y_t)**2)) < d_min:
                    d_min = np.sqrt((sx_t - x_t)**2 + ((sy_t - y_t)**2))
            #if d_min < SAFE_DISTANCE:
             #   self._control.brake +=  0.10

            print('x{},y{}, targetx{}, targety{}'.format(x_t, y_t, lc_x[0], lc_y[0]))

            if v_t < TAEGET_V:
                self._control.throttle += 0.3
                self._control.brake = 0
                self._control.throttle -= 0.10
                self._control.brake = 0

            if v_t > MAX_V:
                self._control.throttle = 0
                self._control.brake += 0.10

            SCENARIO_NUM = self.args.scenario_id
            print(self.args.scenario_id)

            if d_min < 15:
            	self._control.brake += 0.20

            if SCENARIO_NUM == 3:
                if idx > LANE_CHANGE_TIMING:
                    self._control.steer += 0.02
                if idx > LANE_CHANGE_TIMING_END:
                    self._control.steer -= 0.02

            if SCENARIO_NUM == 4:
                if idx > LANE_CHANGE_TIMING:
                    self._control.steer += 0.1
                    self._control.throttle += 0.10
                if idx > LANE_CHANGE_TIMING_END:
                    self._control.steer -= 0.1
                    self._control.throttle -= 0.06

            else:
                if idx < LANE_CHANGE_TIMING:
                    self._control.steer = 0.0
                if idx > LANE_CHANGE_TIMING:
                    self._control.steer -= 0.06
                if idx > LANE_CHANGE_TIMING_END:
                    self._control.steer  = 0

            print(self._control.steer)


            

    def _get_control(self, ego_control, *args):
            first = True
            u_solution = minimize(self._cost_function,
                                        x0=ego_control,
                                        args=(ego_state, surr_states, idx, first),                          
                                        method='SLSQP',
                                        tol = 1e-8)
            predicted_ego = []
            predicted_surr = []
            for n in range(1, self.horizon):
                [ego_state, surr_states] = self._predict_motion(ego_state, u_solution.x, surr_states)
                predicted_ego.append(ego_state)
                predicted_surr.append(surr_states)
            preds_ego.append(predicted_ego)
            preds_surr.append(predicted_surr)
            first = False
            u_solution = minimize(self._cost_function,
                                        x0=u_solution.x,
                                        args=(predicted_ego, predicted_surr, idx, first),                          
                                        method='SLSQP',
                                        tol = 1e-8)

            print('Cost value :{}'.format(u_solution))
            steerCmd = u_solution.x[0]
            pedalCmd = u_solution.x[1]

            if steerCmd < 1 and  steerCmd>=-1:
                self._control.steer = steerCmd
            elif steerCmd > 1:
                self._control.steer = 1
            else:
                self._control.steer = -1

            if pedalCmd > 0:
                self._control.throttle = min([pedalCmd, 1])
                self._control.brake = 0
            else:
                self._control.throttle = 0
                self._control.brake = min([np.abs(pedalCmd), 1])
            ego_vehicle_u_log.append([self._control.steer, self._control.throttle, self._control.brake])
            

     
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
            cost += 100*np.sqrt((x_t - lc_x[idx])**2 + (y_t - lc_y[idx])**2)
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



def load_scenario(scenario_num):
    [target_x, target_y] = get_trajectory(scenario_num)
    target_yaw = get_yaw_angle(target_x, target_y)    
    return target_x, target_y, target_yaw



def get_yaw_angle(x, y):
    yaw = []
    for i in range(0, len(x)):
        if i == len(x)-1:
            yaw.append(yaw[i-1])
            break
        dx = x[i+1] - x[i]
        dy = y[i+1] - y[i]
        yaw.append(np.arctan(dx/(dy+0.00001)))
    return yaw

def generate_lane_change(target_x, target_y):
    lc_x = target_x
    LENGTH = int(len(target_x)/5)
    
    for i in range(LENGTH, len(target_x)):
        lc_x[i] = target_x[i]- 2.5 
    target_yaw = get_yaw_angle(lc_x, target_y)   
    return lc_x, target_y, target_yaw


def get_map():
    points = pd.read_csv(trajectory_file)
    ids = points.iloc[:,0]
    all_x = points.iloc[:,1]
    all_y = points.iloc[:,2]
    all_z = points.iloc[:,3]  
    return [all_x, all_y, all_z]



def get_trajectory(senario_num):
    csv ='scenario_' + str(senario_num) + '.csv'
    sl_csv = os.path.join(driving_data_dir, csv)
    sl = pd.read_csv(sl_csv)
    sx = sl.x.tolist()
    sy = sl.y.tolist()   
    return [sx, sy]
 


def visualize_course():
    all_x, all_y, all_z = get_map()
    plt.figure(figsize=(15,8))
    plt.plot(all_x,all_y,'k.',alpha=0.1)
    for i in range(1,6):
        [sx, sy] = get_trajectory(i)
        plt.plot(sx, sy, marker='.', label=('Scenario_{}'.format(i)))
    plt.grid()
    plt.xlabel('trajectory-x')
    plt.ylabel('trajectory-y')
    plt.axis('equal')
    plt.legend()
    plt.savefig('./fig/scenario.png')
    plt.show()



def game_loop(args):
    pygame.init()
    pygame.font.init()
    world = None
    
    try:
        client = carla.Client(args.host, args.port)
        client.set_timeout(2.0)
        
        display = pygame.display.set_mode(
            (1280, 720),
            pygame.HWSURFACE | pygame.DOUBLEBUF)
        
        target_x, target_y, target_yaw = load_scenario(1)
        lc_x, lc_y, lc_yaw = generate_lane_change(target_x, target_y)
        hud = HUD(1280, 720)
        world = World(client.get_world(), hud)
        #world = World(client.get_world(), hud, [lc_x, lc_y, lc_yaw])

        controller = ModelPredictiveControl(world, [lc_x, lc_y, lc_yaw], args.autopilot, args)
        clock = pygame.time.Clock()
        while True:
            clock.tick_busy_loop(60)
            if controller.parse_events(world, clock):
                return
            world.tick(clock)
            world.render(display)
            pygame.display.flip()
    finally:
        if world is not None:
            world.destroy()
        pygame.quit()        


def run_scenario(scenario_num):   
    target_x, target_y, target_yaw = load_scenario(scenario_num)
    lc_x, lc_y, lc_yaw = generate_lane_change(target_x, target_y)
    all_x, all_y, all_z = get_map()
    
    initial_state = State(x=target_x[0], y=target_y[0], yaw=target_yaw[0], v=TARGET_SPEED)
    surrounding_state = State(x=target_x[0], y=target_y[0] + 50, yaw=target_yaw[0], v=TARGET_SPEED)
    
    time_step = 0
    xmin = min(target_x)
    xmax = max(target_x)
    ymin = min(target_y)
    ymax = max(target_y)
    
    while MAX_TIME >= time_step: 
        if SHOW_ANIMATION:
            
            oa, od, ostate, zbar = control.mpc_control(xref, x0, dref, oa, od)
            
            plt.figure(figsize=(15,8))
            plt.cla()
            plt.plot(all_x,all_y,'k.',alpha=0.1)
            plt.plot(target_x, target_y, "b.", label="StraightPath")
            plt.plot(lc_x, lc_y, "r.", label="TargetPath")
            time_step +=1
            z0 = initial_state
            plot_car(target_x[time_step], target_y[time_step], target_yaw[time_step], steer=0.0, cabcolor="-r", truckcolor="-w")
            plt.grid()
            plt.axis('equal')
            plt.grid()
            plt.xlabel('trajectory-x')
            plt.ylabel('trajectory-y')
            plt.xlim([xmin-15, xmax+5])
            plt.ylim([ymin-15, ymax+5])
            plt.legend()
            plt.show()

# ==============================================================================
# -- main() --------------------------------------------------------------------
# ==============================================================================
def main():
    argparser = argparse.ArgumentParser(
        description='CARLA Manual Control Client')
    argparser.add_argument(
        '-v', '--verbose',
        action='store_true',
        dest='debug',
        help='print debug information')
    argparser.add_argument(
        '--host',
        metavar='H',
        default='127.0.0.1',
        help='IP of the host server (default: 127.0.0.1)')
    argparser.add_argument(
        '-p', '--port',
        metavar='P',
        default=2000,
        type=int,
        help='TCP port to listen to (default: 2000)')
    argparser.add_argument(
        '-a', '--autopilot',
        action='store_true',
        help='enable autopilot')
    argparser.add_argument(
        '--res',
        metavar='WIDTHxHEIGHT',
        default='1280x720',
        help='window resolution (default: 1280x720)')
    argparser.add_argument(
        '--filter',
        metavar='PATTERN',
        default='vehicle.*',
        help='actor filter (default: "vehicle.*")')
    argparser.add_argument(
        '--scenario_id',
        metavar='SCENARIOTID',
        default=0,
        type=int,
        help='please input the scenario id (default is 0)')
    args = argparser.parse_args()

    args.width, args.height = [int(x) for x in args.res.split('x')]

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(format='%(levelname)s: %(message)s', level=log_level)

    logging.info('listening to server %s:%s', args.host, args.port)

    print(__doc__)

    try:

        game_loop(args)

    except KeyboardInterrupt:
        print('\nCancelled by user. Bye!')
        """

    for i in range(len(ego_vehicle_log)):
        x_t, y_t, phi_t, v_t = ego_vehicle_log[i]

        plt.plot(x_t, y_t, 'k.')
        for surr in (surr_vehicle_log[i]):
            sx_t, sy_t, syaw_t, sv_t, sd_t, sa_t = surr
            plt.plot(sx_t, sy_t, 'r.')
        for pred_ego_n in preds_ego:
            x_t_p, y_t_p, phi_t_p, v_t_p = pred_ego_n
            plt.plot(x_t_p, y_t_p, 'g.')

    plt.grid()
    plt.axis('equal')
    plt.xlabel('trajectory-x')
    plt.ylabel('trajectory-y')
    plt.xlim([-800, 800])
    plt.ylim([-400, 450])
    plt.show()

    steer = []
    for data in ego_vehicle_u_log:
        steer.append(ego_vehicle_u_log[0])
    plt.figure()
    plt.plot(steer)
    plt.show()

"""

if __name__ == '__main__':

    main()    


def run_scenario(scenario_num):
    
    target_x, target_y, target_yaw = load_scenario(scenario_num)
    lc_x, lc_y, lc_yaw = generate_lane_change(target_x, target_y)
    all_x, all_y, all_z = get_map()
    
    initial_state = State(x=target_x[0], y=target_y[0], yaw=target_yaw[0], v=TARGET_SPEED)
    surrounding_state = State(x=target_x[0], y=target_y[0] + 50, yaw=target_yaw[0], v=TARGET_SPEED)
    
    time_step = 0
    xmin = min(target_x)
    xmax = max(target_x)
    ymin = min(target_y)
    ymax = max(target_y)
    
    while MAX_TIME >= time_step: 
        if SHOW_ANIMATION:
                        
            plt.figure(figsize=(15,8))
            plt.cla()
            plt.plot(all_x,all_y,'k.',alpha=0.1)
            plt.plot(target_x, target_y, "b.", label="StraightPath")
            plt.plot(lc_x, lc_y, "r.", label="TargetPath")
            time_step +=1
            z0 = initial_state
            plot_car(target_x[time_step], target_y[time_step], target_yaw[time_step], steer=0.0, cabcolor="-r", truckcolor="-w")
            plt.grid()
            plt.axis('equal')
            plt.xlabel('trajectory-x')
            plt.ylabel('trajectory-y')
            plt.xlim([xmin-15, xmax+5])
            plt.ylim([ymin-15, ymax+5])
            plt.legend()
            plt.show()
            
