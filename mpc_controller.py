#!/usr/bin/env python3

"""
2D Controller Class to be used for the CARLA waypoint follower.

author: Hona
"""

import cutils
import numpy as np
import cvxpy
import math
import cubic_spline_planner

class ModelPredictiveController(object):
    def __init__(self, waypoints):
        self.vars                = cutils.CUtils()
        self._current_x          = 0
        self._current_y          = 0
        self._current_yaw        = 0
        self._current_speed      = 0
        self._desired_speed      = 0
        self._current_frame      = 0
        self._current_timestamp  = 0
        self._start_control_loop = False
        self._set_throttle       = 0
        self._set_brake          = 0
        self._set_steer          = 0
        self._waypoints          = waypoints
        self._conv_rad_to_steer  = 180.0 / 70.0 / np.pi
        self._pi                 = np.pi
        self._2pi                = 2.0 * np.pi

    def update_values(self, x, y, yaw, speed, timestamp, frame):
        self._current_x         = x
        self._current_y         = y
        self._current_yaw       = yaw
        self._current_speed     = speed
        self._current_timestamp = timestamp
        self._current_frame     = frame
        if self._current_frame:
            self._start_control_loop = True

    def update_desired_speed(self):
        min_idx       = 0
        min_dist      = float("inf")
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
        self._desired_speed = desired_speed

    def update_waypoints(self, new_waypoints):
        self._waypoints = new_waypoints

    def get_commands(self):
        return self._set_throttle, self._set_steer, self._set_brake

    def set_throttle(self, input_throttle):
        # Clamp the throttle command to valid bounds
        throttle           = np.fmax(np.fmin(input_throttle, 1.0), 0.0)
        self._set_throttle = throttle

    def set_steer(self, input_steer_in_rad):
        # Covnert radians to [-1, 1]
        input_steer = self._conv_rad_to_steer * input_steer_in_rad

        # Clamp the steering command to valid bounds
        steer           = np.fmax(np.fmin(input_steer, 1.0), -1.0)
        self._set_steer = steer

    def set_brake(self, input_brake):
        # Clamp the steering command to valid bounds
        brake           = np.fmax(np.fmin(input_brake, 1.0), 0.0)
        self._set_brake = brake
        
        
        
    def get_linear_model_matrix(self, v, phi, delta):
        
        NX = 4   # x = x, y, v, yaw
        NU = 2  # a = [accel, steer]
        DT = 0.2
        WB = 2.5
        A = np.zeros((NX, NX))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = DT * math.cos(phi)
        A[0, 3] = - DT * v * math.sin(phi)
        A[1, 2] = DT * math.sin(phi)
        A[1, 3] = DT * v * math.cos(phi)
        A[3, 2] = DT * math.tan(delta) / WB
        
        B = np.zeros((NX, NU))
        B[2, 0] = DT
        B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)
        
        C = np.zeros(NX)
        C[0] = DT * v * math.sin(phi) * phi
        C[1] = - DT * v * math.cos(phi) * phi
        C[3] = - DT * v * delta / (WB * math.cos(delta) ** 2)
        
        return A, B, C
        
   
    def pi_2_pi(self, angle):
        while(angle > math.pi):
            angle = angle - 2.0 * math.pi
       
        while(angle < -math.pi):
            angle = angle + 2.0 * math.pi
           
        return angle
   
   
    def smooth_yaw(self, yaw):
        for i in range(len(yaw) - 1):
            dyaw = yaw[i + 1] - yaw[i]

            while dyaw >= math.pi / 2.0:
                yaw[i + 1] -= math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]

            while dyaw <= -math.pi / 2.0:
                yaw[i + 1] += math.pi * 2.0
                dyaw = yaw[i + 1] - yaw[i]

        return yaw
        
   
    def calc_nearest_index(self, state, cx, cy, cyaw, pind):
        
        N_IND_SEARCH = 10
        [x, y, yaw, v] = state
        dx = [x - icx for icx in cx[pind:(pind + N_IND_SEARCH)]]
        dy = [y - icy for icy in cy[pind:(pind + N_IND_SEARCH)]]
        d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]
        
        mind = min(d)
        ind = d.index(mind) + pind
        mind = math.sqrt(mind)
        dxl = cx[ind] - x
        dyl = cy[ind] - y
        
        angle = self.pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))

        if angle < 0:
            mind *= -1
            
        return ind, mind
        
    
    def get_nparray_from_matrix(self, x):
        return np.array(x).flatten()
        
        
    def calc_ref_trajectory(self, state, cx, cy, cyaw, ck, sp, dl, pind):
        [x, y, yaw, v] = state
        NX = 4
        T = 5
        DT = 0.2 # [s] time tick
        xref = np.zeros((NX, T + 1))
        dref = np.zeros((1, T + 1))
        ncourse = len(cx)

        ind, _ = self.calc_nearest_index(state, cx, cy, cyaw, pind)

        if pind >= ind:
            ind = pind

        xref[0, 0] = cx[ind]
        xref[1, 0] = cy[ind]
        xref[2, 0] = sp[ind]
        xref[3, 0] = cyaw[ind]
        dref[0, 0] = 0.0  # steer operational point should be 0

        travel = 0.0

        for i in range(T + 1):
            travel += abs(v) * DT
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
        
    
    def update_state(self, state, a, delta):
        DT = 0.2
        WB = 2.5
        # input check
        [x, y, yaw, v] = state
        x = x + v * math.cos(yaw) * DT
        y = y + v * math.sin(yaw) * DT
        yaw = yaw + v / WB * math.tan(delta) * DT
        v = v + a * DT
        
        return [x, y, yaw, v]


    
    def predict_motion(self, state, oa, od, xref):
        
        xbar = xref * 0.0
        [x, y, yaw, v] = state
        T = 5
        for i, _ in enumerate(state):
            xbar[i, 0] = state[i]

        for (ai, di, i) in zip(oa, od, range(1, T + 1)):
            state = self.update_state(state, ai, di)
            xbar[0, i] = x
            xbar[1, i] = y
            xbar[2, i] = v
            xbar[3, i] = yaw

        return xbar    
        
    
    def linear_mpc_control(self, xref, xbar, state, dref):
        """
        linear mpc control
        xref: reference point
        xbar: operational point
        x0: initial state
        dref: reference steer angle
        """
        print('xbar is {}\n'.format(xbar))
        NX = 4
        NU = 2
        T = 5
        DT = 0.2
        MAX_SPEED = 50
        MIN_SPEED = 0        
        MAX_ACCEL = 1.0
        MAX_STEER = 1.0
        MAX_DSTEER = 1.0
        
        # mpc parameters
        R = np.diag([0.01, 0.01])  # input cost matrix
        Rd = np.diag([0.01, 1.0])  # input difference cost matrix
        Q = np.diag([1.0, 1.0, 0.5, 0.5])  # state cost matrix
        Qf = Q  # state final matrix
        
        x = cvxpy.Variable((NX, T + 1))
        u = cvxpy.Variable((NU, T))

        cost = 0.0
        constraints = []

        for t in range(T):
            cost += cvxpy.quad_form(u[:, t], R)

            if t != 0:
                cost += cvxpy.quad_form(xref[:, t] - x[:, t], Q)

            A, B, C = self.get_linear_model_matrix(
                    xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A * x[:, t] + B * u[:, t] + C]

            if t < (T - 1):
                cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
                constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <=
                            MAX_DSTEER * DT]

        cost += cvxpy.quad_form(xref[:, T] - x[:, T], Qf)

        constraints += [x[:, 0] == state]
        constraints += [x[2, :] <= MAX_SPEED]
        constraints += [x[2, :] >= MIN_SPEED]
        constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
        constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

        prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
        prob.solve(solver=cvxpy.ECOS, verbose=False)

        if prob.status == cvxpy.OPTIMAL or prob.status == cvxpy.OPTIMAL_INACCURATE:
            ox = self.get_nparray_from_matrix(x.value[0, :])
            oy = self.get_nparray_from_matrix(x.value[1, :])
            ov = self.get_nparray_from_matrix(x.value[2, :])
            oyaw = self.get_nparray_from_matrix(x.value[3, :])
            oa = self.get_nparray_from_matrix(u.value[0, :])
            odelta = self.get_nparray_from_matrix(u.value[1, :])

        else:
            print("Error: Cannot solve mpc..")
            oa, odelta, ox, oy, oyaw, ov = None, None, None, None, None, None

        return oa, odelta, ox, oy, oyaw, ov
        
        
    
    def iterative_linear_mpc_control(self, xref, state, dref, oa, od):
        """
        MPC contorl with updating operational point iteraitvely
        """
        DU_TH = 0.1  # iteration finish param
        MAX_ITER = 3
        T = 5
        
        if oa is None or od is None:
            oa = [0.0] * T
            od = [0.0] * T

        for i in range(MAX_ITER):
            xbar = self.predict_motion(state, oa, od, xref)
            poa, pod = oa[:], od[:]
            
            
            oa, od, ox, oy, oyaw, ov = self.linear_mpc_control(xref, xbar, state, dref)
            
            
            print('poa is {}\n'.format(poa))
            print('pod is {}\n'.format(pod))
            du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
            if du <= DU_TH:
                break
            else:
                print("Iterative is max iter")

        return oa, od, ox, oy, oyaw, ov
        
        
    
    def update_controls(self):
        ######################################################
        # RETRIEVE SIMULATOR FEEDBACK
        ######################################################
        x               = self._current_x
        y               = self._current_y
        yaw             = self._current_yaw
        v               = self._current_speed
        self.update_desired_speed()
        v_desired       = self._desired_speed
        t               = self._current_timestamp
        waypoints       = self._waypoints
        throttle_output = 0
        steer_output    = 0
        brake_output    = 0

        # Skip the first frame to store previous values properly
        if self._start_control_loop:
            """
                Controller iteration code block.

                Controller Feedback Variables:
                    x               : Current X position (meters)
                    y               : Current Y position (meters)
                    yaw             : Current yaw pose (radians)
                    v               : Current forward speed (meters per second)
                    t               : Current time (seconds)
                    v_desired       : Current desired speed (meters per second)
                                      (Computed as the speed to track at the
                                      closest waypoint to the vehicle.)
                    waypoints       : Current waypoints to track
                                      (Includes speed to track at each x,y
                                      location.)
                                      Format: [[x0, y0, v0],
                                               [x1, y1, v1],
                                               ...
                                               [xn, yn, vn]]
                                      Example:
                                          waypoints[2][1]: 
                                          Returns the 3rd waypoint's y position

                                          waypoints[5]:
                                          Returns [x5, y5, v5] (6th waypoint)
                
                Controller Output Variables:
                    throttle_output : Throttle output (0 to 1)
                    steer_output    : Steer output (-1.22 rad to 1.22 rad)
                    brake_output    : Brake output (0 to 1)
            """
            
            
            self.vars.create_var('v_previous', 0.0)
            self.vars.v_previous = v
            
            """Get points"""
            T = 5 # Model Predictive Control Time Zone             
            ax = []
            ay = []
            sp = []
            for i in range(T):
                ax.append(waypoints[i][0])
                ay.append(waypoints[i][1])
                sp.append(waypoints[i][2]) #speed profile
            dl = 1.0    # course tick    
            cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(
                ax, ay, ds=dl)
                
            state = [x, y , yaw, v]
            target_ind, _ = self.calc_nearest_index(state, cx, cy, cyaw, 0)
            
            print('target_ind is {}'.format(target_ind) )
            
            oa = [0.0] * T
            odelta = [0.0] * T
            
            print('cyaw is {}'.format(cyaw) )
            cyaw = self.smooth_yaw(cyaw)
            
            print('cyaw is {}'.format(cyaw) )
             print('cyaw is {}'.format(cyaw) )
            
            
            dl = 1.0
            xref, target_ind, dref = self.calc_ref_trajectory(
                state, cx, cy, cyaw, ck, sp, dl, target_ind)
                
            oa, odelta, ox, oy, oyaw, ov = self.iterative_linear_mpc_control(
                xref, state, dref, oa, odelta)
                
            
            
            """
            K_p = 0.5
            K_d = 0.3
            K_i = 0.2
            dt = 0.1
            print("timestep is {}".format(t))
             
            a = K_p * (v_desired - self.vars.v_previous)
            """
            if oa > 0:
                throttle_output = np.abs(oa)
                brake_output    = 0
            else:
                throttle_output = 0
                brake_output    = np.abs(oa)
            steer_output = odelta
            
                
            print("v_desired is {}".format(v_desired))
            print("v_previous is {}".format(self.vars.v_previous))
            
        
            ######################################################
            # SET CONTROLS OUTPUT
            ######################################################
            self.set_throttle(throttle_output)  # in percent (0 to 1)
            self.set_steer(steer_output)        # in rad (-1.22 to 1.22)
            self.set_brake(brake_output)        # in percent (0 to 1)

        ######################################################
        ######################################################
        # MODULE 7: STORE OLD VALUES HERE (ADD MORE IF NECESSARY)
        ######################################################
        ######################################################
        """
            Use this block to store old values (for example, we can store the
            current x, y, and yaw values here using persistent variables for use
            in the next iteration)
        """
        self.vars.v_previous = v  # Store forward speed to be used in next step
