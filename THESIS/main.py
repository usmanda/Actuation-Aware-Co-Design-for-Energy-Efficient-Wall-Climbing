import mujoco
import mujoco.viewer
import numpy as np
import time
import casadi as ca
import matplotlib.pyplot as plt

# =========================================================
# LOAD MODEL
# =========================================================
model = mujoco.MjModel.from_xml_path("robot.xml")
data  = mujoco.MjData(model)
dt = float(model.opt.timestep)

g = 9.81
SIM_TIME_LIMIT = 10.0   # <<< FIXED 10 SECONDS SIMULATION

# =========================================================
# IDs
# =========================================================
ls = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_tip")
rs = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_tip")

jid_hipR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_R")
jid_hipL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_L")

qpos_hipR = int(model.jnt_qposadr[jid_hipR])
qvel_hipR = int(model.jnt_dofadr[jid_hipR])
qpos_hipL = int(model.jnt_qposadr[jid_hipL])
qvel_hipL = int(model.jnt_dofadr[jid_hipL])

ACT_L, ACT_R = 0, 1

# =========================================================
# ADHESION MODEL
# =========================================================
k, d = 6000.0, 350.0

def site_velocity(site_id):
    v = np.zeros(6)
    mujoco.mj_objectVelocity(model, data, mujoco.mjtObj.mjOBJ_SITE, site_id, v, 0)
    return v[:3]

def apply_soft_weld(site_id, anchor, alpha):
    pos = data.site_xpos[site_id]
    vel = site_velocity(site_id)
    F = alpha * (k * (anchor - pos) - d * vel)
    mujoco.mj_applyFT(
        model, data, F, np.zeros(3), pos,
        int(model.site_bodyid[site_id]),
        data.qfrc_applied
    )

def project_to_wall(xyz, wall_x=1.0):
    return np.array([wall_x, xyz[1], xyz[2]])

anchor_L = np.array([1.0, 0.0, 0.9])
anchor_R = np.array([1.0, 0.0, 1.1])

# =========================================================
# NMPC
# =========================================================
def build_nmpc(dt, N):
    U = ca.SX.sym("U", 2*N)
    x0 = ca.SX.sym("x0", 2)
    th_ref = ca.SX.sym("th_ref")
    u_prev = ca.SX.sym("u_prev", 2)

    w_th = ca.SX.sym("w_th")
    w_u  = ca.SX.sym("w_u")
    w_du = ca.SX.sym("w_du")
    w_tv = ca.SX.sym("w_tv")

    Ieff = 0.1
    b    = 20
    mgl  = 0.3
    GEAR = 70.0

    cost = 0
    th, thd = x0[0], x0[1]
    uL_prev, uR_prev = u_prev[0], u_prev[1]

    for k_ in range(N):
        uL = U[2*k_]
        uR = U[2*k_+1]
        tau = GEAR * (uL - uR)

        thdd = (tau - b*thd - mgl*ca.sin(th)) / Ieff
        thd += dt * thdd
        th  += dt * thd

        cost += (
            w_th * (th - th_ref)**2 +
            w_u  * (uL**2 + uR**2) +
            w_du * ((uL - uL_prev)**2 + (uR - uR_prev)**2)
        )
        uL_prev, uR_prev = uL, uR

    cost += w_tv * ca.fmax(0, ca.fabs(thd) - 0.5)**2

    return ca.nlpsol(
        "solver", "ipopt",
        {"x": U, "f": cost,
         "p": ca.vertcat(x0, th_ref, u_prev, w_th, w_u, w_du, w_tv)},
        {"ipopt.print_level": 0, "print_time": 0}
    )

# =========================================================
# MPC PARAMETERS
# =========================================================
N = 20
solver = build_nmpc(dt, N)

W_TH, W_U, W_DU, W_TV = 18.39, 3.26, 96.49, 0.919
THETA_UP, THETA_DOWN = 0.37, 0.24

u_prev = np.zeros(2)
u_sol  = np.zeros(2*N)
u_mag_ref = np.zeros(2)

phase = 1
prev_left_on = prev_right_on = True

# =========================================================
# METRICS
# =========================================================
# torso slide joints are first ones in model
jid_slide_x = 0
qpos_slide_x = int(model.jnt_qposadr[jid_slide_x])
torso_x_log = []
t_log, zcom_log = [], []
E_act_log = []
E_act_cum = 0.0
PE_log = []

m_total = float(np.sum(model.body_mass))
zcom0, PE0 = None, None

def com_z():
    return float(data.subtree_com[0][2])

# =========================================================
# SIMULATION (TIME-BASED)
# =========================================================
with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():

        if data.time >= SIM_TIME_LIMIT:
            break

        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0

        # FSM (unchanged)
        if phase == 1:
            apply_soft_weld(ls, anchor_L, 0.5)
            apply_soft_weld(rs, anchor_R, 0.5)
            phase = 2

        elif phase == 2:
            apply_soft_weld(ls, anchor_L, 0.5)
            apply_soft_weld(rs, anchor_R, 0.05)

            theta = float(data.qpos[qpos_hipR])
            theta_dot = float(data.qvel[qvel_hipR])

            sol = solver(
                x0=u_sol, lbx=-1, ubx=1,
                p=[theta, theta_dot, THETA_UP,
                   u_prev[0], u_prev[1],
                   W_TH, W_U, W_DU, W_TV]
            )
            u_sol = np.array(sol["x"]).flatten()
            u_prev = u_sol[:2]
            data.ctrl[:] = u_prev

            if abs(theta) >= THETA_UP:
                u_mag_ref[:] = np.abs(u_prev)
                phase, prev_right_on = 3, False

        elif phase == 3:
            if not prev_right_on:
                anchor_R = project_to_wall(data.site_xpos[rs])
                prev_right_on = True
            apply_soft_weld(ls, anchor_L, 0.5)
            apply_soft_weld(rs, anchor_R, 0.5)
            phase = 4

        elif phase == 4:
            apply_soft_weld(rs, anchor_R, 0.5)
            apply_soft_weld(ls, anchor_L, 0.05)

            theta = float(data.qpos[qpos_hipL])
            theta_dot = float(data.qvel[qvel_hipL])

            sol = solver(
                x0=u_sol, lbx=-1, ubx=1,
                p=[theta, theta_dot, -THETA_DOWN,
                   u_prev[0], u_prev[1],
                   W_TH, W_U, W_DU, W_TV]
            )
            u_sol = np.array(sol["x"]).flatten()
            u_prev = u_sol[:2]

            data.ctrl[ACT_L] = np.sign(u_prev[0]) * u_mag_ref[0]
            data.ctrl[ACT_R] = np.sign(u_prev[1]) * u_mag_ref[1]

            if abs(theta) <= THETA_DOWN:
                phase, prev_left_on = 5, False

        elif phase == 5:
            if not prev_left_on:
                anchor_L = project_to_wall(data.site_xpos[ls])
                prev_left_on = True
            apply_soft_weld(ls, anchor_L, 0.5)
            apply_soft_weld(rs, anchor_R, 0.5)
            phase = 1
            u_prev[:] = 0
            u_sol[:] = 0

        mujoco.mj_step(model, data)
        viewer.sync()

        # -------- METRICS --------
        zc = com_z()
        if zcom0 is None:
            zcom0 = zc
            PE0 = m_total * g * zcom0

        P_act = float(np.dot(data.qvel, data.qfrc_actuator))
        dE = max(0.0, P_act) * dt
        E_act_cum += dE

        t_log.append(data.time)
        zcom_log.append(zc)
        E_act_log.append(E_act_cum)
        PE_log.append(m_total * g * zc - PE0)
        torso_x_log.append(float(data.qpos[qpos_slide_x]))
        time.sleep(dt)

# =========================================================
# FINAL METRICS
# =========================================================
total_height = zcom_log[-1] - zcom_log[0]
total_time = t_log[-1]
total_energy = E_act_log[-1]
energy_per_meter = total_energy / total_height
theta_log = []

print("\n========== 10s TIME-BASED PERFORMANCE ==========")
print(f"Total Mass                  : {m_total:.3f} kg")
print(f"Height Climbed (COM z)       : {total_height:.3f} m")
print(f"Total Time                  : {total_time:.3f} s")
print(f"Average Speed               : {total_height/total_time:.3f} m/s")
print("------------------------------------------------")
print(f"Total Actuator Energy       : {total_energy:.3f} J")
print(f"Energy per Meter            : {energy_per_meter:.3f} J/m")
print(f"Average Actuator Power      : {total_energy/total_time:.3f} W")
print(f"Potential Energy Increase   : {PE_log[-1]:.3f} J")
print("================================================\n")

