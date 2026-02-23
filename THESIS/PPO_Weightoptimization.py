# =========================================================
# PPO → EXACT MPC WEIGHT OPTIMIZATION (TIME-BASED, 10s)
# ENERGY = positive actuator work
# METRIC = Energy per meter (COM z)
# =========================================================

import numpy as np
import mujoco
import casadi as ca
import gymnasium as gym
from gymnasium import spaces
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# =========================================================
# USER SETTINGS
# =========================================================
XML_PATH = "robot.xml"
N = 20
MAX_SIM_TIME = 10.0
SEED = 7
TOTAL_EPISODES = 200

THETA_UP, THETA_DOWN = 0.54, 0.24
MIN_HEIGHT_OK = 0.5
LOW_HEIGHT_PENALTY = 200.0

# log-scale weight parameterization
LOG_CENTER = np.log([50.0, 1.2, 35.5, 2.5])
LOG_SCALE  = np.ones(4)

W_MIN = np.array([10.0, 0.1, 5.0, 0.1])
W_MAX = np.array([300.0, 20.0, 300.0, 200.0])

def action_to_weights(a):
    w = np.exp(LOG_CENTER + np.clip(a, -1, 1) * LOG_SCALE)
    return tuple(np.clip(w, W_MIN, W_MAX))

# =========================================================
# NMPC
# =========================================================
def build_nmpc(dt, N):

    U = ca.SX.sym("U", 2*N)
    x0     = ca.SX.sym("x0", 2)
    th_ref = ca.SX.sym("th_ref")
    u_prev = ca.SX.sym("u_prev", 2)

    w_th  = ca.SX.sym("w_th")
    w_u   = ca.SX.sym("w_u")
    w_du  = ca.SX.sym("w_du")
    w_tv  = ca.SX.sym("w_tv")

    Ieff = 0.1
    b    = 20
    mgl  = 0.3
    GEAR = 100.0

    cost = 0
    th, thd = x0[0], x0[1]
    uL_prev, uR_prev = u_prev[0], u_prev[1]

    for k in range(N):
        uL = U[2*k]
        uR = U[2*k+1]

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
        {
            "x": U,
            "f": cost,
            "p": ca.vertcat(x0, th_ref, u_prev, w_th, w_u, w_du, w_tv)
        },
        {"ipopt.print_level": 0, "print_time": 0}
    )

# =========================================================
# SINGLE PPO EPISODE (TIME-BASED)
# =========================================================
def run_episode(W, solver):

    W_TH, W_U, W_DU, W_TV = W

    model = mujoco.MjModel.from_xml_path(XML_PATH)
    data  = mujoco.MjData(model)
    dt = float(model.opt.timestep)

    # IDs
    ls = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "left_tip")
    rs = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "right_tip")

    jidR = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_R")
    jidL = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hip_L")

    qposR, qvelR = int(model.jnt_qposadr[jidR]), int(model.jnt_dofadr[jidR])
    qposL, qvelL = int(model.jnt_qposadr[jidL]), int(model.jnt_dofadr[jidL])

    ACT_L, ACT_R = 0, 1

    # Adhesion
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
            int(model.site_bodyid[site_id]), data.qfrc_applied
        )

    def project_to_wall(xyz):
        return np.array([1.0, xyz[1], xyz[2]])

    anchor_L = np.array([1.0, 0.0, 0.9])
    anchor_R = np.array([1.0, 0.0, 1.1])

    # MPC buffers
    u_prev = np.zeros(2)
    u_sol  = np.zeros(2*N)
    u_mag_ref = np.zeros(2)

    phase = 1
    prev_left_on = prev_right_on = True

    # Metrics
    E_act = 0.0
    z0 = None

    def com_z():
        return float(data.subtree_com[0][2])

    # -----------------------------------------------------
    # 10 SECOND TIME-BASED SIMULATION
    # -----------------------------------------------------
    while data.time < MAX_SIM_TIME:

        data.ctrl[:] = 0.0
        data.qfrc_applied[:] = 0.0

        if phase == 1:
            apply_soft_weld(ls, anchor_L, 0.5)
            apply_soft_weld(rs, anchor_R, 0.5)
            phase = 2

        elif phase == 2:
            apply_soft_weld(ls, anchor_L, 0.5)
            apply_soft_weld(rs, anchor_R, 0.05)

            theta = float(data.qpos[qposR])
            theta_dot = float(data.qvel[qvelR])

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

            theta = float(data.qpos[qposL])
            theta_dot = float(data.qvel[qvelL])

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

        # ---- ENERGY + HEIGHT ----
        if z0 is None:
            z0 = com_z()

        P_act = float(np.dot(data.qvel, data.qfrc_actuator))
        E_act += max(0.0, P_act) * dt

    dz = max(com_z() - z0, 1e-6)
    return E_act / dz, dz, data.time, False

# =========================================================
# PPO ENV
# =========================================================
class PPOWeightEnv(gym.Env):

    def __init__(self, solver):
        self.solver = solver
        self.action_space = spaces.Box(-1, 1, (4,), np.float32)
        self.observation_space = spaces.Box(0, 1, (1,), np.float32)
        self.bestE = 1e9
        self.bestW = None

    def reset(self, seed=None, options=None):
        return np.zeros(1, np.float32), {}

    def step(self, action):
        W = action_to_weights(action)
        Epm, h, _, _ = run_episode(W, self.solver)

        reward = -Epm
        if h < MIN_HEIGHT_OK:
            reward -= LOW_HEIGHT_PENALTY * (MIN_HEIGHT_OK - h)

        if Epm < self.bestE and h > 1e-3:
            self.bestE = Epm
            self.bestW = W
            print(f"[NEW BEST] E/m={Epm:.3f}, h={h:.3f}, W={W}")

        return np.zeros(1, np.float32), reward, True, False, {}

# =========================================================
# TRAIN
# =========================================================
if __name__ == "__main__":

    np.random.seed(SEED)
    dt = mujoco.MjModel.from_xml_path(XML_PATH).opt.timestep
    solver = build_nmpc(dt, N)

    env = DummyVecEnv([lambda: PPOWeightEnv(solver)])

    model = PPO(
        "MlpPolicy",
        env,
        n_steps=10,
        batch_size=10,
        gamma=1.0,
        ent_coef=0.02,
        verbose=1,
        seed=SEED
    )

    model.learn(total_timesteps=TOTAL_EPISODES)

    best_env = env.envs[0]
    print("\n========== BEST WEIGHTS (PPO) ==========")
    print("W_TH =", best_env.bestW[0])
    print("W_U  =", best_env.bestW[1])
    print("W_DU =", best_env.bestW[2])
    print("W_TV =", best_env.bestW[3])
    print("=======================================\n")
