#!/usr/bin/env python3
"""Convert the upstream SOLO-12 URDF (example-robot-data) into the MJCF models used here.

    python sim/build_model.py

Inputs : assets/solo12/urdf/solo12_upstream.urdf + assets/solo12/meshes/*.stl (unmodified upstream)
Outputs: assets/solo12/solo12.xml        robot + floor, visual meshes (rendering, evaluation)
         assets/solo12/solo12_train.xml  same physics, no visual meshes (training, smaller in RAM)

What is added on top of the URDF (the URDF has no actuators and effort limits of 1000):
  * a free joint on base_link, a floor plane;
  * primitive collision geoms (foot spheres, shin and thigh capsules, base box) instead of the
    STL collision meshes, no self collision;
  * joint armature / damping / friction loss and joint ranges;
  * position actuators = joint PD (kp = 4 Nm/rad, kd = 0.2 Nm s/rad), torque limit 2.7 Nm
    (SOLO-12 actuators: 9:1 belt reduction, about 2.7 Nm peak);
  * sites + sensors read by the vectorised environment (foot positions/velocities, foot touch,
    knee/base touch, actuator torques).
"""
import os
import re
import numpy as np
import mujoco

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.abspath(os.path.join(HERE, "..", "assets", "solo12"))
URDF = os.path.join(ASSETS, "urdf", "solo12_upstream.urdf")

LEGS = ["FL", "FR", "HL", "HR"]
JOINTS = [f"{l}_{j}" for l in LEGS for j in ("HAA", "HFE", "KFE")]

# physical parameters (documented in README)
KP, KD, TAU_MAX = 4.0, 0.2, 2.7
ARMATURE, DAMPING, FRICTIONLOSS = 3.0e-4, 0.01, 0.01
RANGES = {"HAA": (-1.0, 1.0), "HFE": (-2.6, 2.6), "KFE": (-3.0, 3.0)}
TIMESTEP = 0.002


def urdf_to_spec():
    src = open(URDF).read()
    src = re.sub(r"package://example-robot-data/robots/solo_description/meshes/stl/(solo12|with_foot)/", "", src)
    src = re.sub(r"(<robot[^>]*>)", r'\1<mujoco><compiler meshdir="%s" discardvisual="false" '
                 r'fusestatic="false" balanceinertia="true" strippath="true"/></mujoco>' % os.path.join(ASSETS, "meshes"),
                 src, count=1)
    return mujoco.MjSpec.from_string(src)


def build(with_visual=True):
    spec = urdf_to_spec()
    spec.modelname = "solo12"
    spec.option.timestep = TIMESTEP
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    spec.option.iterations = 20
    spec.option.ls_iterations = 10
    spec.compiler.degree = False

    # ---- geoms: drop URDF collision meshes, keep visual meshes in group 2 ------------------------
    for g in list(spec.geoms):
        if g.contype != 0 or g.conaffinity != 0 or not with_visual:
            spec.delete(g)
        else:
            g.group = 2
            g.rgba = [0.75, 0.75, 0.78, 1.0]
    if not with_visual:
        for msh in list(spec.meshes):
            spec.delete(msh)

    def coll(body, **kw):
        g = body.add_geom(**kw)
        g.contype, g.conaffinity, g.condim, g.group = 1, 0, 3, 3
        g.friction = [1.0, 0.005, 0.0001]
        g.rgba = [0.9, 0.3, 0.2, 0.4]
        return g

    base = spec.body("base_link")
    base.add_freejoint(name="root")
    coll(base, name="base_box", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.2, 0.075, 0.03])
    base.add_site(name="imu", pos=[0, 0, 0], size=[0.01, 0.01, 0.01])
    s = base.add_site(name="base_touch", type=mujoco.mjtGeom.mjGEOM_BOX, size=[0.215, 0.09, 0.045])
    s.rgba = [0, 0, 0, 0]
    for l in LEGS:
        side = 1.0 if l[1] == "L" else -1.0
        up = spec.body(f"{l}_UPPER_LEG")
        coll(up, name=f"{l}_thigh", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.016, 0, 0],
             fromto=[0, side * 0.0197, -0.01, 0, side * 0.0300, -0.15])
        lo = spec.body(f"{l}_LOWER_LEG")
        coll(lo, name=f"{l}_shin", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[0.011, 0, 0],
             fromto=[0, side * 0.0073, -0.005, 0, side * 0.008, -0.125])
        s = lo.add_site(name=f"{l}_knee", pos=[0, side * 0.007, 0], size=[0.03, 0, 0])
        s.rgba = [0, 0, 0, 0]
        ft = spec.body(f"{l}_FOOT")
        coll(ft, name=f"{l}_foot", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[0.016, 0, 0])
        s = ft.add_site(name=f"{l}_foot", pos=[0, 0, 0], size=[0.021, 0, 0])
        s.rgba = [0, 0, 0, 0]

    # ---- joints and actuators -------------------------------------------------------------------
    for jn in JOINTS:
        j = spec.joint(jn)
        j.armature, j.frictionloss = ARMATURE, FRICTIONLOSS
        j.damping = [DAMPING, 0.0, 0.0]
        j.limited = mujoco.mjtLimited.mjLIMITED_TRUE
        j.range = list(RANGES[jn.split("_")[1]])
    for jn in JOINTS:
        a = spec.add_actuator(name=jn, target=jn, trntype=mujoco.mjtTrn.mjTRN_JOINT)
        a.set_to_position(kp=KP, kv=KD)
        a.forcelimited = mujoco.mjtLimited.mjLIMITED_TRUE
        a.forcerange = [-TAU_MAX, TAU_MAX]
        a.ctrllimited = mujoco.mjtLimited.mjLIMITED_FALSE

    # ---- sensors (order matters: the environment slices sensordata by name) ---------------------
    S = mujoco.mjtSensor
    for l in LEGS:
        spec.add_sensor(name=f"{l}_foot_pos", type=S.mjSENS_FRAMEPOS, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{l}_foot")
    for l in LEGS:
        spec.add_sensor(name=f"{l}_foot_vel", type=S.mjSENS_FRAMELINVEL, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{l}_foot")
    for l in LEGS:
        spec.add_sensor(name=f"{l}_foot_touch", type=S.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{l}_foot")
    for l in LEGS:
        spec.add_sensor(name=f"{l}_knee_touch", type=S.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE, objname=f"{l}_knee")
    spec.add_sensor(name="base_touch", type=S.mjSENS_TOUCH, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="base_touch")
    for jn in JOINTS:
        spec.add_sensor(name=f"{jn}_tau", type=S.mjSENS_ACTUATORFRC, objtype=mujoco.mjtObj.mjOBJ_ACTUATOR, objname=jn)
    spec.add_sensor(name="imu_gyro", type=S.mjSENS_GYRO, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")
    spec.add_sensor(name="base_linvel", type=S.mjSENS_VELOCIMETER, objtype=mujoco.mjtObj.mjOBJ_SITE, objname="imu")

    # ---- world: floor, light, camera ------------------------------------------------------------
    tex = spec.add_texture(name="grid", type=mujoco.mjtTexture.mjTEXTURE_2D, builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER,
                           rgb1=[0.82, 0.84, 0.86], rgb2=[0.62, 0.65, 0.68], width=512, height=512)
    mat = spec.add_material(name="grid", texrepeat=[4, 4], texuniform=True, reflectance=0.05)
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "grid"
    sky = spec.add_texture(name="sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX, builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT,
                           rgb1=[0.95, 0.96, 0.98], rgb2=[0.75, 0.80, 0.86], width=256, height=256)
    floor = spec.worldbody.add_geom(name="floor", type=mujoco.mjtGeom.mjGEOM_PLANE, size=[0, 0, 0.05], material="grid")
    floor.contype, floor.conaffinity, floor.condim = 0, 1, 3
    floor.friction = [1.0, 0.005, 0.0001]
    lt = spec.worldbody.add_light(name="sun", pos=[0, 0, 3], dir=[0, 0, -1])
    lt.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    lt.diffuse = [0.7, 0.7, 0.7]
    spec.visual.headlight.ambient = [0.35, 0.35, 0.35]
    spec.visual.global_.offwidth = 1280
    spec.visual.global_.offheight = 720

    # joint locks used to simulate a broken leg that is stuck in a fixed pose (inactive by default;
    # the harness switches them on with data.eq_active). One equality per joint: q = polycoef[0].
    for jn in JOINTS:
        e = spec.add_equality(name=f"lock_{jn}", type=mujoco.mjtEq.mjEQ_JOINT, objtype=mujoco.mjtObj.mjOBJ_JOINT,
                              name1=jn)
        e.active = False
        e.data[:5] = [0.0, 0.0, 0.0, 0.0, 0.0]

    # stand pose keyframe (example-robot-data "standing": HAA +-0.1, HFE +-0.8, KFE -+1.6)
    q0 = default_pose()
    k = spec.add_key(name="stand")
    k.qpos = np.concatenate([[0, 0, 0.245, 1, 0, 0, 0], q0]).tolist()
    k.ctrl = q0.tolist()
    return spec


def default_pose():
    q = []
    for l in LEGS:
        haa = 0.1 if l[1] == "L" else -0.1
        hfe, kfe = (0.8, -1.6) if l[0] == "F" else (-0.8, 1.6)
        q += [haa, hfe, kfe]
    return np.array(q)


def main():
    for name, vis in (("solo12.xml", True), ("solo12_train.xml", False)):
        spec = build(vis)
        m = spec.compile()
        xml = spec.to_xml()
        # absolute mesh directory -> relative, so the file is portable
        xml = xml.replace(os.path.join(ASSETS, "meshes"), "meshes")
        open(os.path.join(ASSETS, name), "w").write(xml)
        print(f"{name}: nq={m.nq} nv={m.nv} nu={m.nu} ngeom={m.ngeom} nsensordata={m.nsensordata} "
              f"mass={m.body_mass.sum():.3f} kg")


if __name__ == "__main__":
    main()
