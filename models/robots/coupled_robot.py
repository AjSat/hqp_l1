from pybullet_utils import bullet_client as bc
from pybullet_utils import urdfEditor as ed
import pybullet
import pybullet_data
import time

p0 = bc.BulletClient(connection_mode=pybullet.DIRECT)
p0.setAdditionalSearchPath(pybullet_data.getDataPath())

p1 = bc.BulletClient(connection_mode=pybullet.DIRECT)
p1.setAdditionalSearchPath(pybullet_data.getDataPath())

p3 = bc.BulletClient(connection_mode=pybullet.DIRECT)
p3.setAdditionalSearchPath(pybullet_data.getDataPath())

#can also connect using different modes, GUI, SHARED_MEMORY, TCP, UDP, SHARED_MEMORY_SERVER, GUI_SERVER

rot_table = p1.loadURDF("models/robots/rotating_torso.urdf", useFixedBase = True)
kuka = p0.loadURDF("kuka_iiwa/model.urdf")
kuka2 = p3.loadURDF("kuka_iiwa/model2.urdf")

ed0 = ed.UrdfEditor()
ed0.initializeFromBulletBody(rot_table, p1._client)
ed1 = ed.UrdfEditor()
ed1.initializeFromBulletBody(kuka, p0._client)
ed2 = ed.UrdfEditor()
ed2.initializeFromBulletBody(kuka2, p3._client)
#ed1.saveUrdf("combined.urdf")

parentLinkIndex = 1

jointPivotXYZInParent = [0, -0.15, 0]
jointPivotRPYInParent = [0, 0, 0]

jointPivotXYZInChild = [0, 0, 0]
jointPivotRPYInChild = [0, 0, 0]

newjoint = ed0.joinUrdf(ed1, parentLinkIndex, jointPivotXYZInParent, jointPivotRPYInParent,
                        jointPivotXYZInChild, jointPivotRPYInChild, p0._client, p1._client)
newjoint.joint_type = p0.JOINT_FIXED

parentLinkIndex = 1

jointPivotXYZInParent = [0, 0.15, 0]
jointPivotRPYInParent = [0, 0, 0]

jointPivotXYZInChild = [0, 0, 0]
jointPivotRPYInChild = [0, 0, 0]

newjoint = ed0.joinUrdf(ed2, parentLinkIndex, jointPivotXYZInParent, jointPivotRPYInParent,
                        jointPivotXYZInChild, jointPivotRPYInChild, p3._client, p1._client)
newjoint.joint_type = p0.JOINT_FIXED

ed0.saveUrdf("combined.urdf")

print(p0._client)
print(p1._client)
print("p0.getNumBodies()=", p0.getNumBodies())
print("p1.getNumBodies()=", p1.getNumBodies())

pgui = bc.BulletClient(connection_mode=pybullet.GUI)
pgui.configureDebugVisualizer(pgui.COV_ENABLE_RENDERING, 0)

orn = [0, 0, 0, 1]
ed0.createMultiBody([0, 0, 0], orn, pgui._client)
pgui.setRealTimeSimulation(1)

pgui.configureDebugVisualizer(pgui.COV_ENABLE_RENDERING, 1)
while (pgui.isConnected()):
  pgui.getCameraImage(320, 200, renderer=pgui.ER_BULLET_HARDWARE_OPENGL)
  time.sleep(1. / 240.)
