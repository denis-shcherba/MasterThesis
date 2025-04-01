import robotic as ry
from MasterThesis.high_level_methods import RobotEnviroment

C = ry.Config()
C.addFile(ry.raiPath('scenarios/pandaSingle.g'))

C.addFrame('way1') \
    .setShape(ry.ST.marker, [.1]) \
    .setPosition([.25,.1,1.]) \
    .setColor([1,.5,0]) 

C.addFrame('way2') \
    .setShape(ry.ST.marker, [.1]) \
    .setPosition([.05,.1,1.]) \
    .setColor([1,.5,0]) 

qHome = C.getJointState()

C.setJointState(qHome)
limits = C.getJointLimits()
verbose = 0

for i in range(20):

    RoboEnv = RobotEnviroment(C, verbose=verbose, sim=True)
    RoboEnv.move_to_point(C.getFrame("way1").getPosition(), relPos=[.1, 0, 0])
    RoboEnv.move_to_point(C.getFrame("way2").getPosition())

        