import robotic as ry
from high_level_methods import RobotEnviroment

C = ry.Config()
C.addFile(ry.raiPath('scenarios/pandaSingle.g'))

C.addFrame('way1') \
    .setShape(ry.ST.marker, [.1]) \
    .setPosition([.3,.1,.7]) \
    .setColor([1,.5,0]) 

C.addFrame('way2') \
    .setShape(ry.ST.marker, [.1]) \
    .setPosition([.0,.1,.7]) \
    .setColor([1,.5,0]) \

# C.addFrame("box") \
#     .setPosition([.15, .1, .7]) \
#     .setShape(ry.ST.ssBox, size=[.1, .1, .1, 0.005]) \
#     .setColor([.8, .3, .6]) \
#     .setContact(1) \
#     .setMass(.1) 



qHome = C.getJointState()

C.setJointState(qHome)
limits = C.getJointLimits()
verbose = 0


RoboEnv = RobotEnviroment(C, verbose=verbose, sim=True)
RoboEnv.move_to_point(C.getFrame("way1").getPosition(),straight_line=False, accumulated_collisions = False)
RoboEnv.move_to_point(C.getFrame("way2").getPosition(), straight_line=True, accumulated_collisions = True)

        