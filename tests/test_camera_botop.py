import robotic as ry
import numpy as np
from MasterThesis.shelf import generate_shelf

print('robotic version:', ry.__version__, ry.compiled())

CAMERA = "cameraStatic"  # or "cameraWrist" 


C = ry.Config()
C.addFile(ry.raiPath('../rai-robotModels/scenarios/pandaFloatingFixGripper.g'))
pos = np.array([1., 0., .3])
generate_shelf(C, pos, base_quaternion=[1, 0, 0, 1], openings_small=[4, 11], equidistant=False)
C.view(False, 'this is your workspace data structure C -- NOT THE SIMULTATION')

pcl = C.addFrame('pcl', CAMERA)
C.addFrame("cameraWP", CAMERA).setShape(ry.ST.marker, [.1]) 

C.view(True)

bot = ry.BotOp(C, useRealRobot=False)

q = bot.get_qHome()
q[1] = q[1] + .2


pcl = C.getFrame("pcl")
pcl.setShape(ry.ST.pointCloud, [2]) # the size here is pixel size for display
bot.sync(C)

count = 0

while bot.getKeyPressed()!=ord('q'):
    image, depth, points = bot.getImageDepthPcl(CAMERA)
    pcl.setPointCloud(points, image)
    pcl.setColor([1,0,0])
    bot.sync(C, .1)
    
    if bot.getTimeToEnd()<=0.:
        bot.moveTo(q)
        bot.moveTo(bot.get_qHome())
        count = count + 1
    if count>=100:
        break

