import robotic as ry
import numpy as np

C = ry.Config()

for i in range(10):  
    C.addFrame(f"target_book") \
            .setPosition([0, 0, 1]) \
            .setShape(ry.ST.ssBox, size=[.4, .2, .1, 0.005]) \
            .setColor(np.random.rand(3)) \
            .setContact(1) \
            .setMass(.1)
    C.view(True)
    C.delFrame()
    C(f"target_book")