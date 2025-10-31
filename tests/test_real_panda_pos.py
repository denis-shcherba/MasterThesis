import robotic as ry

C = ry.Config()
C.addFile("$RAI_PATH/scenarios/pandaSingleThesis.g")

bot=ry.BotOp(C, useRealRobot=True)

bot.sync(C)

print(C.getFrame("l_gripper").getPosition())

bot.home(C)

