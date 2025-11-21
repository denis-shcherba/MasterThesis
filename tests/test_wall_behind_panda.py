import robotic as ry

C = ry.Config()
C.addFile("$RAI_PATH/scenarios/pandaSingle.g")

#C.setJointState([0.01078953, -1.0382526, 0.0044672, -2.32419086, 0.0131875, 1.69684752, -0.71352486])
C.addFrame("wall_behind_panda").setPosition([0, -.515, 1.4]).setShape(ry.ST.box, [2.5, 0.03, 1.5])
C.view(True)
bot = ry.BotOp(C, True)
bot.sync(C)
print(bot.get_q())

# [ 0.01078953 -1.0382526   0.0044672  -2.32419086  0.0131875   1.69684752 -0.71352486]