import robotic as ry

C = ry.Config()
C.addFile("$RAI_PATH/scenarios/pandaSingle.g")

bot = ry.BotOp(C, True)
bot.sync(C, .1)

print("current q:", bot.get_q())
print("current l_gripper pos", C.eval(ry.FS.position, ["l_gripper"])[0])

