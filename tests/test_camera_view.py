import robotic as ry

C = ry.Config()
C.addFile("$RAI_PATH/scenarios/pandaSingle.g")

C.viewer().setCamera(C.getFrame("cameraWrist"))
C.view(True)