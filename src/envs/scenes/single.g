Include: <$RAI_PATH/scenarios/pandaSingle.g>

Edit cameraWrist { Q: [-0.0239713, 0.0481723, 0.16886, 0.39354, 0.00971287, -0.00283292, -0.919252], zRange: [.01, 10]}
Edit l_panda_finger_joint1: { limits: [.0, .0] }


cameraStaticTable(world): {
 Q: "t(0 .56 1.57) d(180 0 1 0)",
 shape: camera, size: [.1],
 focalLength: 0.895, width: 640, height: 360, zRange: [.01, 10]
}

