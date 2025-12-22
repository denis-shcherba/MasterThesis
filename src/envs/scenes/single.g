Include: <$RAI_PATH/scenarios/pandaSingle.g>

Edit cameraWrist { Q: [-0.0239713, 0.0481723, 0.16886, 0.39354, 0.00971287, -0.00283292, -0.919252], zRange: [.01, 10]}
Edit l_panda_finger_joint1: { limits: [.0, .0] }


cameraStaticTableTop(world): {
 Q: "t(0 .56 1.57) d(180 0 1 0)",
 shape: camera, size: [.1],
 focalLength: 0.895, width: 640, height: 360, zRange: [.01, 10]
}

# [-0.03767177  0.9016358   1.28276813 -0.00540696  0.00881296  0.85232008 -0.52291833]
cameraStaticTableTripod(world): {
 Q: "t(0 .75 1.28) d(135 1 0 0) d(180 0 0 1)",
 shape: camera, size: [.1],
 focalLength: 0.895, width: 640, height: 360, zRange: [.01, 10]
}

cameraShelf(world): {
 shape: camera, size: [.1],
 focalLength: 1, width: 224, height: 224, zRange: [.01, 10]
}
