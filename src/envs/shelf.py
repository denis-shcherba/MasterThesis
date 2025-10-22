import numpy as np
import robotic as ry

def generate_shelf(C: ry.Config, pos: np.ndarray, openings_small: list[int]=[4, 6], small_opening_dims: list[float]=[.21, .21, .21],  just_front: bool=False, base_quaternion: list[float]=[1,0,0,0], shelf_lip: bool=False, equidistant=True):
    # TODO: More efficient piece building, don't repeat pieces!
    inner_wall_width = .02

    w = small_opening_dims[0]*openings_small[0]
    d = w
    h = small_opening_dims[1]*openings_small[1]

    base_height = .05

    C.addFrame("shelf_base") \
        .setPosition(pos + np.array([0., 0., base_height*.5])) \
        .setShape(ry.ST.ssBox, size=[w, d, base_height, 0.005]) \
        .setColor([.8, .8, .8]) \
        .setContact(1) \
        .setQuaternion(base_quaternion)

    if not just_front:
        C.addFrame("shelf_middle", "shelf_base") \
            .setRelativePosition([0, 0, base_height*.5+h*.5]) \
            .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., inner_wall_width, h, 0.005]) \
            .setColor([1., 1., 0.])

    sides_count = 1 if just_front else 2
    for s in range(sides_count):
        p = d*.5-small_opening_dims[2]
        p *= 1. if s == 0 else -1.
        C.addFrame(f"shelf_back_{s}", "shelf_base") \
            .setRelativePosition([p, 0, base_height*.5+h*.5]) \
            .setShape(ry.ST.ssBox, size=[inner_wall_width, w, h, 0.005]) \
            .setColor([1., 1., 0.]) \
            .setContact(1)
        if equidistant:

            for j in range(openings_small[1]):
                for i in range(openings_small[0]):
                    p = small_opening_dims[2]*.5
                    p *= 1. if s == 0 else -1.
                    opening_pos = np.array([
                        p,
                        i*small_opening_dims[0] - openings_small[0]*small_opening_dims[0]*.5 + small_opening_dims[0]*.5,
                        j*small_opening_dims[1] - openings_small[1]*small_opening_dims[1]*.5 + small_opening_dims[1]*.5
                        ])
                    C.addFrame(f"small_box_left_{s}_{i}_{j}", f"shelf_back_{s}") \
                        .setRelativePosition(opening_pos - np.array([0., small_opening_dims[0]*.5, 0.])) \
                        .setShape(ry.ST.ssBox, size=[small_opening_dims[2], inner_wall_width, small_opening_dims[1], 0.005]) \
                        .setColor([1., 1., 0.]) \
                        .setContact(1)
                    C.addFrame(f"small_box_right_{s}_{i}_{j}", f"shelf_back_{s}") \
                        .setRelativePosition(opening_pos - np.array([0., -small_opening_dims[0]*.5, 0.])) \
                        .setShape(ry.ST.ssBox, size=[small_opening_dims[2], inner_wall_width, small_opening_dims[1], 0.005]) \
                        .setColor([1., 1., 0.]) \
                        .setContact(1)

                    C.addFrame(f"small_box_top_{s}_{i}_{j}", f"shelf_back_{s}") \
                        .setRelativePosition(opening_pos - np.array([0., 0., -small_opening_dims[1]*.5])) \
                        .setShape(ry.ST.ssBox, size=[small_opening_dims[2], small_opening_dims[0], inner_wall_width, 0.005]) \
                        .setColor([1., 1., 0.]) \
                        .setContact(1)

                    C.addFrame(f"small_box_bottom_{s}_{i}_{j}", f"shelf_back_{s}") \
                        .setRelativePosition(opening_pos - np.array([0., 0., small_opening_dims[1]*.5])) \
                        .setShape(ry.ST.ssBox, size=[small_opening_dims[2], small_opening_dims[0], inner_wall_width, 0.005]) \
                        .setColor([1., 1., 0.]) \
                        .setContact(1)
                    
                    p = -small_opening_dims[2]*.5
                    p *= 1. if s == 0 else -1.
                    if shelf_lip:
                        C.addFrame(f"small_box_blocker_{s}_{i}_{j}", f"shelf_back_{s}") \
                            .setRelativePosition(opening_pos - np.array([p, 0, small_opening_dims[1]*.5-.015])) \
                            .setShape(ry.ST.ssBox, size=[inner_wall_width, small_opening_dims[0], .03, 0.005]) \
                            .setColor([1., 1., 0.]) \
                            .setContact(1)
                    
                    C.addFrame(f"small_box_inside_{s}_{i}_{j}", f"shelf_back_{s}") \
                        .setRelativePosition(opening_pos) \
                        .setShape(ry.ST.ssBox, size=[small_opening_dims[2], small_opening_dims[0], small_opening_dims[1], 0.005]) \
                        .setColor([0., 0., 0., .2]) \
                        .setContact(0)
                
                if not just_front:
                        p = w*.25
                        p *= 1. if s == 0 else -1.
                        C.addFrame(f"big_box_bottom_{s}_{j}", "shelf_base") \
                            .setRelativePosition([0., p, base_height*.5 + j*small_opening_dims[1]]) \
                            .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., w*.5, inner_wall_width, 0.005]) \
                            .setColor([1., 1., 0.]) \
                            .setContact(1)
                        
                        C.addFrame(f"big_box_top_{s}_{j}", "shelf_base") \
                            .setRelativePosition([0., p, base_height*.5 + (j+1)*small_opening_dims[1]]) \
                            .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., w*.5, inner_wall_width, 0.005]) \
                            .setColor([1., 1., 0.]) \
                            .setContact(1)
                        
                        C.addFrame(f"big_box_inside_{s}_{j}", "shelf_base") \
                            .setRelativePosition([0., p, base_height*.5 + j*small_opening_dims[1] + small_opening_dims[1]*.5]) \
                            .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., w*.5, small_opening_dims[1], 0.005]) \
                            .setColor([0., 0., 0., .2]) \
                            .setContact(0)
                        
                        p = w*.5
                        p *= 1. if s == 0 else -1.
                        if shelf_lip:
                            C.addFrame(f"big_box_blocker_{s}_{j}", "shelf_base") \
                                .setRelativePosition([0., p, base_height*.5 + j*small_opening_dims[1]+.015]) \
                                .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., inner_wall_width, .03, 0.005]) \
                                .setColor([1., 1., 0.]) \
                                .setContact(1)
                
        else:
            p = w*.25
            p *= 1. if s == 0 else -1.
            
            floor_offsets = [.35, .43, .51, .18, .15, .2, .15, .15, .15, .12]  # Last 2 entries TODO


            opening_pos = np.array([
                        p,
                        1*small_opening_dims[0] - openings_small[0]*small_opening_dims[0]*.5 + small_opening_dims[0]*.5,
                        1*small_opening_dims[1] - openings_small[1]*small_opening_dims[1]*.5 + small_opening_dims[1]*.5
                        ])
            
            C.addFrame(f"small_box_right_{s}", "shelf_base") \
                    .setRelativePosition([w*.375, p*2 + d - small_opening_dims[2]*4, base_height + 1.04]) \
                    .setShape(ry.ST.ssBox, size=[small_opening_dims[2], inner_wall_width, small_opening_dims[1]*11, 0.005]) \
                    .setColor([1., 1., 0.]) \
                    .setContact(1)
            
            C.addFrame(f"small_box_left_{s}", "shelf_base") \
                    .setRelativePosition([-w*.375, p*2 + d - small_opening_dims[2]*4, base_height + 1.04]) \
                    .setShape(ry.ST.ssBox, size=[small_opening_dims[2], inner_wall_width, small_opening_dims[1]*11, 0.005]) \
                    .setColor([1., 1., 0.]) \
                    .setContact(1)

            for i, offset in enumerate(floor_offsets):
                
                if i == 0:
                    C.addFrame(f"big_xy_bottom_{s}_0", "shelf_base") \
                        .setRelativePosition([0., p, base_height + floor_offsets[0]]) \
                        .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., w*.5, inner_wall_width, 0.001]) \
                        .setColor([1., 1., 0.]) \
                        .setContact(1) \


                    C.addFrame(f"big_box_inside_{s}_0", f"big_xy_bottom_{s}_0") \
                        .setRelativePosition([0., 0., -floor_offsets[1]/2+.5*base_height]) \
                        .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., w*.5, floor_offsets[1]-base_height, 0.005]) \
                        .setColor([0., 1., 1., .0]) \
                        .setContact(0)

                    # Wände links (und später recht von der Regalwand) auskommentiert weil TODO, nicht wirklich nötig grade
                    # C.addFrame(f"small_box_right_{s}_0", "shelf_base") \
                    #     .setRelativePosition([w*.375, p*2 + d - small_opening_dims[2]*4, base_height + .35 - floor_offsets[0]/2]) \
                    #     .setShape(ry.ST.ssBox, size=[small_opening_dims[2], inner_wall_width, floor_offsets[0], 0.005]) \
                    #     .setColor([1., 1., 0.]) \
                    #     .setContact(1)
                    
            
                else:
                    C.addFrame(f"big_xy_bottom_{s}_{i}", f"big_xy_bottom_{s}_{i-1}") \
                        .setRelativePosition([0., 0, offset]) \
                        .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., w*.5, inner_wall_width, 0.001]) \
                        .setColor([1., 1., 0.]) \
                        .setContact(1) \
                        .setAttribute("friction", 1e-4) \
                    
                    if i==2:
                        C.addFrame(f"big_box_inside_{s}_{i}", f"big_xy_bottom_{s}_{i}") \
                            .setRelativePosition([0., inner_wall_width, -floor_offsets[i]/2]) \
                            .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., w*.5-inner_wall_width, floor_offsets[i]-base_height/2, 0.005]) \
                            .setColor([0., 1., 1., 0]) \
                            .setContact(0)
                    
                    
                    # Wände links (und später recht von der Regalwand) auskommentiert weil TODO, nicht wirklich nötig grade
                    # C.addFrame(f"small_box_right_{s}_{i}", f"small_box_right_{s}_{i-1}") \
                    #     .setRelativePosition([0, 0 , floor_offsets[i]]) \
                    #     .setShape(ry.ST.ssBox, size=[small_opening_dims[2], inner_wall_width, floor_offsets[i], 0.005]) \
                    #     .setColor([1., 1., 0.]) \
                    #     .setContact(1)
                    


            p = w*.5
            p *= 1. if s == 0 else -1.
            
            if shelf_lip:
                #TODO
                pass
            
    C.addFrame("cameraStatic").setShape(ry.ST.camera, size=[.1]) \
        .setPosition(C.getFrame("big_xy_bottom_0_1").getPosition()+np.array([-.22*w, 0, 0]) + np.array([-.25, 0, floor_offsets[2]])) \
        .setQuaternion([np.cos(np.deg2rad(125/2)), 0, np.sin(np.deg2rad(125/2)), 0]) \
        .setAttribute("focalLength", 1.5) \
        .setAttribute("width", 640) \
        .setAttribute("height", 360) 
    
    # if "l_gripper" in C.getFrameNames():
    #     C.delFrame("camerwaWrist")
    #     C.addFrame("cameraWrist", "l_panda_joint7").setShape(ry.ST.camera, size=[.1]) \
    #         .setRelativePose([-0.0239713, 0.0481723, 0.16886, 0.39354, 0.00971287, -0.00283292, -0.919252]) \
    #         # .setAttribute("width", 640) \
    #         # .setAttribute("height", 360) 

if __name__ == "__main__":
    C = ry.Config()
    pos = np.array([0., 0., 0.])
    generate_shelf(C, pos)
    C.view(True)
