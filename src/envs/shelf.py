import numpy as np
import robotic as ry

def generate_shelf(C: ry.Config, pos: np.ndarray, openings_small: list[int]=[4, 6], small_opening_dims: list[float]=[.22, .22, .19],  just_front: bool=False, base_quaternion: list[float]=[1,0,0,0], shelf_lip: bool=False, equidistant=True, floor_offsets: list[float]=[0.35, 0.43, 0.30, 0.18, 0.15, 0.2, 0.15, 0.15, 0.15, 0.12]):
    # TODO: More efficient piece building, don't repeat pieces!
    inner_wall_width = .02

    w = small_opening_dims[0]*openings_small[0]
    d = .76
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
            .setColor([1., 1., 0.]) \
            .setContact(1)

    sides_count = 1 if just_front else 2
    for s in range(sides_count):
        p = d*.5-small_opening_dims[2]
        p *= 1. if s == 0 else -1.
        C.addFrame(f"shelf_back_{s}", "shelf_base") \
            .setRelativePosition([p, 0, base_height*.5+h*.5]) \
            .setShape(ry.ST.ssBox, size=[inner_wall_width, w, h, 0.005]) \
            .setColor([1., 1., 0.]) \
            .setContact(1)
        p = w*.25
        p *= 1. if s == 0 else -1.
        

        opening_pos = np.array([
                    p,
                    1*small_opening_dims[0] - openings_small[0]*small_opening_dims[0]*.5 + small_opening_dims[0]*.5,
                    1*small_opening_dims[1] - openings_small[1]*small_opening_dims[1]*.5 + small_opening_dims[1]*.5
                    ])
        
        C.addFrame(f"small_box_right_{s}", "shelf_base") \
                .setRelativePosition([w*.325, p*2 + d - small_opening_dims[2]*4, base_height + 1.04]) \
                .setShape(ry.ST.ssBox, size=[small_opening_dims[2], inner_wall_width, small_opening_dims[1]*11, 0.005]) \
                .setColor([1., 1., 1.]) \
                .setContact(1)
        
        C.addFrame(f"small_box_left_{s}", "shelf_base") \
                .setRelativePosition([-w*.325, p*2 + d - small_opening_dims[2]*4, base_height + 1.04]) \
                .setShape(ry.ST.ssBox, size=[small_opening_dims[2], inner_wall_width, small_opening_dims[1]*11, 0.005]) \
                .setColor([1., 1., 0.]) \
                .setContact(1)

        for i, offset in enumerate(floor_offsets):
            print(d - small_opening_dims[2]*2, w*.5)
            # quit()
            if i == 0:
                C.addFrame(f"big_xy_bottom_{s}_0", "shelf_base") \
                    .setRelativePosition([0., p, base_height + floor_offsets[0]]) \
                    .setShape(ry.ST.ssBox, size=[.38, w*.5, inner_wall_width, 0.001]) \
                    .setColor([1., 1., 0.]) \
                    .setContact(1) \


                C.addFrame(f"big_box_inside_{s}_0", f"big_xy_bottom_{s}_0") \
                    .setRelativePosition([0., 0., -floor_offsets[1]/2+.5*base_height]) \
                    .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., w*.5, floor_offsets[1]-base_height, 0.005]) \
                    .setColor([0., 1., 1., .0]) \
                    .setContact(0)

        
            else:
                C.addFrame(f"big_xy_bottom_{s}_{i}", f"big_xy_bottom_{s}_{i-1}") \
                    .setRelativePosition([0., 0, offset]) \
                    .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., w*.5, inner_wall_width, 0.001]) \
                    .setColor([1., 1., 0.]) \
                    .setContact(1) \
                    .setAttributes({"friction": 1e-4}) \
                
                if i==2:
                    C.addFrame(f"big_box_inside_{s}_{i}", f"big_xy_bottom_{s}_{i}") \
                        .setRelativePosition([0., inner_wall_width, -floor_offsets[i]/2]) \
                        .setShape(ry.ST.ssBox, size=[d - small_opening_dims[2]*2., w*.5-4*inner_wall_width, floor_offsets[i]-base_height/2, 0.005]) \
                        .setColor([0., 1., 1., .0]) \
                        .setContact(0)
                


            p = w*.5
            p *= 1. if s == 0 else -1.

            
    C.getFrame("cameraShelf") \
        .setPosition(C.getFrame("big_xy_bottom_0_1").getPosition()+np.array([-.22*w, 0, .1]) + np.array([-.25, 0, floor_offsets[2]])) \
        .setQuaternion([np.cos(np.deg2rad(140/2)), 0, np.sin(np.deg2rad(140/2)), 0]) 
        # .setAttributes({"focalLength": .895, "width": 224, "height": 224})


if __name__ == "__main__":
    C = ry.Config()
    pos = np.array([0., 0., 0.])
    generate_shelf(C, pos)
    C.view(True)
