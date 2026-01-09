import numpy as np
import robotic as ry

def generate_shelf(C: ry.Config, pos: np.ndarray, w = .38, d=.447, h=2, base_quaternion: list[float]=[1,0,0,0], floor_offsets: list[float]=[0.35, 0.43, 0.30, 0.18, 0.15, 0.2, 0.15, 0.15, 0.15, 0.12]):
    # TODO: More efficient piece building, don't repeat pieces!
    inner_wall_width = .02

    w = w
    d = d
    h = h

    base_height = .05

    C.addFrame("shelf_base") \
        .setPosition(pos + np.array([0., 0., base_height*.5])) \
        .setShape(ry.ST.ssBox, size=[w, d, base_height, 0.005]) \
        .setColor([.8, .8, .8]) \
        .setContact(1) \
        .setQuaternion(base_quaternion)


    p = w*.5+inner_wall_width*.5
    C.addFrame(f"shelf_back_0", "shelf_base") \
        .setRelativePosition([p, d/2-inner_wall_width*.5, base_height*.5+h*.5]) \
        .setShape(ry.ST.ssBox, size=[inner_wall_width, d+inner_wall_width, h, 0.005]) \
        .setColor([1., 1., 0.]) \
        .setContact(1)
    C.addFrame(f"shelf_back_1", "shelf_base") \
        .setRelativePosition([-p, d/2-inner_wall_width*.5, base_height*.5+h*.5]) \
        .setShape(ry.ST.ssBox, size=[inner_wall_width, d+inner_wall_width, h, 0.005]) \
        .setColor([1., 1., 0.]) \
        .setContact(1)

    C.addFrame("shelf_middle", "shelf_base") \
        .setRelativePosition([0, -inner_wall_width/2, base_height*.5+h*.5]) \
        .setShape(ry.ST.ssBox, size=[w, inner_wall_width, h, 0.005]) \
        .setColor([1., 1., 0.]) \
        .setContact(1)
    
    C.addFrame(f"small_box_right_0", "shelf_base") \
            .setRelativePosition([w*.75, d, base_height + 1.04]) \
            .setShape(ry.ST.ssBox, size=[w/2, inner_wall_width, h, 0.005]) \
            .setColor([1., 1., 0.]) \
            .setContact(1)
    
    C.addFrame(f"small_box_right_1", "shelf_base") \
            .setRelativePosition([-w*.75, d, base_height + 1.04]) \
            .setShape(ry.ST.ssBox, size=[w/2, inner_wall_width, h, 0.005]) \
            .setColor([1., 1., 0.]) \
            .setContact(1)
    

    for i, offset in enumerate(floor_offsets):
        print(d, w*.5)
        # quit()
        if i == 0:
            C.addFrame(f"big_xy_bottom_0_0", "shelf_base") \
                .setRelativePosition([0., d/2, base_height + floor_offsets[0]]) \
                .setShape(ry.ST.ssBox, size=[w, d, inner_wall_width, 0.001]) \
                .setColor([1., 1., 0.]) \
                .setContact(1) \


            C.addFrame(f"big_box_inside_0_0", f"big_xy_bottom_0_0") \
                .setRelativePosition([0., 0., -floor_offsets[1]/2+.5*base_height]) \
                .setShape(ry.ST.ssBox, size=[w, d, floor_offsets[1]-base_height, 0.005]) \
                .setColor([0., 1., 1., .0]) \
                .setContact(0)

    
        else:
            C.addFrame(f"big_xy_bottom_0_{i}", f"big_xy_bottom_0_{i-1}") \
                .setRelativePosition([0., 0, offset]) \
                .setShape(ry.ST.ssBox, size=[w, d, inner_wall_width, 0.001]) \
                .setColor([1., 1., 0.]) \
                .setContact(1) \
                .setAttributes({"friction": 1e-4}) \
            
            # if i==2:
            #     C.addFrame(f"big_box_inside_0_{i}", f"big_xy_bottom_0_{i}") \
            #         .setRelativePosition([0., inner_wall_width, -floor_offsets[i]/2]) \
            #         .setShape(ry.ST.ssBox, size=[d, w-4*inner_wall_width, floor_offsets[i]-base_height/2, 0.005]) \
            #         .setColor([0., 1., 1., .0]) \
            #         .setContact(0)


            
    C.getFrame("cameraShelf") \
        .setPosition(C.getFrame("big_xy_bottom_0_1").getPosition()+np.array([-.22*w, 0, .1]) + np.array([-.25, 0, floor_offsets[2]])) \
        .setQuaternion([np.cos(np.deg2rad(140/2)), 0, np.sin(np.deg2rad(140/2)), 0]) 
        # .setAttributes({"focalLength": .895, "width": 224, "height": 224})


if __name__ == "__main__":
    C = ry.Config()
    pos = np.array([0., 0., 0.])
    generate_shelf(C, pos)
    C.view(True)
