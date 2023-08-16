import numpy as np
import vispy.scene
from vispy.scene import visuals
import sys


# Make a canvas and add simple view
canvas = vispy.scene.SceneCanvas(keys='interactive', show=True)
view = canvas.central_widget.add_view()


# 使用 kitti 数据， n*3 
img_id = 0  # 2，3 is not able for pcl;
path = "000001.bin"   ## Path ## need to be changed
points = np.fromfile(path, dtype=np.float32).reshape(-1, 4)



# create scatter object and fill in the data
scatter = visuals.Markers()
scatter.set_data(points[:,:3], edge_color=None, face_color=(1, 1, 1, .5), size=5)

view.add(scatter)
# view.camera = 'turntable'  # or try 'arcball'

# add a colored 3D axis for orientation
axis = visuals.XYZAxis(parent=view.scene)


if __name__ == '__main__':
    if sys.flags.interactive != 1:
        vispy.app.run()




