import numpy as np
import os
import sys
import subprocess

try:
    from mayavi import mlab
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "mayavi"])
    from mayavi import mlab

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def generate_colors(num_colors):
    # Generate a colormap with `num_colors` distinct colors
    colors = plt.cm.tab20(np.linspace(0, 1, num_colors))
    return ListedColormap(colors)

def color_from_labels(labels):
    n_colors = np.max(labels) + 1
    reds = np.linspace(0, 1, n_colors)
    greens = np.linspace(1, 0, n_colors)

    def paint_label(label):
        # if its a noise point
        if label == 0:
            # color it white
            return (1, 1, 1)
        else:
            return (reds[label], greens[label], 0)
    return [paint_label(label) for label in labels]

def load_data(filename):
    data = np.loadtxt(filename)
    return data

def plot_point_cloud(point_cloud, labels=None):
    x = point_cloud[:, 0]
    y = point_cloud[:, 1]
    z = point_cloud[:, 2]

    mlab.points3d(x, y, z, mode='point')

def animate_point_clouds(root_folder, file_list, labels_folder, interval=1000):
    mlab.figure(bgcolor=(0, 0, 0))

    mlab.view(azimuth=45, elevation=45)

    # create empty visualization
    # colors: https://docs.enthought.com/mayavi/mayavi/mlab_changing_object_looks.html
    plt = mlab.points3d(0, 0, 0, mode="point", colormap="gist_rainbow")

    @mlab.show
    @mlab.animate(delay=interval)
    def anim():
        f = mlab.gcf()
        for filename in file_list:
            print(f"filename: {filename}")
            point_cloud_path = os.path.join(root_folder, filename)
            point_cloud = load_data(point_cloud_path)

            labels_path = os.path.join(labels_folder, filename)
            labels = load_data(labels_path).astype(int)
                        

            # point cloud without the pointsd that have been classified as noise
            filtered_point_cloud = point_cloud[labels > 0]
            labels_filtered = labels[labels > 0]

            plt.mlab_source.reset(x=filtered_point_cloud[:, 0], 
                                  y=filtered_point_cloud[:, 1], 
                                  z=filtered_point_cloud[:, 2],
                                  scalars=labels_filtered / np.max(labels_filtered)
                                  )
            yield
    anim()

def main():
    # conversions, inputs
    point_clouds_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                       "clustering/inputs/pointclouds_sequences/1")
    point_cloud_file_names = os.listdir(point_clouds_folder)

    point_cloud_file_names = sorted(point_cloud_file_names, key=lambda x: int(x.split("_")[1].split(".txt")[0]))

    labels_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), 
                                 "clustering/predictions/pointclouds_sequences/1")

    # Set the interval (in milliseconds) for displaying each point cloud (1 second in this case)
    display_rate = 1000  # 1 second per point cloud

    # Animate the sequence of point clouds
    animate_point_clouds(point_clouds_folder, 
                         point_cloud_file_names, 
                         labels_folder=labels_folder, 
                         interval=display_rate)


if __name__ == "__main__":
    main()
