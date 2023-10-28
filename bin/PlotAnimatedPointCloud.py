import numpy as np
import os
import sys
import subprocess
import copy

from py_helpers import IO

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



def draw_bounding_boxes_3d(point_cloud, labels, fig, bounding_box_matrix_plot:np.ndarray, color=(1, 1, 1)):
    """
    Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (3, 8) for XYZs of the box corners
        fig: figure handler
        color: RGB value tuple in range (0, 1), box line color
    """
    min_x = np.min(point_cloud[:, 0], axis=0)
    max_x = np.max(point_cloud[:, 0], axis=0)
    min_y = np.min(point_cloud[:, 1], axis=0)
    max_y = np.max(point_cloud[:, 1], axis=0)
    min_z = np.min(point_cloud[:, 2], axis=0)
    max_z = np.max(point_cloud[:, 2], axis=0)
    
    bounding_box_3d = np.array([
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, max_z],
        [min_x, max_y, max_z]
    ]).T

    if bounding_box_matrix_plot.size == 0:
        bounding_box_matrix_plot

    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        mlab.plot3d([bounding_box_3d[0, i], bounding_box_3d[0, j]], 
                    [bounding_box_3d[1, i], bounding_box_3d[1, j]],
                    [bounding_box_3d[2, i], bounding_box_3d[2, j]], 
                    tube_radius=None, 
                    line_width=2, 
                    color=color, 
                    figure=fig)

        i, j = k + 4, (k + 1) % 4 + 4
        mlab.plot3d([bounding_box_3d[0, i], bounding_box_3d[0, j]], 
                    [bounding_box_3d[1, i], bounding_box_3d[1, j]],
                    [bounding_box_3d[2, i], bounding_box_3d[2, j]], 
                    tube_radius=None, 
                    line_width=2, 
                    color=color, 
                    figure=fig)

        i, j = k, k + 4
        mlab.plot3d([bounding_box_3d[0, i], bounding_box_3d[0, j]], 
                    [bounding_box_3d[1, i], bounding_box_3d[1, j]],
                    [bounding_box_3d[2, i], bounding_box_3d[2, j]], 
                    tube_radius=None, 
                    line_width=2, 
                    color=color, 
                    figure=fig)
    return fig
    

def draw_bounding_boxes_3d_2(point_cloud, labels, bounding_box_matrix_plot):
    """
    Draw 3D bounding boxes
    Args:
        gt_boxes3d: numpy array (3, 8) for XYZs of the box corners
        fig: figure handler
        color: RGB value tuple in range (0, 1), box line color
    """
    min_x = np.min(point_cloud[:, 0], axis=0)
    max_x = np.max(point_cloud[:, 0], axis=0)
    min_y = np.min(point_cloud[:, 1], axis=0)
    max_y = np.max(point_cloud[:, 1], axis=0)
    min_z = np.min(point_cloud[:, 2], axis=0)
    max_z = np.max(point_cloud[:, 2], axis=0)
    
    bounding_box_3d = np.array([
        [min_x, min_y, min_z],
        [max_x, min_y, min_z],
        [max_x, max_y, min_z],
        [min_x, max_y, min_z],
        [min_x, min_y, max_z],
        [max_x, min_y, max_z],
        [max_x, max_y, max_z],
        [min_x, max_y, max_z]
    ]).T

    for k in range(0, 4):
        i, j = k, (k + 1) % 4
        bounding_box_matrix_plot[k][0].mlab_source.set(
            x=[bounding_box_3d[0, i], bounding_box_3d[0, j]], 
            y=[bounding_box_3d[1, i], bounding_box_3d[1, j]],
            z=[bounding_box_3d[2, i], bounding_box_3d[2, j]])

        i, j = k + 4, (k + 1) % 4 + 4
        bounding_box_matrix_plot[k][1].mlab_source.set(
            x=[bounding_box_3d[0, i], bounding_box_3d[0, j]], 
            y=[bounding_box_3d[1, i], bounding_box_3d[1, j]],
            z=[bounding_box_3d[2, i], bounding_box_3d[2, j]])

        i, j = k, k + 4
        bounding_box_matrix_plot[k][2].mlab_source.set(
            x=[bounding_box_3d[0, i], bounding_box_3d[0, j]], 
            y=[bounding_box_3d[1, i], bounding_box_3d[1, j]],
            z=[bounding_box_3d[2, i], bounding_box_3d[2, j]])

def redraw_labelled_point_cloud(point_cloud_xyz, labels, points_plot):
    points_plot.mlab_source.reset(
                x=point_cloud_xyz[:, 0],
                y=point_cloud_xyz[:, 1],
                z=point_cloud_xyz[:, 2],
                scalars=labels
            )

def animate_point_clouds_from_txt_files(
    root_folder, file_list, labels_folder, interval=1000
):
    mlab.figure(bgcolor=(0, 0, 0), size=(1080, 720))

    mlab.view(azimuth=45, elevation=45)

    # create empty visualization
    # colors: https://docs.enthought.com/mayavi/mayavi/mlab_changing_object_looks.html
    points_plot = mlab.points3d(0, 0, 0, mode="point", colormap="gist_rainbow")

    def make_bounding_box_plot_init():
        return mlab.plot3d([0, 0], [0, 0], [0, 0], tube_radius=None, line_width=2, color=(1, 1, 1))
    
    # bounding_box_matrix_plot = [[make_bounding_box_plot_init() for _ in range(3)] for _ in range(4)]

    @mlab.show
    @mlab.animate(delay=interval)
    def anim():
        figure = mlab.gcf()
        for filename in file_list:
            print(f"filename: {filename}")
            point_cloud_path = os.path.join(root_folder, filename)
            point_cloud = IO.auto_decode(point_cloud_path, dtype=np.float32)

            labels_path = os.path.join(labels_folder, filename)
            labels = IO.auto_decode(labels_path, dtype=np.int64)
            unique_labels = np.unique(labels)

            print(point_cloud.shape)
            print(labels.shape)
            print(f"Unique labels: {unique_labels.shape[0]}")

            redraw_labelled_point_cloud(point_cloud, labels, points_plot)
            
            # draw_bounding_boxes_3d_2(point_cloud, labels, bounding_box_matrix_plot)
            yield

    anim()


def animate_point_clouds_from_bin_files(
    root_folder, file_list, labels_folder, interval=1000
):
    mlab.figure(bgcolor=(0, 0, 0), size=(1080, 720))

    mlab.view(azimuth=45, elevation=45)

    # create empty visualization
    # colors: https://docs.enthought.com/mayavi/mayavi/mlab_changing_object_looks.html
    plot = mlab.points3d(0, 0, 0, mode="point", colormap="gist_rainbow")

    @mlab.show
    @mlab.animate(delay=interval)
    def anim():
        f = mlab.gcf()
        for filename in file_list:
            print(f"filename: {filename}")
            point_cloud_path = os.path.join(root_folder, filename)
            point_cloud = IO.auto_decode(point_cloud_path, np.float32, n_features=4)[:, :3]

            labels_path = os.path.join(labels_folder, filename)
            labels = IO.auto_decode(labels_path, np.int64, n_features=1).ravel()

            unique_labels = np.unique(labels)

            print(point_cloud.shape)
            print(labels.shape)
            print(f"Unique labels: {unique_labels.shape[0]}")

            # point cloud without the pointsd that have been classified as noise
            filtered_point_cloud = point_cloud[labels > 0]
            labels_filtered = labels[labels > 0]

            plot.mlab_source.reset(
                x=filtered_point_cloud[:, 0],
                y=filtered_point_cloud[:, 1],
                z=filtered_point_cloud[:, 2],
                scalars=labels_filtered / np.max(labels_filtered),
            )
            yield

    anim()


def main():
    folder_name = "0000"
    # conversions, inputs
    point_clouds_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "clustering/inputs/pointclouds_sequences/" + folder_name,
    )
    point_cloud_file_names = os.listdir(point_clouds_folder)

    files_extension = point_cloud_file_names[0].split(".")[-1]

    if files_extension == "txt":
        point_cloud_file_names = sorted(point_cloud_file_names, key=lambda x: int(x.split("_")[1].split(".txt")[0]))
    elif files_extension == "bin":    
        point_cloud_file_names = sorted(point_cloud_file_names, key=lambda x: int(x.split(".bin")[0]))
    else:
        print("file format not supported")

    labels_folder = os.path.join(
        os.path.dirname(os.path.realpath(__file__)),
        "clustering/predictions/pointclouds_sequences/" + folder_name,
    )

    # Set the interval (in milliseconds) for displaying each point cloud
    display_rate = 250

    if files_extension == "txt":
        animate_point_clouds_from_txt_files(
            point_clouds_folder,
            point_cloud_file_names,
            labels_folder=labels_folder,
            interval=display_rate,
        )
    elif files_extension == "bin":
        animate_point_clouds_from_bin_files(
            point_clouds_folder,
            point_cloud_file_names,
            labels_folder=labels_folder,
            interval=display_rate,
        )
    else:
        print("file format not supported")


if __name__ == "__main__":
    main()