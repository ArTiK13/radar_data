import pandas as pd
import math
import matplotlib.pyplot as plt
import typing as tp
import json


#constant gradient
contrast_0_to_1 : float = 0.5
opacity_0_to_1 : float = 0.5 

#constant shift_delt_t
index_radar_data : int = 0  #0 to 100
radar_df : pd.DataFrame = pd.read_csv(f"data/processed data/radar_data_{index_radar_data}.csv")
radar_df : pd.DataFrame = radar_df[radar_df["QPDH0"] < 0.3]



def gradient(lidar_df : pd.DataFrame, contrast_cof : float = contrast_0_to_1, opacity_cof : float = opacity_0_to_1) -> list[float]:
    """
    Converts the reflection lidar_df column to a gradient from 0 to 1.

    Parameters
    ----------
    lidar_df : pd.DataFrame
        A lidar dataframe that has a reflection column
    gradient_cof : float
        Float from the segment [0, 1]. The smaller the number, the stronger the contrast
        

    Returns
    -------
    list[float]
        a list containing floats of [0, 1] that are responsible for the color and opacity in RGB
    """
    gradient_list : list[float] = []

    #хотим выбрать степенную функцию, для которой средний градиент переходит в 0.5
    mean_gradient : float= sum(lidar_df["r, (reflectance)"]) / (len(lidar_df["r, (reflectance)"]))
    power_normalization : float = math.log(2) / math.log(mean_gradient) 


    for reflect in lidar_df["r, (reflectance)"]:
        color : float = (reflect / 255) ** power_normalization

        #переводим в [-1, 1] берём корень и переаодим обратно в [0, 1]
        if((2 * color - 1) > 0):
            color_normalize : float = 0.5 * ((abs((2 * color - 1)) ** contrast_0_to_1) + 1)
        else:
            color_normalize : float  = 0.5 * (-(abs((2 * color - 1)) ** contrast_0_to_1) + 1)
        
        gradient_list.append((color_normalize, 0, 1 - color_normalize, opacity_cof))
    return gradient_list




def shift_delt_t(delta_t : float = 0.06) -> pd.DataFrame:
    """
    Moves radar_df to delta_t in time

    Parameters
    ----------
    delta_t : float
        Time travel radar_df
        

    Returns
    -------
    pd.DataFrame
        radar_df which has shifted to delta_t
    """

    with open(f"../data/raw data/radar_positions.json", "r") as file:
        radar_positions : dict[float, tp.Any] = {float(k): v for k, v in json.load(file).items()}

    for i, cords in radar_positions.items():
        for j, ax in enumerate(("X, (m)", "Y, (m)")):
            radar_df.loc[radar_df["radar_idx"] == i, ax] -= cords[j]

    vector_length : pd.Series = (radar_df['X, (m)']**2 + radar_df['Y, (m)']**2)**0.5
    radar_df['RadialDelta'] = (delta_t - radar_df['(radar_point_ts - lidar_ts), (s)']) * radar_df['AbsoluteRadialVelocity']
    radar_df['XwithDelta'] = radar_df['X, (m)'] * (vector_length + radar_df['RadialDelta']) / vector_length
    radar_df['YwithDelta'] = radar_df['Y, (m)'] * (vector_length + radar_df['RadialDelta']) / vector_length


    for i, cords in radar_positions.items():
        for j, ax in enumerate(("XwithDelta", "YwithDelta")):
            radar_df.loc[radar_df["radar_idx"] == i, ax] += cords[j]

    for i, cords in radar_positions.items():
        for j, ax in enumerate(("X, (m)", "Y, (m)")):
            radar_df.loc[radar_df["radar_idx"] == i, ax] += cords[j]
    
    return radar_df


def standart_plot_generator(type_radar_lidar : {"radar", "lidar", "radar_lidar"}, frame_num : int, path_file : str, title_graph : str = None,
                color_radar : str | list[float] | list[tuple[float]]= "green", color_lidar : str | list[float] | list[tuple[float]] = None, 
                figsize_ : tuple[int] = (22, 10), x_lim : list[int] = [-110, 110], y_lim :list[int] = [-50, 50]) -> None:
    """
    Saves a plt file in which the radar or lidar points (maybe both). 
    The data is taken from processed data with the frame_num number and stored along the path_file path with the title_graph header. 
    The color_radar color is selected for the radar, and the color_lidar color is selected for the lidar.
    The dimensions of the graph figsize_ and the limits on the x and y axes are x_lim and y_lim.

    Parameters
    ----------
    type_radar_lidar : {"radar", "lidar", "radar_lidar"}
        Point type: radar, lidar, or both

    frame_num : int
        The frame number selected for building

    path_file : str
        The file path to save is calculated from the initial folder
        
    title_graph : str = None
        Title graph. Default is "Frame {title_graph}"
    
    color_radar : str | list[float] | list[tuple[float]] = "green"
        The color of the radar in RGB. The default is green

    color_lidar : str | list[float] | list[tuple[float]] = None
        Lidar color. By default, gradient(frame_lidar)

    figsize_ : tuple[int] = (22, 10)
        The dimensions of the plt. Default (22, 11)

    x_lim : list[int] = [-110, 110]
        Dimensions on the x-axis. Default is [-110, 110]

    y_lim :list[int] = [-50, 50]
        Dimensions on the y-axis. Default is [-50, 50]


    Returns
    -------
    None
        Saves the file from the plt
    """
    
    plt.figure(figsize_)
    plt.xlim(x_lim)
    plt.ylim(y_lim)

    #radar
    if(type_radar_lidar == "radar" or type_radar_lidar == "radar_lidar"):
        frame_radar : pd.DataFrame = pd.read_csv(f"data/processed data/radar_data_{frame_num}.csv")
        plt.scatter(frame_radar[frame_radar["QPDH0"] < 0.3]["X, (m)"], frame_radar[frame_radar["QPDH0"] < 0.3]["Y, (m)"], s=1, c = color_radar)

    #lidar
    if(type_radar_lidar == "lidar" or type_radar_lidar == "radar_lidar"):
        frame_lidar : pd.DataFrame = pd.read_csv(f"data/processed data/lidar_data_{frame_num}.csv")
        if(color_lidar is None):
            plt.scatter(frame_lidar["X, (m)"], frame_lidar["Y, (m)"], s=1, c=gradient(frame_lidar))
        else:
            plt.scatter(frame_lidar["X, (m)"], frame_lidar["Y, (m)"], s=1, c=color_lidar)


    if title_graph is None:
        plt.title(f"Frame {title_graph}")
    else:
        plt.title(title_graph)

    
    plt.savefig(path_file)
    plt.close()
    return None

