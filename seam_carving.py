# -*- coding: utf-8 -*-
# 
# Seam Carving Implementation
# Will Emmanuel
# 

import numpy as np
import cv2
import sys

def load_image_greyscale(name='images/fig8.png'):
    return cv2.imread('images/fig8.png', 0)

def load_image(name='images/fig5.png'):
    return cv2.imread(name)

def save(img,name='output'):
    cv2.imwrite('{name}.png'.format(name=name),img)

def energy(img):
    return np.absolute(cv2.Laplacian(img,cv2.CV_64F))

def cost_grid_vertical(energy):
    costs = np.zeros(energy.shape)
    # set first column to the first column of energy
    costs[0] = energy[0]
    for i in range(1,costs.shape[0]):
        for j in range(0,costs.shape[1]):
            vals = [
                costs[i-1][j+1] if j < costs.shape[1]-1 else float('inf'),
                costs[i-1][j],
                costs[i-1][j-1] if j > 0 else float('inf')
            ]
            costs[i][j] = np.amin(vals) + energy[i][j]
    return costs

def cost_grid_horizontal(energy):
    costs = np.zeros(energy.shape)
    # set first row to the first row of energy
    costs[:,0] = energy[:,0]
    for j in range(1,costs.shape[1]):
        for i in range(0,costs.shape[0]):
            vals = [
                costs[i+1][j-1] if i < costs.shape[0]-1 else float('inf'),
                costs[i][j-1],
                costs[i-1][j-1] if i > 0 else float('inf')
            ]
            costs[i][j] = np.amin(vals) + energy[i][j]
    return costs

def get_shortest_vertical_path(costs, excluding=None):
    # Last row will have all costs
    shape = costs.shape
    min_costs = costs[shape[0]-1]
    if excluding is not None:
        excluding_bottom_row = excluding[shape[0]-1]
        excluding_bottom_row = excluding_bottom_row * np.full(excluding_bottom_row.shape, np.inf)
        excluding_bottom_row = np.nan_to_num(excluding_bottom_row)
        min_costs = excluding_bottom_row + min_costs

    min_cost_index = min_costs.argmin()
    min_cost = min_costs[min_cost_index]
    indicies = [(shape[0]-1, min_cost_index)]
    for i in range(shape[0]-2, -1, -1):
        if excluding is None:
            vals = [
                (costs[i][min_cost_index+1] if min_cost_index < shape[1]-1 else float('inf'), min_cost_index+1),
                (costs[i][min_cost_index], min_cost_index),
                (costs[i][min_cost_index-1] if min_cost_index > 0 else float('inf'), min_cost_index-1)
            ]

        else:
            j = min_cost_index
            vals = [
                (costs[i][j+1] if (j < shape[1]-1 and excluding[i][j+1] == 0) else float('inf'), j+1),
                (costs[i][j] if (excluding[i][j] == 0) else float('inf'), j),
                (costs[i][j-1] if (j > 0 and excluding[i][j-1] == 0) else float('inf'), j-1)
            ]
            vals = list(filter(lambda x: (x[1] < shape[1]-1 and x[1] > 0), vals))

        min_cost_index = min(vals, key=lambda x:x[0])[1]
        indicies.append((i, min_cost_index))
    return (indicies,min_cost)

def get_shortest_horizontal_path(costs, excluding=None):
    # Last col will have all costs
    shape = costs.shape
    min_costs = costs[:,shape[1]-1]
    if excluding is not None:
        excluding_bottom_col = excluding[:,shape[1]-1]
        excluding_bottom_col = excluding_bottom_col * np.full(excluding_bottom_col.shape, np.inf)
        excluding_bottom_col = np.nan_to_num(excluding_bottom_col)
        min_costs = excluding_bottom_col + min_costs

    min_cost_index = min_costs.argmin()
    indicies = [(shape[1]-1, min_cost_index)]
    min_cost = min_costs[min_cost_index]
    for i in range(shape[1]-2, -1, -1):
        if excluding is None:
            vals = [
                (costs[min_cost_index+1][i] if min_cost_index < shape[0]-1 else float('inf'), min_cost_index+1),
                (costs[min_cost_index][i], min_cost_index),
                (costs[min_cost_index-1][i] if min_cost_index > 0 else float('inf'), min_cost_index-1)
            ]
        else:
            j = min_cost_index
            vals = [
                (costs[j+1][i] if (j < shape[0]-1 and excluding[j+1][i] == 0) else float('inf'), j+1),
                (costs[j][i] if excluding[j][i] == 0 else float('inf'), j),
                (costs[j-1][i] if (j > 0 and excluding[j-1][i] == 0) else float('inf'), j-1)
            ]
            vals = list(filter(lambda x: (x[1] < shape[0]-1 and x[1] > 0), vals))

        min_cost_index = min(vals, key=lambda x:x[0])[1]
        indicies.append((i, min_cost_index))
    return (indicies,min_cost)

def highlight_path(img, paths, vertical=True):
    for i in range(0, img.shape[0]):
        for path in paths:
            if vertical:
                img[path[0]][path[1]] = [255, 0, 0]
            else:
                img[path[1]][path[0]] = [255, 0, 0]
    return img

def insert_vertical_path(img, paths):
    new_img = np.zeros((img.shape[0], img.shape[1]+1, img.shape[2]))
    for path in paths:
        x=path[1]
        y=path[0]
        new_img[y, 0:x,:] = img[y, 0:x,:]
        new_img[y, x:new_img.shape[1],:] = img[y, x-1:new_img.shape[1]+1,:]
    return new_img

def insert_horizontal_path(img, paths):
    new_img = np.zeros((img.shape[0]+1, img.shape[1], img.shape[2]))
    for path in paths:
        x=path[0]
        y=path[1]
        new_img[0:y, x,:] = img[0:y, x,:]
        new_img[y:new_img.shape[0], x,:] = img[y-1:new_img.shape[0]+1, x,:]
    return new_img

def remove_vertical_path(img, paths):
    new_img = np.zeros((img.shape[0], img.shape[1]-1, img.shape[2]))
    for path in paths:
        x=path[1]
        y=path[0]
        new_img[y, 0:x,:] = img[y, 0:x,:]
        new_img[y, x:new_img.shape[1],:] = img[y, x+1:new_img.shape[1]+1,:]
    return new_img

def remove_horizontal_path(img, paths):
    new_img = np.zeros((img.shape[0]-1, img.shape[1], img.shape[2]))
    for path in paths:
        x=path[0]
        y=path[1]
        new_img[0:y, x,:] = img[0:y, x,:]
        new_img[y:new_img.shape[0], x,:] = img[y+1:new_img.shape[0]+1, x,:]
    return new_img

def remove_column(img):
    energies = energy(img[:,:,0]) + energy(img[:,:,1]) + energy(img[:,:,2])
    cost = cost_grid_vertical(energies)
    paths,test = get_shortest_vertical_path(cost)
    return remove_vertical_path(img, paths)

def remove_row(img):
    energies = energy(img[:,:,0]) + energy(img[:,:,1]) + energy(img[:,:,2])
    cost = cost_grid_horizontal(energies)
    paths,test = get_shortest_horizontal_path(cost)
    return remove_horizontal_path(img, paths)

def remove_columns(img, count):
    for i in range(count):
        img = remove_column(img)
    return img

def remove_rows(img, count):
    for i in range(count):
        img = remove_row(img)
    return img

def run():
    if len(sys.argv) != 4 or sys.argv[1] is None or sys.argv[2] is None or sys.argv[3] is None:
        print "Please give image name, reduction in width, and reduction in height."
        sys.exit(1)

    img = load_image(sys.argv[1])
    if img is None:
        print "Invalid image."
        sys.exit(1)

    widthDelta = int(sys.argv[2])
    heightDelta = int(sys.argv[3])

    img = remove_columns(img, widthDelta)
    img = remove_rows(img, heightDelta)
    save(img)

if sys.argv[0] == './seam_carving.py':
    run()
