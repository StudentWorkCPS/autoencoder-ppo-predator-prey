import numpy as np

np.random.seed(0)

range_min = .5
range_max = 1.5

def pos_gen(rangex,rangey):
    
    x = np.random.uniform(rangex[0],rangex[1])
    y = np.random.uniform(rangey[0],rangey[1])

    x = round(x,2)
    y = round(y,2)

    return x,y

def predator_gen(idx):
    yr = [-range_max,-range_min]
    
    if idx == 0:
        xr = [-range_max,-range_min]
    elif idx == 1:
        xr = [range_min,range_max]

    angle = np.random.randint(0,360)

    return pos_gen(xr,yr),angle

def prey_gen():
    angle = np.random.randint(0,360)
    return pos_gen([-range_max,range_min],[range_min,1]),angle

def round_rob(rob,size):
    (x,y),(a) = rob
    
    return (int(round(x/2 * size / 2)),int(round(y/2 * size / 2))),a

def in_bounds(pos,range):
    (x,y) = pos
    (xmin,xmax),(ymin,ymax) = range

    return xmin <= x <= xmax and ymin <= y <= ymax

def world_to_string(pred1,pred2,prey):
    size = 20
    (x1,y1),(a1) = round_rob(pred1,size)
    (x2,y2),(a2) = round_rob(pred2,size)
    (x3,y3),(a3) = round_rob(prey,size)
    
    print(x3,y3)

    line = "".join(["--" for i in range(size+1)])
    for i in range(size):
        line += "\n|"    
        for j in range(size):
            xj = size / 2 - j
            yi = size / 2 - i

            if x1 == xj and y1 == yi:
                line += angle_to_string(a1) +"1"
            elif x2 == xj and y2 == yi:
                line += angle_to_string(a2) +"2"
            elif x3 == xj and y3 == yi:
                line += angle_to_string(a3) +"3"
            elif xj == 0 or yi == 0:
                line += ".."
            else:
                line += "  "
        line += "|"
       
    line += "\n"
    line += "".join(["--" for i in range(size+1)])
    return line

arrows_emojis = ["↑","↗️","➡️","↘️","↓","↙️","←","↖️"]


def angle_to_string(angle):
    # display the angle with arrows
    idx = int(round(angle / 45))  % 8

    return arrows_emojis[idx]
    

data = {'p1':[], 'p2':[], 'prey': []}


for i in range(10):
    p1_pos,p1_a = predator_gen(0)
    p2_pos,p2_a = predator_gen(1)
    prey_pos,pr_a = prey_gen()

    data['p1'].append((p1_pos,p1_a))
    data['p2'].append((p2_pos,p2_a))
    data['prey'].append((prey_pos,pr_a))

    print("------------------------")
    print("Iteration: ",i)
    
    print(world_to_string((p1_pos,p1_a),(p2_pos,p2_a),(prey_pos,pr_a)))

    print(f'Predator 1: {p1_pos} {p1_a}° ({angle_to_string(p1_a)})')
    print(f'Predator 2: {p2_pos} {p2_a}°  ({angle_to_string(p2_a)})')
    print(f'Prey: {prey_pos} {pr_a}° ({angle_to_string(pr_a)})')
    print("")

import json 
with open('data.json', 'w') as outfile:
    json.dump(data, outfile)