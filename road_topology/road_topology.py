# Autor:    Patiphon Narksri
# Date:     17/08/2019

import carla
import pandas as pd

# Establish connection
client = carla.Client('localhost', 2000)
client.set_timeout(10.0) # seconds
# Get the environment information
world = client.get_world()
# Get debugger
debug = world.debug

# Get road network
ROADNETWORK_RES = 0.2 # [m]
map = world.get_map()
print(map)

waypoint_list = map.generate_waypoints(ROADNETWORK_RES)
id = []
x = []
y = []
z = []

start_node = []
end_node = []

for point in waypoint_list:
    # Get lane points
    id.append(point.id)
    x.append(point.transform.location.x)
    y.append(point.transform.location.y)
    z.append(point.transform.location.z)
    debug.draw_point(location=point.transform.location)

    # Get connections
    start_node.append(point.id)
    nodes = []
    for next_node in point.next(ROADNETWORK_RES):
        nodes.append(next_node.id)
        id.append(next_node.id)
        x.append(next_node.transform.location.x)
        y.append(next_node.transform.location.y)
        z.append(next_node.transform.location.z)
        debug.draw_point(location=point.transform.location, color=carla.Color(0, 255, 0))
    end_node.append(nodes)

point = pd.DataFrame({'id':id,'x':x,'y':y,'z':z})
point.to_csv('points_carla.csv', index=False, header=False)

with open('nodes_carla.csv', 'w') as file:
    for id, start in enumerate(start_node):
        string = str(start)
        for ends in end_node[id]:
            string = string + ', ' + str(ends)
        file.write(string)
        file.write('\n')
