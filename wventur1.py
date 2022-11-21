import streamlit as st
from typing import List, Tuple, Dict, Callable
from copy import deepcopy
from queue import PriorityQueue
import numpy as np
import sys
import os
import time

# Environment Setting
full_world = [
['🌾', '🌾', '🌾', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🗻', '🗻', '🗻', '🗻', '🗻', '🗻', '🗻', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🗻', '🗻', '🗻', '🪨', '🪨', '🪨', '🗻', '🗻', '🪨', '🪨'],
['🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🐊', '🐊', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🪨', '🌾'],
['🌾', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🌲', '🌲', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🪨', '🌾'],
['🌾', '🪨', '🪨', '🪨', '🗻', '🗻', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🪨', '🌾', '🌾'],
['🌾', '🪨', '🪨', '🗻', '🗻', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🪨', '🪨', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🪨', '🪨', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🌾', '🐊', '🐊', '🌾', '🌾', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🪨', '🪨', '🗻', '🗻', '🗻', '🗻', '🌾', '🌾', '🌾', '🐊', '🌾', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🪨', '🪨', '🗻', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🗻', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾'],
['🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🌾', '🌾', '🪨', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾'],
['🐊', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🪨', '🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🌾', '🪨', '🗻', '🪨', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🌲', '🌲', '🪨', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🗻', '🌾', '🌾', '🌲', '🌲', '🌲', '🌲', '🪨', '🪨', '🪨', '🪨', '🌾', '🐊', '🐊', '🐊', '🌾', '🌾', '🪨', '🗻', '🪨', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🗻', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🗻', '🗻', '🗻', '🪨', '🪨', '🌾', '🐊', '🌾', '🪨', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🗻', '🗻', '🗻', '🌾', '🌾', '🗻', '🗻', '🗻', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🗻', '🗻', '🗻', '🗻', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🗻', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾'],
['🌾', '🌾', '🌾', '🌾', '🗻', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '🪨', '🪨', '🪨', '🪨', '🗻', '🗻', '🌲', '🌲', '🌲', '🌲', '🌲', '🌾', '🗻', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🪨', '🗻', '🗻', '🗻', '🌲', '🌲', '🗻', '🗻', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🪨', '🗻', '🗻', '🗻', '🗻', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🌾', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🌾', '🪨', '🪨', '🌾', '🌾', '🪨', '🪨', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🪨', '🪨', '🌾', '🐊', '🐊', '🐊', '🐊', '🐊'],
['🪨', '🗻', '🪨', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🗻', '🗻', '🗻', '🪨', '🪨', '🗻', '🗻', '🌾', '🗻', '🗻', '🪨', '🪨', '🐊', '🐊', '🐊', '🐊'],
['🪨', '🗻', '🗻', '🗻', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🗻', '🗻', '🗻', '🗻', '🪨', '🪨', '🪨', '🪨', '🗻', '🗻', '🗻', '🐊', '🐊', '🐊', '🐊'],
['🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾', '🌾', '🪨', '🪨', '🪨', '🌾', '🌾', '🌾']
]
MOVES = [(0,-1), (1,0), (0,1), (-1,0)]
COSTS = { '🌾': 1, '🌲': 3, '🪨': 5, '🐊': 7, '🗻': float('inf')}

# Functions
def get_neighbors(position: tuple, moves: List[Tuple[int, int]], world: List[List[str]]) -> list:
    possible_neighbors = [tuple(np.subtract(position, move))
                          for move in moves]
    neighbors = []
    # Checks if it is within the world and then appends to actual neighbors
    for neighbor in possible_neighbors:
        if (neighbor[0] in range(len(world)) and
                neighbor[1] in range(len(world[0]))):
            neighbors.append(neighbor)
    return neighbors

def heuristic(start: tuple, goal: tuple) -> int:
    h = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
    return h

def a_star_search(world: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int], moves: List[Tuple[int, int]], heuristic: Callable) -> List[Tuple[int, int]]:
    frontier = PriorityQueue()
    frontier.put(start, 0)

    path = {start: None}
    g_n = {start: 0}
    while not frontier.empty():
        current_node = frontier.get()
        if current_node == goal:
            final = [current_node]
            while current_node in path:
                current_node = path[current_node]
                final.append(current_node)
            final = final[::-1]
            path_actions = [tuple(np.subtract(j, i)) for i, j in zip(final[1:], final[2:])]
            final_path_actions = []
            for action in path_actions:
                reversed_action = action[::-1]
                final_path_actions.append(reversed_action)
            break
        neighbors = get_neighbors(current_node, moves, world)
        for neighbor in neighbors:

            new_g = g_n[current_node] + costs.get(world[neighbor[0]][neighbor[1]])

            if neighbor not in g_n or new_g < g_n[neighbor]:
                g_n[neighbor] = new_g
                priority = new_g + heuristic(neighbor, goal)
                frontier.put(neighbor, priority)
                path[neighbor] = current_node

    return final_path_actions

def pretty_print_solution(world: List[List[str]], path: List[Tuple[int, int]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int]) -> int:
    dictionary = {(0, 1): '⏬', (1, 0): '⏩', (0, -1): '⏫', (-1, 0): '⏪', goal: '🎁'}

    total_cost = 0
    for action in path:
        world[start[1]][start[0]] = dictionary.get(action)
        start = tuple(np.add(start, action))
        total_cost += costs.get(world[start[1]][start[0]])
        world[start[1]][start[0]] = dictionary.get(action)

    world[goal[1]][goal[0]] = dictionary.get(goal)

    for row in world:
        row_string = ''
        for char in row:
            row_string += f'{str(char):<1} '.replace('.', ' ')
        st.write(row_string)
    return total_cost

def print_environment(world:List[List[str]]):
    for row in world:
        row_string = ''
        for char in row:
            row_string += f'{str(char):<1} '.replace('.', ' ')
        st.write(row_string)
    return


# Streamlit App
# Title
st.title("State Space Search with A* Search")
# A* Search Documentation if checked
if st.checkbox("Show A* Search Documentation"):
    st.markdown('## a_star_search\n'
                '`a_star_search` is a best-first search algorithm that uses f(n) = g(n) + h(n). '
                'Whether `a_star_search` is cost-optimal depends on the ley property that the `heuristic` is admissible, '
                'an admissible heuristic is one that never overestimates the cost to reach the goal. The algorithm starts off in the '
                'Initial State, at the frontier it then gets the neighbors and calculates the f(n), the estimated cost of the best path that it continues. '
                'After evaluating all the neighbors, the one with the lowest cost will then be pushed into the priority queue.'
                ' Now the next state in the priority queue is evaluated and etc. The algorithm continues until it reaches the goal State. \n'
                
                '* **h(n)** : Heuristic from Greedy Search, estimated cost of the cheapest path from the state at node n to goal state \n'
                '* **g(n)** : Path cost from the initial State to node n (cost of path taken so far) \n'
                '* **f(n)** : estimated cost of the best path that continues \n'
                '\n'
                '* **world** List[List[str]]: the actual context for the navigation problem.\n'
                '* **start** Tuple[int, int]: the starting location of the bot, `(x, y)`.\n'
                '* **goal** Tuple[int, int]: the desired goal position for the bot, `(x, y)`.\n'
                '* **costs** Dict[str, int]: is a `Dict` of costs for each type of terrain in **world**.\n'
                '* **moves** List[Tuple[int, int]]: the legal movement model expressed in offsets in **world**.\n'
                '* **heuristic** Callable: is a heuristic function that returns an estimate of the total cost $f(x)$ from the'
                ' start to the goal through the current node, $x$. The heuristic function might change with the movement model. '
                'EDITED: Estimated cost of getting from n to the goal\n'
                '\n'
                '**returns** List[Tuple[int, int]]: the offsets needed to get from start state to the goal as a `List`.\n')
# A* Search Code if checked
if st.checkbox("Show A* Search Code"):
    with st.echo():
        def a_star_search(world: List[List[str]], start: Tuple[int, int], goal: Tuple[int, int], costs: Dict[str, int], moves: List[Tuple[int, int]], heuristic: Callable) -> List[Tuple[int, int]]:
            frontier = PriorityQueue()
            frontier.put(start, 0)

            path = {start: None}
            g_n = {start: 0}
            while not frontier.empty():
                current_node = frontier.get()
                if current_node == goal:
                    final = [current_node]
                    while current_node in path:
                        current_node = path[current_node]
                        final.append(current_node)
                    final = final[::-1]
                    path_actions = [tuple(np.subtract(j, i)) for i, j in zip(final[1:], final[2:])]
                    final_path_actions = []
                    for action in path_actions:
                        reversed_action = action[::-1]
                        final_path_actions.append(reversed_action)
                    break
                neighbors = get_neighbors(current_node, moves, world)
                for neighbor in neighbors:

                    new_g = g_n[current_node] + costs.get(world[neighbor[0]][neighbor[1]])

                    if neighbor not in g_n or new_g < g_n[neighbor]:
                        g_n[neighbor] = new_g
                        priority = new_g + heuristic(neighbor, goal)
                        frontier.put(neighbor, priority)
                        path[neighbor] = current_node

            return final_path_actions
st.header("The World")
# Renders the Environment
st.markdown("```\n "
            "token   terrain    cost \n"
            "🌾       plains     1\n"
            "🌲       forest     3\n"
            "🪨       hills      5\n"
            "🐊       swamp      7\n"
            "🗻       mountains  impassible\n"
            "```")
if st.checkbox("Show/Hide"):
    st.write(print_environment(full_world))

# Sidebar
st.sidebar.header('Parameters')
# gets the user's input for the position interested in
def get_input():
    # Start Position
    start_x = st.sidebar.slider("Start X Position", 0, len(full_world)-1)
    start_y = st.sidebar.slider("Start Y Position", 0, len(full_world[0])-1)
    start = (start_x, start_y)
    # Goal Position
    goal_x = st.sidebar.slider("Goal X Position", 0, len(full_world)-1)
    goal_y = st.sidebar.slider("Goal Y Position", 0, len(full_world[0])-1)
    goal = (goal_x, goal_y)
    return start, goal


# Runs the event

start, goal = get_input()

st.sidebar.subheader('Start - %s %s Goal - %s %s' % (full_world[start[1]][start[0]],start,
                                             full_world[goal[1]][goal[0]], goal))
if st.sidebar.button('Run'):
    with st.spinner("Waiting..."):
        time.sleep(0.3)
    st.sidebar.success("Done!")
    st.subheader('Path Taken')
    # gets path
    path = a_star_search(full_world, start, goal, COSTS, MOVES, heuristic)
    # get path cost
    path_cost = pretty_print_solution(full_world, path, start, goal, COSTS)

    # Print Results
    st.sidebar.header('Results:')
    st.sidebar.subheader(f"Total Path Cost: {path_cost}")
    st.sidebar.markdown(f'Path: {path}')
# ---------------------------------------------------------------------------------------------------------------------
