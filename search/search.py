# search.py
# ---------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


"""
In search.py, you will implement generic search algorithms which are called by
Pacman agents (in searchAgents.py).
"""
import random

import util


# Defining a class node which will help  me implement the alg in a much easier way
class CustomNode:
    def __init__(self, parent, action, state):
        self.parent = parent
        self.action = action
        self.state = state

    def getParent(self):
        return self.parent
    def getAction(self):
        return self.action
    def getState(self):
        return self.state


class CustomNodeAStar(CustomNode):
    def __init__(self, parent, action, state, cost, eval):
        CustomNode.__init__(self, parent, action, state)
        self.cost = cost
        self.eval = eval


    def getCost(self):
        return self.cost
    def getEval(self):
        return self.eval


class CustomNodeUniform(CustomNode):
    def __init__(self, parent, action, state, cost):
        CustomNode.__init__(self, parent=parent, action=action, state=state)
        self.cost = cost
    def getCost(self):
        return self.cost



class SearchProblem:
    """
    This class outlines the structure of a search problem, but doesn't implement
    any of the methods (in object-oriented terminology: an abstract class).

    You do not need to change anything in this class, ever.
    """

    def getStartState(self):
        """
        Returns the start state for the search problem.
        """
        util.raiseNotDefined()

    def isGoalState(self, state):
        """
          state: Search state

        Returns True if and only if the state is a valid goal state.
        """
        util.raiseNotDefined()

    def getSuccessors(self, state):
        """
          state: Search state

        For a given state, this should return a list of triples, (successor,
        action, stepCost), where 'successor' is a successor to the current
        state, 'action' is the action required to get there, and 'stepCost' is
        the incremental cost of expanding to that successor.
        """
        util.raiseNotDefined()

    def getCostOfActions(self, actions):
        """
         actions: A list of actions to take

        This method returns the total cost of a particular sequence of actions.
        The sequence must be composed of legal moves.
        """
        util.raiseNotDefined()


def tinyMazeSearch(problem):
    """
    Returns a sequence of moves that solves tinyMaze.  For any other maze, the
    sequence of moves will be incorrect, so only use this for tinyMaze.
    """
    from game import Directions
    s = Directions.SOUTH
    w = Directions.WEST
    return  [s, s, w, s, w, w, s, w]



def getPath(node):
    path = []
    while node.getAction() != None:
        path.insert(0, node.getAction())
        node = node.getParent()
    return path


def randomSearch(problem):
    current = problem.getStartState()
    solution = []
    while (not (problem.isGoalState(current))):
        succ = problem.getSuccessors(current)
        no_of_successors = len(succ)
        random_succ_index = int(random.random() * no_of_successors)
        next = succ[random_succ_index]
        current = next[0]
        solution.append(next[1])
    print "The solution is ", solution
    return solution


# DFS implementation for pacman
def depthFirstSearch(problem):

    visited = dict() #used to keep track of visited nodes
    state = problem.getStartState()
    stack = util.Stack() # we will need a stack in order to implement the algorithm
    node = CustomNode(None, None, state) # CustomNode was design only for DFS, BFS
    stack.push(node)

    while not stack.isEmpty():
        # getting the next node in the stack
        node = stack.pop()
        state = node.getState()

        # if is visited we try something else
        if visited.has_key(hash(state)):
            continue

        # else if is not visited we assign it as visited
        visited[hash(state)] = True

        # if we end up in the goal state, we return the path to it
        if problem.isGoalState(state) == True:
            return getPath(node)

        for child in problem.getSuccessors(state):
            if not visited.has_key(hash(child[0])):
                # child[0] position/state of the next node
                # child[1] direction
                # we create a new node
                nextNode = CustomNode(node, child[1], child[0])
                # we add it to the stack
                stack.push(nextNode)
    return [] # if it doesn't find any path we will return an empty list



# BFS implementation for pacman
def breadthFirstSearch(problem):
    visited = dict()
    # getting the initial possition of the pacman
    state = problem.getStartState()

    queue = util.Queue() # we will need a queue in order to implement the algorithm
    node = CustomNode(None, None, state) # CustomNode was design only for DFS, BFS
    queue.push(node)

    while not queue.isEmpty():
        # getting the next node in the queue
        node = queue.pop()
        state = node.getState()

        # if is visited we try something else
        if visited.has_key(state):
            continue

        # else if is not visited we assign it as visited
        visited[state] = True

        # if we end up in the goal state, we return the path to it
        if problem.isGoalState(state) == True:
            return getPath(node)

        for child in problem.getSuccessors(state):
            if not visited.has_key(hash(child[0])):
                # child[0] position/state of the next node
                # child[1] direction
                # we create a new node
                nextNode = CustomNode(node, child[1], child[0])
                # we add it to the queue
                queue.push(nextNode)

    return [] # if it doesn't find any path we will return an empty list

# Uniform cost search implementation for pacman
def uniformCostSearch(problem):
    visited = dict()
    state = problem.getStartState()
    # This algorithm uses a Priority Queue
    queue = util.PriorityQueue()

    # CustomNodeUnform was designed for this tipe of problem
    # It inherits from CustomNode
    node = CustomNodeUniform(parent=None, action=None, state=state, cost=0)
    queue.push(node, node.getCost())

    while not queue.isEmpty():
        node = queue.pop()
        state = node.getState()
        cost = node.getCost()

        if visited.has_key(state):
            continue
        visited[state] = True
        # we find the point
        if problem.isGoalState(state) == True:
            return getPath(node)

        for child in problem.getSuccessors(state):
            if not visited.has_key(hash(child[0])):
                # child[2] = cost
                # UNIFORM-COST SEARCH expands the node n with the lowest path cost g(n)
                nextNode = CustomNodeUniform(node, child[1], child[0], cost= (cost + child[2]))
                queue.push(nextNode, nextNode.getCost())

    return []  # if it doesn't find any path we will return an empty list





def nullHeuristic(state, problem=None):
    """
    A heuristic function estimates the cost from the current state to the nearest
    goal in the provided SearchProblem.  This heuristic is trivial.
    """
    return 0
# A* implementation for pacman
# The big difference between A* an UC is that, A* has a brain
def aStarSearch(problem, heuristic=nullHeuristic):

    # this algo. also uses a PriorityQueue
    priorityQueue = util.PriorityQueue()
    visited = dict()

    state = problem.getStartState()
    # CustomNodeAStar was designed for this tipe of problem
    # It inherits from CustomNodeAStar
    node = CustomNodeAStar(None, None, state, 0, heuristic(state, problem))
    priorityQueue.push(node, node.getCost() + node.getEval())

    while not priorityQueue.isEmpty():
        node = priorityQueue.pop()
        state = node.getState()
        cost = node.getCost()

        # if is visited we try something else
        if visited.has_key(state):
            continue

        # else -> make it visited
        visited[state] = True
        if problem.isGoalState(state) == True:
            return getPath(node)


        for child in problem.getSuccessors(state):
            if not visited.has_key(child[0]):
                # createing the node, using the formula f = g + h(estimated cost from state to goal)
                nextNode = CustomNodeAStar(parent=node, action=child[1], state=child[0], cost= (child[2] + cost), eval=heuristic(child[0], problem))
                priorityQueue.push(nextNode, nextNode.getCost() + nextNode.getEval())

    return []  # if it doesn't find any path we will return an empty list


# Abbreviations
bfs = breadthFirstSearch
dfs = depthFirstSearch
astar = aStarSearch
ucs = uniformCostSearch
rs = randomSearch