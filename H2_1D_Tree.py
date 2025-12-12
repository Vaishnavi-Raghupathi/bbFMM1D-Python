from H2_1D_Tree_Functions import *
from numpy import arange


class H2_1D_Tree:
    """H2_1D_TREE Class for the FMM Tree"""
    
    def __init__(self, nChebNodes, charge, location, N, m):
        """Initialize the FMM Tree"""
        self.nChebNodes = nChebNodes
        self.rank = nChebNodes
        self.N = N
        self.m = m
        self.maxLevels = 0
        self.chargeTree = charge
        self.locationTree = location
        
        self.cNode = get_Standard_Chebyshev_Nodes(self.nChebNodes)
        self.TNode = get_Standard_Chebyshev_Polynomials(self.nChebNodes, self.nChebNodes, self.cNode)
        self.R = get_Transfer(self.nChebNodes, self.cNode, self.TNode)
        
        self.center, self.radius = get_Center_Radius(location)
        
        self.root = H2_1D_Node(0, 0)
        self.root.nNeighbor = 0
        self.root.nInteraction = 0
        self.root.N = N
        self.root.center = self.center
        self.root.radius = self.radius
        self.root.index = arange(0, N)
        self.root.isRoot = True
        
        print('\n Assigning children...')
        assign_Children(self, self.root, self.R, nChebNodes, self.cNode, self.TNode)
        print(' Done.')
        
        build_Tree(self.root)
        
        print('\n Maximum levels is: %d' % self.maxLevels)
