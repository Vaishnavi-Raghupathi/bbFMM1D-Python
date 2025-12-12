
from numpy import zeros, array
# -------------------------------------------------------------------------------------------------------------------- #
class H2_1D_Node:
    """H2_1D_Node Class for the nodes of the FMM Tree"""
    isEmpty = True
    isLeaf = True
    
    def __init__(self, nLevel, nodeNumber):
        """Constructor initializes each node of the FMM Tree"""
        # In 1D, each node has at most 2 neighbors (left and right)
        self.neighbor = array([H2_1D_Node for count in range(2)])
        
        # In 1D, interaction list includes well-separated nodes
        # For a node at level k, this includes nodes that are children of neighbors
        # of the parent but not neighbors themselves (max 3 nodes in 1D)
        self.interaction = array([H2_1D_Node for count in range(3)])
        
        # Center of the interval (scalar in 1D)
        self.center = 0.0
        
        # Charge information
        self.charge = 0
        self.chargeComputed = False
        
        # In 1D, each node has 2 children (left and right subintervals)
        self.child = array([H2_1D_Node for count in range(2)])
        
        # Indices of particles in this node
        self.index = array([], dtype=int)
        
        # Node status flags
        self.isEmpty = True
        self.isLeaf = True
        self.isRoot = False
        
        # Location (scalar in 1D)
        self.location = 0.0
        
        # Number of particles in this node
        self.N = 0
        
        # Number of nodes in interaction list
        self.nInteraction = 0
        
        # Level in the tree
        self.nLevel = nLevel
        
        # Number of neighbors
        self.nNeighbor = 0
        
        # Node number at this level
        self.nodeNumber = nodeNumber
        
        # Charges at Chebyshev nodes (multipole expansion coefficients)
        self.nodeCharge = zeros([])
        
        # Potential at Chebyshev nodes (local expansion coefficients)
        self.nodePotential = zeros([])
        
        # Parent node
        self.parent = H2_1D_Node
        
        # Computed potential values
        self.potential = zeros((1, 1))
        
        # SVD-related matrices for compression
        self.R = zeros([])
        
        # Radius (half-length) of the interval
        self.radius = 0.0
        
        # Scaled Chebyshev nodes for this interval
        self.scaledCnode = zeros([])
        
        # Left and right boundaries of the interval
        self.left_boundary = 0.0
        self.right_boundary = 0.0
