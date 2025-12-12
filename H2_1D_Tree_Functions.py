from math import pi
from numpy import zeros, ones, dot, append, cos, arange
from H2_1D_Node import H2_1D_Node


def get_Standard_Chebyshev_Nodes(nChebNodes):
    """Obtains standard Chebyshev nodes in interval [-1,1]"""
    return cos((arange(0, nChebNodes) + 0.5) * pi / nChebNodes).reshape(nChebNodes, 1)


def get_Standard_Chebyshev_Polynomials(nChebPoly, N, x):
    """Computes Chebyshev polynomials at given points"""
    T = zeros((N, nChebPoly))
    T[:, 0] = 1.0
    if nChebPoly > 1:
        T[:, 1] = x.flatten()
        for k in range(2, nChebPoly):
            T[:, k] = 2.0 * x.flatten() * T[:, k - 1] - T[:, k - 2]
    return T


def get_Transfer(nChebNodes, cNode, TNode):
    """Evaluates transfer from two children to parent"""
    S = get_Transfer_From_Parent_CNode_To_Children_CNode(nChebNodes, cNode, TNode)
    
    Transfer = zeros((2, nChebNodes, nChebNodes))
    Transfer[0] = S[0:nChebNodes, 0:nChebNodes]
    Transfer[1] = S[nChebNodes:2*nChebNodes, 0:nChebNodes]
    
    R = zeros((2, nChebNodes, nChebNodes))
    R[0] = Transfer[0]
    R[1] = Transfer[1]
    
    return R


def get_Transfer_From_Parent_CNode_To_Children_CNode(nChebNodes, cNode, TNode):
    """Interpolation from parent Chebyshev nodes to children Chebyshev nodes"""
    childcNode = zeros((2 * nChebNodes, 1))
    childcNode[0:nChebNodes, 0] = 0.5 * (cNode.flatten() - 1)
    childcNode[nChebNodes:2*nChebNodes, 0] = 0.5 * (cNode.flatten() + 1)
    
    Transfer = get_Standard_Chebyshev_Polynomials(nChebNodes, 2 * nChebNodes, childcNode)
    Transfer = (2.0 * dot(Transfer, TNode.T) - 1) / nChebNodes
    
    return Transfer


def get_Center_Radius(location):
    """Computes center and radius of smallest interval containing data"""
    maxX = location.max()
    minX = location.min()
    
    center = 0.5 * (maxX + minX)
    radius = 0.5 * (maxX - minX)
    
    return center, radius


def assign_Children(Tree, node, R, nChebNodes, cNode, TNode):
    """Assigns children to the given node"""
    if node.N == 0:
        node.isLeaf = True
        node.isEmpty = True
        return
    
    node.potential = zeros((node.N, Tree.m))
    node.nodePotential = zeros((Tree.rank, Tree.m))
    node.nodeCharge = zeros((Tree.rank, Tree.m))
    node.isEmpty = False
    node.isLeaf = False
    node.location = zeros(node.N)
    
    get_Scaled_ChebNode(node, cNode)
    
    for k in range(node.N):
        node.location[k] = Tree.locationTree[node.index[k]]
    
    # Leaf cell operation
    if node.N <= 2 * Tree.rank:
        node.isLeaf = True
        node.R = get_Transfer_From_Parent_To_Children(node.N, nChebNodes, node.location, 
                                                      node.center, node.radius, TNode)
        if Tree.maxLevels < node.nLevel:
            Tree.maxLevels = node.nLevel
    else:
        # Create two children
        for k in range(2):
            node.child[k] = H2_1D_Node(node.nLevel + 1, k)
            node.child[k].parent = node
            
            # Left child (k=0) or right child (k=1)
            node.child[k].center = node.center + (k - 0.5) * node.radius
            node.child[k].radius = node.radius * 0.5
            node.child[k].N = 0
        
        # Assign particles to children
        for k in range(node.N):
            if Tree.locationTree[node.index[k]] < node.center:
                node.child[0].index = append(node.child[0].index, node.index[k])
                node.child[0].N += 1
            else:
                node.child[1].index = append(node.child[1].index, node.index[k])
                node.child[1].N += 1
        
        # Recursively assign children
        for k in range(2):
            assign_Children(Tree, node.child[k], R, nChebNodes, cNode, TNode)


def get_Scaled_ChebNode(node, cNode):
    """Evaluates scaled Chebyshev nodes in node's interval"""
    node.scaledCnode = node.center + node.radius * cNode.flatten()


def get_Transfer_From_Parent_To_Children(N, nChebNodes, location, center, radius, TNode):
    """Interpolation from Chebyshev nodes to particle locations"""
    standLocation = (location - center) / radius
    standLocation = standLocation.reshape(-1, 1)
    
    Transfer = get_Standard_Chebyshev_Polynomials(nChebNodes, N, standLocation)
    Transfer = (2.0 * Transfer.dot(TNode.T) - 1) / nChebNodes
    
    return Transfer


def build_Tree(node):
    """Builds the FMM Tree"""
    if node.isEmpty:
        return
    
    if not node.isLeaf:
        # Initialize neighbor arrays for children
        for i in range(2):
            node.child[i].neighbor = [H2_1D_Node(0, 0) for _ in range(2)]
        
        assign_Siblings(node)
        
        # Assign cousins from parent's neighbors
        for k in range(2):
            if not node.neighbor[k].isLeaf and not node.neighbor[k].isEmpty:
                assign_Cousin(node, k)
        
        # Recursively build tree for children
        for k in range(2):
            build_Tree(node.child[k])


def assign_Siblings(node):
    """Assign siblings to children of the node"""
    # Child 0 (left) has child 1 (right) as neighbor
    node.child[0].neighbor[1] = node.child[1]
    node.child[0].nNeighbor += 1
    
    # Child 1 (right) has child 0 (left) as neighbor
    node.child[1].neighbor[0] = node.child[0]
    node.child[1].nNeighbor += 1


def assign_Cousin(node, neighborNumber):
    """Assign cousins to children of the node"""
    
    if neighborNumber == 0 and not node.neighbor[0].isEmpty:
        # Left neighbor's children
        # Child 0: far cousin goes to interaction list, near cousin is neighbor
        node.child[0].interaction[node.child[0].nInteraction] = node.neighbor[0].child[0]
        node.child[0].neighbor[0] = node.neighbor[0].child[1]
        node.child[0].nInteraction += 1
        node.child[0].nNeighbor += 1
        
        # Child 1: both cousins go to interaction list
        node.child[1].interaction[node.child[1].nInteraction] = node.neighbor[0].child[0]
        node.child[1].interaction[node.child[1].nInteraction + 1] = node.neighbor[0].child[1]
        node.child[1].nInteraction += 2
    
    elif neighborNumber == 1 and not node.neighbor[1].isEmpty:
        # Right neighbor's children
        # Child 0: both cousins go to interaction list
        node.child[0].interaction[node.child[0].nInteraction] = node.neighbor[1].child[0]
        node.child[0].interaction[node.child[0].nInteraction + 1] = node.neighbor[1].child[1]
        node.child[0].nInteraction += 2
        
        # Child 1: near cousin is neighbor, far cousin goes to interaction list
        node.child[1].neighbor[1] = node.neighbor[1].child[0]
        node.child[1].interaction[node.child[1].nInteraction] = node.neighbor[1].child[1]
        node.child[1].nInteraction += 1
        node.child[1].nNeighbor += 1
