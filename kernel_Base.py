import numpy as np

def calculate_Potential(custom_Kernel, tree, charges):
    potential = np.zeros([tree.N, tree.m])
    set_Tree_Potential_Zero(tree.root, tree.rank, tree.m)
    set_Node_Charge_Zero(tree.root, tree.rank, tree.m)
    tree.chargeTree = charges
    update_Charge(tree, tree.root)
    print('\nCalculating potential...')
    potential = calculate_Potential_Recursive(custom_Kernel, tree, tree.root, potential)
    print('Done.')
    return potential

def set_Tree_Potential_Zero(node, rank, m):
    if not node.isEmpty:
        node.potential = np.zeros([node.N, m])
        node.nodePotential = np.zeros([rank, m])
        for k in range(2):
            set_Tree_Potential_Zero(node.child[k], rank, m)

def set_Node_Charge_Zero(node, rank, m):
    if not node.isEmpty:
        node.chargeComputed = False
        node.charge = np.zeros([node.N, m])
        node.nodeCharge = np.zeros([rank, m])
        for k in range(2):
            set_Node_Charge_Zero(node.child[k], rank, m)

def update_Charge(tree, node):
    if node.isLeaf:
        get_Charge(tree, node)
        node.nodeCharge += np.dot(node.R.T, node.charge)
    else:
        for k in range(2):
            update_Charge(tree, node.child[k])
            if not node.child[k].isEmpty:
                node.nodeCharge += np.dot(tree.R[k, :, :].T, node.child[k].nodeCharge)

def get_Charge(tree, node):
    if not node.chargeComputed:
        node.chargeComputed = True
        node.charge = tree.chargeTree[node.index]

def calculate_Potential_Recursive(custom_Kernel, tree, node, potential):
    if not node.isEmpty:
        if node.isLeaf:
            if not node.isRoot:
                for k in range(len(node.neighbor)):
                    if not node.neighbor[k].isEmpty:
                        get_Charge(tree, node.neighbor[k])
                        node.potential += np.dot(
                            custom_Kernel(node.N, node.location, 
                                        node.neighbor[k].N, node.neighbor[k].location),
                            node.neighbor[k].charge)
                
                node.potential += np.dot(node.R, node.nodePotential)
                
                node.potential += np.dot(
                    custom_Kernel(node.N, node.location, node.N, node.location),
                    node.charge)
                
                potential = transfer_Potential_To_Tree(node, potential)
        else:
            computePotential = False
            for k in range(len(node.neighbor)):
                if not node.isRoot:
                    if not node.neighbor[k].isEmpty:
                        if node.neighbor[k].isLeaf:
                            get_Charge(tree, node.neighbor[k])
                            node.potential += np.dot(
                                custom_Kernel(node.N, node.location,
                                            node.neighbor[k].N, node.neighbor[k].location),
                                node.neighbor[k].charge)
                            computePotential = True
            
            calculate_NodePotential_M2L(custom_Kernel, node, tree.nChebNodes)
            transfer_NodePotential_L2L(node, tree.R)
            
            if computePotential:
                potential = transfer_Potential_To_Tree(node, potential)
            
            for k in range(2):
                potential = calculate_Potential_Recursive(custom_Kernel, tree, node.child[k], potential)
    
    return potential

def transfer_Potential_To_Tree(node, potential):
    potential[node.index, :] = potential[node.index, :] + node.potential[:node.N, :]
    return potential

def calculate_NodePotential_M2L(custom_Kernel, node, nChebNodes):
    for k in range(2):
        if not node.child[k].isEmpty:
            for i in range(node.child[k].nInteraction):
                if not node.child[k].interaction[i].isEmpty:
                    K = kernel_Cheb_1D(custom_Kernel, nChebNodes,
                                      node.child[k].scaledCnode,
                                      nChebNodes,
                                      node.child[k].interaction[i].scaledCnode)
                    node.child[k].nodePotential += np.dot(K, node.child[k].interaction[i].nodeCharge)

def kernel_Cheb_1D(custom_Kernel, M, xVec, N, yVec):
    if xVec.ndim == 1:
        xPoints = xVec[:M].reshape(-1, 1)
        yPoints = yVec[:N].reshape(-1, 1)
    else:
        xPoints = xVec[:M, :]
        yPoints = yVec[:N, :]
    K = custom_Kernel(M, xPoints, N, yPoints)
    return K

def transfer_NodePotential_L2L(node, R):
    for k in range(2):
        if not node.child[k].isEmpty:
            node.child[k].nodePotential += np.dot(R[k, :, :], node.nodePotential)