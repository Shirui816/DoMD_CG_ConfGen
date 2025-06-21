import networkx as nx
import numba as nb
import numpy as np

from domd_cgbuilder.cg_mol import CGMol


@nb.jit(nopython=True, nogil=True)
def _cell_w(x, box, ib):
    ret = 0
    n_cell = 1
    for i in range(0, x.shape[0]):
        tmp = x[i] / box[i]
        if tmp < -0.5 or tmp > 0.5:
            return -1
        ret = ret + np.floor((tmp + 0.5) * ib[i]) * n_cell
        n_cell = n_cell * ib[i]
    return ret


@nb.jit(nopython=True, nogil=True)
def _cell_i(ix, ib):
    ret = 0
    n_cell = 1
    for i in range(0, ix.shape[0]):
        tmp = (ix[i] + ib[i]) % ib[i]
        ret = ret + tmp * n_cell
        n_cell = n_cell * ib[i]
    return ret


@nb.jit(nopython=True, nogil=True)
def pbc_dist(r, d):
    return np.sqrt(np.sum(r - d * np.rint(r / d)) ** 2)


@nb.jit(nopython=True, nogil=True)
def pbc(r, d):
    return r - d * np.rint(r / d)


class CellList(object):
    def __init__(self, box, rc):
        self.box = box
        self.rc = rc
        self.cells = {}
        self.cell_map = {}
        self.ib = np.asarray(box / rc, np.int64)
        self.dim = (self.box.shape[0],) * 3
        print("Building cell list...")
        self._build_cell_map()

    def _w_cell(self, x):
        return _cell_w(x, self.box, self.ib)

    def _i_cell(self, ix):
        return _cell_i(ix, self.ib)

    def _build_cell_map(self):
        for ix in np.ndindex(self.ib):
            ic = self._i_cell(np.asarray(ix))
            self.cells[ic] = []
            self.cell_map[ic] = {}
            for j in range(self.box.shape[0] ** 3):
                jc = np.asarray(np.unravel_index(j, self.dim)) - 1
                self.cell_map[ic][j] = self._i_cell(jc + ic)

    def add_x(self, x):
        ic = self._w_cell(x)
        self.cells[ic].append(x)

    def iter_neighbors(self, x):
        ic = self._w_cell(x)
        for jc in self.cell_map[ic]:  # include self-cell
            x_in_jc = self.cells[jc]
            for xj in x_in_jc:
                yield xj


def is_valid(x, cl, r=0.2):
    failed_p = False
    for nei in cl.iter_neighbors(x):
        if pbc_dist(x - nei) < r:
            failed_p = True
            return failed_p
    return failed_p


@nb.jit(nopython=True, nogil=True)
def _step_w(r, d=3):
    step = np.random.random(d) - 0.5
    step = step / np.sum(step ** 2) ** 0.5 * r
    return step


def embed_system(system: list[CGMol], box, rc=1):
    r"""Arbitrary configuration generator based on self-avoiding random walk method
    The system contains a series of molecules, i.e., nx.Graph objects
    Random walk coordinates are generated based on edges from edge_dfs, for each molecule,
    a node with given coordinate is treated as the source node for the dfs transversal.
    """
    cl = CellList(box, rc)
    for mol in system:
        s_node = None
        for bead in mol.nodes:
            if bead['x'] is not None:
                s_node = bead
        if s_node is not None:
            cl.add_x(mol.nodes[s_node]['x'])

    # 1st: Add all possible source node coordinates to the cell-list
    # the source node choice algorithm is as same as below, making sure
    # that even multiple nodes has given position, only the last is chosen

    for mol in system:
        s_node = None
        for bead in mol.nodes:
            if bead['x'] is not None:
                s_node = bead
        # the source bead is the bead with given coordinate
        # only one for each molecule, for Brownian bridge is
        # too f**king difficult for generating arbitrary graph.
        for edge in nx.edge_dfs(mol, source=s_node):
            ip, jp = edge
            if mol.nodes[ip]['x'] is None:
                xi = (np.random.random() - 0.5) * (box - 0.2)  # tolerance
                while not is_valid(xi, cl):
                    xi = (np.random.random() - 0.5) * (box - 0.2)
                cl.add_x(xi)
            else:
                # since xi is xj in the precursor edge, this should not be problem
                # even if the node has a given coordinate, it has been overwritten
                # by the precursor generation
                xi = mol.nodes[ip]['x']

            step_r = mol.nodes[ip]['r'] + mol.nodes[jp]['r']
            step_i = _step_w(step_r, d=box.shape[0])
            xj_raw = xi + step_i
            xj_img = pbc(xj_raw, box)
            # whether xj is valid
            while not is_valid(xj_img, cl):
                step_i = _step_w(step_r, d=box.shape[0])
                xj_raw = xi + step_i
                xj_img = pbc(xj_raw, box)
            mol.nodes[jp]['x'] = xj_img
            mol.nodes[jp]['img'] = np.asarray(xj_raw / (box / 2), dtype=np.int64)
            cl.add_x(xj_img)
    return system
