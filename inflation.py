import cv2
import numpy as np
import argparse
from scipy.sparse import coo_matrix, linalg
import scipy.ndimage as ndi

def depth2orthomesh(depth, x_step=1, y_step=1, scale=[1.0, 1.0, 1.0], minus_depth=True):
    vertices = []
    faces = []
    if len(depth.shape) != 2:
        return None
    h, w = depth.shape
    vertex_id = 0
    added_table = {}
    for y in range(0, h, y_step):
        for x in range(0, w, x_step):
            added_table[(y, x)] = -1
    max_connect_z_diff = 99999.9
    # TODO
    # pixel-wise loop in pure python is toooooooo slow
    for y in range(0, h, y_step):
        for x in range(0, w, x_step):
            d = depth[y, x]
            if d <= 0.000001:
                continue
            if minus_depth:
                d = -d

            vertices.append([x * scale[0], y * scale[1], d * scale[2]])

            added_table[(y, x)] = vertex_id

            current_index = vertex_id
            upper_left_index = added_table[((y - y_step), (x - x_step))]
            upper_index = added_table[((y - y_step), x)]
            left_index = added_table[(y, (x - x_step))]

            upper_left_diff = np.abs(depth[y - y_step, x - x_step] - d)
            upper_diff = np.abs(depth[y - y_step, x] - d)
            left_diff = np.abs(depth[y, x - x_step] - d)

            if upper_left_index > 0 and upper_index > 0\
               and upper_left_diff < max_connect_z_diff\
               and upper_diff < max_connect_z_diff:
                faces.append([upper_left_index, current_index, upper_index])

            if upper_left_index > 0 and left_index > 0\
                and upper_left_diff < max_connect_z_diff\
                    and left_diff < max_connect_z_diff:
                faces.append([upper_left_index, left_index, current_index])

            vertex_id += 1
    return vertices, faces


def _make_ply_txt(vertices, faces, color=[], normal=[]):
    header_lines = ["ply", "format ascii 1.0",
                    "element vertex " + str(len(vertices)),
                    "property float x", "property float y", "property float z"]
    has_normal = len(vertices) == len(normal)
    has_color = len(vertices) == len(color)
    if has_normal:
        header_lines += ["property float nx",
                         "property float ny", "property float nz"]
    if has_color:
        header_lines += ["property uchar red", "property uchar green",
                         "property uchar blue", "property uchar alpha"]
    # no face
    header_lines += ["element face " + str(len(faces)),
                     "property list uchar int vertex_indices", "end_header"]
    header = "\n".join(header_lines) + "\n"

    data_lines = []
    for i in range(len(vertices)):
        line = [vertices[i][0], vertices[i][1], vertices[i][2]]
        if has_normal:
            line += [normal[i][0], normal[i][1], normal[i][2]]
        if has_color:
            line += [int(color[i][0]), int(color[i][1]), int(color[i][2]), 255]
        line_txt = " ".join([str(x) for x in line])
        data_lines.append(line_txt)
    for f in faces:
        line_txt = " ".join(['3'] + [str(int(x)) for x in f])
        data_lines.append(line_txt)

    data_txt = "\n".join(data_lines)

    ply_txt = header + data_txt

    return ply_txt


def writeMeshAsPly(path, vertices, faces):
    with open(path, 'w') as f:
        txt = _make_ply_txt(vertices, faces)
        f.write(txt)


def inflationByDistanceTransform(mask, activation_func=None):
    dist = cv2.distanceTransform(mask, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    depth = dist
    if activation_func is None:
        return depth
    activated_depth = depth.copy()
    h, w = depth.shape
    for j in range(h):
        for i in range(w):
            #print(d, activation_func(d))
            activated_depth[j, i] = activation_func(depth[j, i])
    return activated_depth


def activation_tanh(factor):
    return lambda x: np.tanh(x * factor) / factor


# Implementation of the following paper
# "Notes on Inflating Curves" [Baran and Lehtinen 2009].
# http://alecjacobson.com/weblog/media/notes-on-inflating-curves-2009-baran.pdf
def inflationByBaran_old(mask, use_sparse=True):
    max_depth = 1
    h, w = mask.shape
    depth = np.zeros((h, w))
    img2param_idx = {}
    param_idx = 0

    def get_idx(x, y):
        return y * w + x

    for y in range(h):
        for x in range(w):
            c = mask[y, x]
            if c != 0:
                img2param_idx[get_idx(x, y)] = param_idx
                param_idx += 1
    num_param = len(img2param_idx.keys())
    triplets = []
    cur_row = 0
    # 4 neighbor laplacian
    for y in range(1, h-1):
        for x in range(1, w-1):
            c = mask[y, x]
            if c == 0:
                continue
            triplets.append([cur_row, img2param_idx[get_idx(x, y)], -4.0])
            kernels = [(y, x - 1), (y, x + 1), (y - 1, x), (y + 1, x)]
            for kernel in kernels:
                jj, ii = kernel
                if mask[jj, ii] != 0:
                    triplets.append([cur_row, img2param_idx[get_idx(ii, jj)], 1.0])
            cur_row += 1  # Go to the next equation
    # Prepare right hand side
    b = np.zeros((num_param, 1))
    rhs = -4.0
    cur_row = 0
    for y in range(1, h-1):
        for x in range(1, w-1):
            c = mask[y, x]
            if c == 0:
                continue
            b[cur_row] = rhs
            cur_row += 1
    if use_sparse:
        # Sparse matrix version
        data, row, col = [], [], []
        for tri in triplets:
            row.append(tri[0])
            col.append(tri[1])
            data.append(tri[2])
        data = np.array(data)
        row = np.array(row, dtype=int)
        col = np.array(col, dtype=int)
        A = coo_matrix((data, (row, col)), shape=(num_param, num_param))
        x = linalg.spsolve(A, b)
    else:
        # Dense matrix version
        A = np.zeros((num_param, num_param))
        # Set from triplets
        for tri in triplets:
            row = tri[0]
            col = tri[1]
            val = tri[2]
            A[row, col] = val
        x = np.linalg.solve(A, b)

    # Fills up the depth array with the computed depths
    z_min = min(x)
    print(z_min)
    for j in range(1, h-1):
        for i in range(1, w-1):
            c = mask[j, i]
            if c == 0:
                continue

            # Check for edge pixels
            if (mask[j - 1, i] == 0 or mask[j + 1, i] == 0 or mask[j, i - 1] == 0 or mask[j, i + 1] == 0 or mask[j-1, i - 1] == 0 or mask[j+1, i + 1] == 0 or mask[j-1, i + 1] == 0 or mask[j-1, i-1] == 0):# or  mask[j - 2, i] == 0 or mask[j + 2, i] == 0 or mask[j, i - 2] == 0 or mask[j, i + 2] == 0):
                # For edge pixels, assign full depth
                depth[j, i] = max_depth  # Where max_depth is a pre-defined value for the maximum depth
                continue

            idx = img2param_idx[get_idx(i, j)]
            # setting z = √ h
            if depth[j, i] != max_depth:
              depth[j, i] = np.sqrt(x[idx])-z_min
    print(np.amin(depth))


    return depth


def distance_to_edge(mask):
    """
    Compute the Euclidean distance for each mesh pixel to the nearest edge pixel
    """
    distance_transform = ndi.distance_transform_edt(mask)
    return distance_transform

    # Compute distance of every pixel to the nearest edge in the mask
    distances = distance_to_edge(mask)

    # Fills up the depth array with the computed depths
    for j in range(1, h-1):
        for i in range(1, w-1):
            c = mask[j, i]
            if c == 0:
                continue

            idx = img2param_idx[get_idx(i, j)]
            # setting z = √ h
            if x[idx] != 0:
                depth[j, i] = np.sqrt(x[idx])
            else:
                depth[j, i] = x[idx]
            
            # Check if this pixel is within r distance from edge
            if distances[j, i] < r:
                # Interpolate depth based on distance to edge
                depth[j, i] *= (distances[j, i] / r)

    # Returns the depth map
    return depth


from scipy.sparse import coo_matrix
import numpy as np
from scipy.sparse.linalg import spsolve


def inflationByBaran(mask, use_sparse=True):
    """Function to create a depth image from a mask."""
    
    #1. 4 neighbor laplacian
    img2param_idx = {}
    param_idx = 0
    h, w = mask.shape
    depth = np.zeros((h, w)) # Depth array

    def get_idx(x, y):
        return y * w + x

    for y in range(h):
        for x in range(w):
            c = mask[y, x]
            if c != 0:
                img2param_idx[get_idx(x, y)] = param_idx
                param_idx += 1

    num_param = len(img2param_idx.keys())
    triplets = []

    cur_row = 0
    for y in range(1, h-1):
        for x in range(1, w-1):
            c = mask[y, x]
            if c == 0:
                continue

            triplets.append([cur_row, img2param_idx[get_idx(x, y)], -4.0])
            kernels = [(y, x - 1), (y, x + 1), (y - 1, x), (y + 1, x)]

            for kernel in kernels:
                jj, ii = kernel
                if mask[jj, ii] != 0:
                    triplets.append([cur_row, img2param_idx[get_idx(ii, jj)], 1.0])

            cur_row += 1

    #2. Prepare right hand side
    b = np.zeros((num_param, 1))
    rhs = -4.0
    cur_row = 0
    for y in range(1, h-1):
        for x in range(1, w-1):
            c = mask[y, x]
            if c == 0:
                continue
            b[cur_row] = rhs
            cur_row += 1

    #3. Sparse matrix
    data, row, col = [], [], []
    for tri in triplets:
        row.append(tri[0])
        col.append(tri[1])
        data.append(tri[2])

    # Creating the sparse matrix using data, row, and col
    data = np.array(data)
    row = np.array(row, dtype=int)
    col = np.array(col, dtype=int)
    A = coo_matrix((data, (row, col)), shape=(num_param, num_param))

    # Solve for 'x' using scipy's sparse linear solver
    x = linalg.spsolve(A, b)

     # Fills up the depth array with the computed depths
    z_min = min(x)
    max_depth = 1

    for j in range(h):
        for i in range(w):
            # Skip vertices not in mask
            if get_idx(i, j) not in img2param_idx:
                continue
            idx = img2param_idx[get_idx(i, j)]

            # If the vertex is at the edge of the mask, set the depth to max_depth
            if (mask[j - 1, i] == 0 or mask[j + 1, i] == 0 or 
                mask[j, i - 1] == 0 or mask[j, i + 1] == 0 or
                mask[j-1, i-1] == 0 or mask[j+1, i+1] == 0 or 
                mask[j-1, i+1] == 0 or mask[j-1, i-1] == 0):
                    depth[j, i] = max_depth
            else:
                # Otherwise, set the depth to the solution of the Poisson's equation
                depth[j, i] = np.sqrt(x[idx]) - z_min

    return depth

def inflate_mesh(mask_path, output_path):

    # Read mask of mesh
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask[mask > 100] = 255
    mask[mask <= 100] = 0
    
    # Infer depth
    depth = inflationByBaran(mask)

    # Apply depth to mesh
    vertices, faces = depth2orthomesh(depth)

    #Export mesh
    writeMeshAsPly(output_path + 'inflated_mesh.ply', vertices, faces)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Processing Script")

    parser.add_argument('--mask_path', type=str, default="data/character.png", help='Path to the masks')
    parser.add_argument('--output_path', type=str, default="output/", help='Path to output the results')

    args = parser.parse_args()

    mask_path = args.mask_path
    output_path = args.output_path


    # Read mask of mesh
    inflate_mesh(mask_path, output_path)
