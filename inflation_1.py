import cv2
import numpy as np
import argparse
from scipy.sparse import coo_matrix, linalg

def depth2orthomesh(depth, x_step=1, y_step=1, scale=[1.0, 1.0, 1.0], minus_depth=True):
    vertices = []
    faces = []
    if len(depth.shape) != 2:
        return None
    h, w = depth.shape
    vertex_id = 0
    added_table = {}
    boundary_vertices = []
    for y in range(0, h, y_step):
        for x in range(0, w, x_step):
            d = depth[y, x]
            if d <= 0.000001:
                continue
            if minus_depth:
                d = -d

            vertices.append([x * scale[0], y * scale[1], d * scale[2]])

            added_table[(y, x)] = vertex_id
            # If this is an edge vertex, save it to boundary list
            if x==0 or y==0 or x==w-1 or y==h-1:
                boundary_vertices.append([x*scale[0], 0, d*scale[2]])

            upper_vertex_idx = added_table.get((y - y_step, x))
            left_vertex_idx = added_table.get((y, x - x_step))

            if upper_vertex_idx is not None:
                faces.append([upper_vertex_idx, vertex_id, left_vertex_idx])
            if left_vertex_idx is not None:
                faces.append([left_vertex_idx, vertex_id, upper_vertex_idx])

            vertex_id += 1

    # Now add the boundary vertices
    vertices = vertices + boundary_vertices

    # And now the boundary faces
    for face in faces:
        for i in range(len(face)-1):
            faces.append([face[i], face[i+1], face[i]+len(vertices)//2, face[i+1]+len(vertices)//2])
        faces.append([face[-1], face[0], face[-1]+len(vertices)//2, face[0]+len(vertices)//2])

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
def inflationByBaran(mask, use_sparse=True):
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
        row = np.array(row, dtype=np.int)
        col = np.array(col, dtype=np.int)
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

    for j in range(1, h-1):
        for i in range(1, w-1):
            c = mask[j, i]
            if c == 0:
                continue
            idx = img2param_idx[get_idx(i, j)]
            # setting z = √ h
            depth[j, i] = np.sqrt(x[idx])
    return depth


def visualizeDepth(depth, path='', dmin=0, dmax=50, cm_name='viridis'):
    import matplotlib.pyplot as plt
    cm = plt.get_cmap(cm_name)
    colors = (np.array(cm.colors) * 255).astype(np.uint8)
    colors = colors[..., ::-1] # -> BGR
    
    normed = np.clip((depth - dmin) / (dmax - dmin), 0, 1)
    normed = (normed * 255).astype(np.uint8)
    
    vis = colors[normed]
    if path != '':
        cv2.imwrite(path, vis)
    return vis

def mirror_vertices(vertices):
    mirrored_vertices = []
    for vertex in vertices:
        x, y, z = vertex
        mirrored_vertices.append([x, y, -z])  # Negate the x-coordinate to mirror along x-axis
    return mirrored_vertices

def mirror_faces(faces):
    mirrored_faces = []
    for face in faces:
        mirrored_faces.append(face[::-1])  # Reverse the order of vertices in the face
    return mirrored_faces

def combine_mesh(vertices, faces, mirrored_vertices, mirrored_faces):
    combined_vertices = vertices + mirrored_vertices
    offset = len(vertices)
    combined_faces = faces + [[idx+offset for idx in face] for face in mirrored_faces]
    return combined_vertices, combined_faces

def add_connecting_faces(vertices, mirrored_vertices, faces, w, h, x_step, y_step):
   
    h = h // y_step  # need to adjust grid size if we had step > 1 in depth2orthomesh
    w = w // x_step

    offset = len(vertices)  # index where mirrored vertices start
    
    # traverse the edge vertices row by row
    for y in range(h - 1):  # subtract 1 to avoid out of bounds
        org_idx_top = y * w
        org_idx_bot = (y + 1) * w
        mir_idx_top = org_idx_top + offset
        mir_idx_bot = org_idx_bot + offset
        # Make sure to add vertices in correct order (counter-clockwise) to get correct outward-facing normal
        faces.append([org_idx_top, org_idx_bot, mir_idx_bot])
        faces.append([mir_idx_bot, mir_idx_top, org_idx_top])

    # then traverse edge vertices column by column
    for x in range(w - 1):  # subtract 1 to avoid out of bounds
        org_idx_lef = h * w - 1 - x
        org_idx_rig = h * w - x
        mir_idx_lef = org_idx_lef + offset
        mir_idx_rig = org_idx_rig + offset
        faces.append([org_idx_lef, org_idx_rig, mir_idx_rig])
        faces.append([mir_idx_rig, mir_idx_lef, org_idx_lef])
        
    return faces

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Processing Script")

    parser.add_argument('--mask_path', type=str, help='Path to the masks')
    parser.add_argument('--output_path', type=str, help='Path to output the results')

    args = parser.parse_args()

    mask_path = args.mask_path
    output_path = args.output_path
    x_step = 1
    y_step = 1

    print(mask_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask[mask > 100] = 255
    mask[mask <= 100] = 0
    h, w = mask.shape

    
    depth = inflationByBaran(mask)
    print(output_path+'output_baran_.jpg')
    visualizeDepth(depth, output_path+'/output_baran_.jpg')
    vertices, faces = depth2orthomesh(depth, x_step, y_step)
    mirrored_faces = mirror_faces(faces)
    mirrored_vertices = mirror_vertices(vertices)
    writeMeshAsPly(output_path + 'output_baran.ply', vertices, faces)
    writeMeshAsPly(output_path + 'output_baran_mirror.ply', mirrored_vertices, mirrored_faces)

    # Get combined vertices and faces
    combined_vertices, combined_faces = combine_mesh(vertices, faces, mirrored_vertices, mirrored_faces)

    # Add the connecting faces
    #combined_faces = add_connecting_faces(vertices, mirrored_vertices, combined_faces, w, h, x_step, y_step)

    # Write combined mesh as Ply
    writeMeshAsPly(output_path + 'output_baran_combined.ply', combined_vertices, combined_faces)
