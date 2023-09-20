import os
import cv2
import numpy as np
import argparse
from scipy.sparse import coo_matrix, linalg
from scipy.sparse.linalg import spsolve
import scipy.ndimage as ndi
import trimesh

def dither_image(image_path):
    # Load grayscale image using cv2
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    # Apply dithering using openCV
    image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    image = cv2.GaussianBlur(image, (0,0), sigmaX=3, sigmaY=3, borderType = cv2.BORDER_DEFAULT)
    image = cv2.addWeighted(image, 1.5, image, -0.5, 0, image)

    # Convert back to single channel
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Save the result
    cv2.imwrite('output/pillow/depth_dithered.png', image)

def calculate_normal(y, x, scale, depth, radius=1):
    scale_inv = [1/s for s in scale]

    # Apply boundary conditions.
    left = max(0, x - radius)
    right = min(depth.shape[1] - 1, x + radius)
    top = max(0, y - radius)
    bottom = min(depth.shape[0] - 1, y + radius)

    # Calculate derivatives.
    dzdx = (depth[y, right] - depth[y, left]) * scale_inv[2] / scale_inv[0] / (2.0 * radius)
    dzdy = (depth[bottom, x] - depth[top, x]) * scale_inv[2] / scale_inv[1] / (2.0 * radius)

    # Determine the length of the vector [dzdx, dzdy, 1].
    d = np.sqrt(dzdx * dzdx + dzdy * dzdy + 1)

    return [-dzdx / d, -dzdy / d, 1 / d]

def depth2orthomesh(depth, color_img_path, x_step=1, y_step=1, scale=[1.0, 1.0, 1.0], minus_depth=True, displacement_factor=.1):
    
    # Load the color image
    color_img = cv2.imread(color_img_path)
    color_img = cv2.cvtColor(color_img, cv2.COLOR_BGR2RGB)

    vertices = []
    faces = []
    colors = []

    if len(depth.shape) != 2:
        return None

    h, w = depth.shape
    vertex_id = 0
    added_table = {}
    
    # Add a border of 'invalid' pixels to the depth image (Any significant value can be taken for depth)
    depth = cv2.copyMakeBorder(depth, top=1, bottom=1, left=1, right=1, borderType=cv2.BORDER_CONSTANT, value=0)

    for y in range(1, h + 1, y_step):
        for x in range(1, w + 1, x_step):
            added_table[(y, x)] = -1

    max_connect_z_diff = 99999.9

    for y in range(1, h + 1, y_step):
        for x in range(1, w + 1, x_step):

            d = depth[y, x]
            if d <= 0.0001:
                continue
            if minus_depth:
                d = -d

            norm_vector = calculate_normal(y, x, scale, depth)
            vertices.append([
                (x - 1) * scale[0] + norm_vector[0]*displacement_factor,
                (y - 1) * scale[1] + norm_vector[1]*displacement_factor,
                d * scale[2] + norm_vector[2]*displacement_factor])
            
            added_table[(y, x)] = vertex_id

            # Get color from color image
            color = color_img[y - 1, x - 1]
            colors.append(color)
            
            current_index = vertex_id
            upper_left_index = added_table[(y - y_step, x - x_step)]
            upper_index = added_table[(y - y_step, x)]
            left_index = added_table[(y, x - x_step)]
            
            upper_left_diff = np.abs(depth[y - y_step, x - x_step] - d)
            upper_diff = np.abs(depth[y - y_step, x] - d)
            left_diff = np.abs(depth[y, x - x_step] - d)
            
            if upper_left_index > 0 and upper_index > 0 and upper_left_diff < max_connect_z_diff and upper_diff < max_connect_z_diff:
                faces.append([upper_left_index, current_index, upper_index])
            
            if upper_left_index > 0 and left_index > 0 and upper_left_diff < max_connect_z_diff and left_diff < max_connect_z_diff:
                faces.append([upper_left_index, left_index, current_index])
                
            vertex_id += 1
                
    print(vertices)
    for i in range(len(vertices)):
        vertices[i][2] = 0 if abs(vertices[i][2]) < 1 else vertices[i][2]

    evertices = {}
    for i in range(len(vertices)):
        if (vertices[i][2] == 0):
            evertices[i] = 1

    facesx = []
    for i in range(len(faces)):
        a = vertices[faces[i][0]]
        b = vertices[faces[i][1]]
        c = vertices[faces[i][2]]
        if a+b+c != 0:
            facesx.append(faces[i])

    faces = facesx
    vertices2 = []
    faces2 = []
    colors2 = []
    for i in range(0,len(vertices)):

        vertices2.append([vertices[i][0],vertices[i][1],-vertices[i][2]])
        colors2.append(colors[i])

    offset = len(vertices)
    def getv(i):
        if i in evertices:
            return i
        return i + offset

    for i in range(0,len(faces)):

        faces2.append([getv(faces[i][0]),getv(faces[i][2]),getv(faces[i][1])])
    
    
    return vertices+vertices2, faces+faces2, colors2


def _make_ply_txt(vertices, faces, color=[]):
    header_lines = ["ply", "format ascii 1.0",
                    "element vertex " + str(len(vertices)),
                    "property float x", "property float y", "property float z"]
    
    header_lines += ["property uchar red", "property uchar green",
                     "property uchar blue", "property uchar alpha"]
    # no face
    header_lines += ["element face " + str(len(faces)),
                     "property list uchar int vertex_indices", "end_header"]
    header = "\n".join(header_lines) + "\n"

    data_lines = []
    for i in range(len(vertices)):
        line = [vertices[i][0], vertices[i][1], vertices[i][2]]
        
        line += [int(color[i][0]), int(color[i][1]), int(color[i][2]), 255]
        line_txt = " ".join([str(x) for x in line])
        data_lines.append(line_txt)

    for f in faces:
        line_txt = " ".join(['3'] + [str(int(x)) for x in f])
        data_lines.append(line_txt)

    ply_txt = header + "\n".join(data_lines)

    return ply_txt


def writeMeshAsPly(path, vertices, faces, colors):
    with open(path, 'w') as f:
        txt = _make_ply_txt(vertices, faces, colors)
        f.write(txt)

def write_obj_file(path, vertices, colors, faces):

    # Didn't see RGB normalization in your code. If colors are not in [0-1], uncomment this.
    # colors = colors / 255.0

    with open(path, 'w') as f:
        for v in vertices:
            f.write(f'v {v[0]} {v[1]} {v[2]}\n')
        for c in colors:
            f.write(f'vt {c[0]} {c[1]} {c[2]}\n')
        for face in faces:
            f.write(f'f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n')


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
    x = linalg.spsolve(A.astype(np.float64), b.astype(np.float64))

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
                mask[j-1, i+1] == 0 or mask[j+1, i-1] == 0):
                    depth[j, i] = max_depth
            else:
                # Otherwise, set the depth to the solution of the Poisson's equation
                depth[j, i] = np.sqrt(x[idx]) #- z_min

    return depth


def displace_depth_by_color(depth_img_path, mask, depth, alpha=0.5):
    # read the grayscale image
    #gray_img = cv2.imread(depth_img_path, cv2.IMREAD_GRAYSCALE)
    # Read the image as 16-bit
    gray_img = cv2.imread(depth_img_path, flags=cv2.IMREAD_ANYDEPTH)
    
    # Normalize the image
    gray_img = cv2.normalize(gray_img, None, 0, 65535, cv2.NORM_MINMAX)
    #gray_img = cv2.GaussianBlur(gray_img, (15, 15), 0) # add guassian blur to remove big jumps gradient values

    # mask the image
    gray_img[mask <= 100] = 0
    
    # normalize the pixel values to range [0,1]
    normalized_img = cv2.normalize(gray_img.astype('float64'), None, alpha=0.0, beta=1.0, norm_type=cv2.NORM_MINMAX)
    
    # Displace depth based on pixel values
    displaced_depth = ((1 - alpha) * depth) + (alpha * normalized_img * depth)
    
    return displaced_depth


def write_obj_file(obj_filename, vertices, faces, texture_filename):
    """
    Writes provided vertices and faces to an .obj file with UVs and material.

    Parameters
    ----------
    vertices : ndarray
        A numpy array of vertices.
    faces : list
        A list of faces.
    texture_filename : str
        Name of the texture image file.
    obj_filename : str
        Name of the output .obj file. Default is 'output.obj'.
    """
    image = cv2.imread(texture_filename, cv2.IMREAD_UNCHANGED)
    h, w = image.shape[:2]

    # Each vertex matches with a pixel, and we will map this one-to-one relation into UV
    uv_coordinates = [(v[0]/(w-1), 1.0 - v[1]/(h-1)) for v in vertices]  # We invert y axis because texture origin (0,0) is top-left 

    with open(obj_filename, 'w') as f:
        f.write(f"mtllib {texture_filename.split('.')[0]}.mtl\n")
        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")
        for uv in uv_coordinates:
            f.write(f"vt {uv[0]} {uv[1]}\n")
        f.write("usemtl Material\n")
        for face in faces:
            f.write(f"f {face[0]+1}/{face[0]+1} {face[1]+1}/{face[1]+1} {face[2]+1}/{face[2]+1}\n")

def write_mtl_file(mtl_filename, texture_filename):
    """
    Writes a basic .mtl file with given texture image filename sample.

    Parameters
    ----------
    texture_filename : str
        Name of the texture image file.
    mtl_filename : str, optional
        Name of the output .mtl file (should match the filename referenced 
        in the .obj file). Default is 'material.mtl'.
    """
    with open(mtl_filename, 'w') as f:
        f.write("newmtl Material\n")
        f.write(f"map_Kd {texture_filename}\n")


def inflate_mesh(mask_path, color_img_path, output_path, depth_map_path, apply_depth_map, max_depth, depth_map_weight):

    # Read mask of mesh
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    mask[mask > 100] = 255
    mask[mask <= 100] = 0
    
    # Infer depth
    depth = inflationByBaran(mask)
    #depth = inflationByDistanceTransform(mask)
    #depth = depth * distance_transform

    # Apply depthmap
    if apply_depth_map:
        print("Applying depthmap")
        depth = displace_depth_by_color(depth_map_path, mask, depth, depth_map_weight)

    depth = depth * max_depth

    # Apply depth to mesh
    vertices, faces, colors = depth2orthomesh(depth, color_img_path)

    #Export mesh
    #writeMeshAsPly(output_path + 'inflated_mesh.ply', vertices, faces, colors)
    write_obj_file(os.path.join(output_path,'inflated_mesh.obj'), vertices, faces, color_img_path)
    write_mtl_file(os.path.join(output_path,'inflated_mesh.mtl'), color_img_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Image Processing Script")

    parser.add_argument('--mask_path', type=str, default="output/vase/mask.png", help='Path to the masks')
    parser.add_argument('--color_img_path', type=str, default="output/vase/colormap.png", help='Path to the original image')
    parser.add_argument('--depth_map_path', type=str, default="output/vase/depth.png", help='Path to the original image')
    parser.add_argument('--output_path', type=str, default="output/vase/", help='Path to output the results')

    args = parser.parse_args()

    mask_path = args.mask_path
    color_img_path = args.color_img_path
    depth_map_path = args.depth_map_path
    output_path = args.output_path

    # Read mask of mesh
    inflate_mesh(mask_path, color_img_path, output_path, depth_map_path, apply_depth_map=False, max_depth=1, depth_map_weight=1)
