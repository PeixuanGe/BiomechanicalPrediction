import os
import numpy as np
import math

import torch

from data.dataset import stiffness_mat2arr, strength_mat2arr

'''def euler_rotation_matrix(phi1, Phi, phi2):
    phi1 = math.radians(phi1)
    Phi = math.radians(Phi)
    phi2 = math.radians(phi2)
    cos_phi1 = math.cos(phi1)
    cos_phi2 = math.cos(phi2)
    cos_Phi = math.cos(Phi)
    sin_phi1 = math.sin(phi1)
    sin_phi2 = math.sin(phi2)
    sin_Phi = math.sin(Phi)
    a11 = cos_phi1 * cos_phi2 - cos_Phi * sin_phi1 * sin_phi2
    a12 = cos_Phi * cos_phi1 * sin_phi2 + sin_phi1 * cos_phi2
    a13 = sin_Phi * sin_phi2
    a21 = -cos_phi1 * sin_phi2 - cos_Phi * sin_phi1 * cos_phi2
    a22 = -cos_Phi * cos_phi1 * cos_phi2 - sin_phi1 * sin_phi2
    a23 = sin_Phi * cos_phi2
    a31 = sin_Phi * sin_phi1
    a32 = -sin_Phi * cos_phi1
    a33 = cos_Phi
    return np.array([[a11, a12, a13],
                     [a21, a22, a23],
                     [a31, a32, a33]])'''


def get_rotation_matrix(x_angle, y_angle, z_angle):
    rx = torch.tensor([[1, 0, 0],
                       [0, torch.cos(torch.deg2rad(torch.tensor(x_angle))),
                        -torch.sin(torch.deg2rad(torch.tensor(x_angle)))],
                       [0, torch.sin(torch.deg2rad(torch.tensor(x_angle))),
                        torch.cos(torch.deg2rad(torch.tensor(x_angle)))]])
    ry = torch.tensor(
        [[torch.cos(torch.deg2rad(torch.tensor(y_angle))), 0, torch.sin(torch.deg2rad(torch.tensor(y_angle)))],
         [0, 1, 0],
         [-torch.sin(torch.deg2rad(torch.tensor(y_angle))), 0, torch.cos(torch.deg2rad(torch.tensor(y_angle)))]],
    )
    rz = torch.tensor(
        [[torch.cos(torch.deg2rad(torch.tensor(z_angle))), -torch.sin(torch.deg2rad(torch.tensor(z_angle))), 0],
         [torch.sin(torch.deg2rad(torch.tensor(z_angle))), torch.cos(torch.deg2rad(torch.tensor(z_angle))), 0],
         [0, 0, 1]])
    rotation_matrix = torch.matmul(rz, torch.matmul(ry, rx))
    rotation_matrix = rotation_matrix.numpy()
    return rotation_matrix


def T_sigma_matrix(euler_matrix):
    R11, R12, R13 = euler_matrix[0]
    R21, R22, R23 = euler_matrix[1]
    R31, R32, R33 = euler_matrix[2]
    return np.array([[R11 ** 2, R21 ** 2, R31 ** 2, 2 * R11 * R21, 2 * R21 * R31, 2 * R31 * R11],
                     [R12 ** 2, R22 ** 2, R32 ** 2, 2 * R12 * R22, 2 * R22 * R32, 2 * R32 * R12],
                     [R13 ** 2, R23 ** 2, R33 ** 2, 2 * R13 * R23, 2 * R23 * R33, 2 * R33 * R13],
                     [R11 * R12, R21 * R22, R31 * R32, R11 * R22 + R12 * R21, R21 * R32 + R31 * R22,
                      R31 * R12 + R32 * R11],
                     [R12 * R13, R22 * R23, R33 * R32, R23 * R12 + R13 * R22, R22 * R33 + R32 * R23,
                      R32 * R13 + R12 * R33],
                     [R11 * R13, R23 * R21, R33 * R31, R13 * R21 + R11 * R23, R23 * R31 + R21 * R33,
                      R33 * R11 + R31 * R13]])


def T_eps_matrix(euler_matrix):
    R11, R12, R13 = euler_matrix[0]
    R21, R22, R23 = euler_matrix[1]
    R31, R32, R33 = euler_matrix[2]
    return np.array([[R11 ** 2, R21 ** 2, R31 ** 2, R11 * R21, R21 * R31, R31 * R11],
                     [R12 ** 2, R22 ** 2, R32 ** 2, R12 * R22, R22 * R32, R32 * R12],
                     [R13 ** 2, R23 ** 2, R33 ** 2, R13 * R23, R23 * R33, R33 * R13],
                     [2 * R11 * R12, 2 * R21 * R22, 2 * R31 * R32, R11 * R22 + R12 * R21, R21 * R32 + R31 * R22,
                      R31 * R12 + R32 * R11],
                     [2 * R12 * R13, 2 * R22 * R23, 2 * R33 * R32, R23 * R12 + R13 * R22, R22 * R33 + R32 * R23,
                      R32 * R13 + R12 * R33],
                     [2 * R11 * R13, 2 * R23 * R21, 2 * R33 * R31, R13 * R21 + R11 * R23, R23 * R31 + R21 * R33,
                      R33 * R11 + R31 * R13]])


def stiffness_force_symmetric(mat):
    mat = np.array(mat)
    c12 = (mat[0][1] + mat[1][0]) / 2
    c13 = (mat[0][2] + mat[2][0]) / 2
    c23 = (mat[1][2] + mat[2][1]) / 2
    mat[0][1] = c12
    mat[1][0] = c12
    mat[0][2] = c13
    mat[2][0] = c13
    mat[1][2] = c23
    mat[2][1] = c23
    return mat


def rotate_tensor(tensor, rotation_matrix, is_stiffness=True):
    T_matrix = T_sigma_matrix(rotation_matrix) if is_stiffness else rotation_matrix
    T_matrix_inv = np.transpose(T_matrix)
    if is_stiffness:
        T_matrix[abs(T_matrix) < 1e-8] = 0
        T_matrix_inv[abs(T_matrix_inv) < 1e-8] = 0
        tensor[abs(tensor) < 1e-8] = 0
        rotated_tensor = T_matrix_inv @ tensor @ T_matrix
        # return stiffness_force_symmetric(rotated_tensor) if is_stiffness else np.abs(rotated_tensor)
        res = stiffness_force_symmetric(rotated_tensor)
        res[abs(res) < 1e-8] = 0
        return res
    else:
        rotated_tensor = T_matrix_inv @ tensor @ T_matrix
        return np.abs(rotated_tensor)


def generate_rotated_tensors(base_folder, output_folder, rotation_angles):
    property_folders = ['1_properties', '2_properties', '3_properties', '4_properties',
                        '1_rand_properties', '2_rand_properties', '3_rand_properties', '4_rand_properties']

    for folder in property_folders:
        full_path = os.path.join(base_folder, folder)
        for file in os.listdir(full_path):
            if file.endswith('_C_norm.txt'):
                stiffness_mat = np.loadtxt(os.path.join(full_path, file))
                stiffness_mat = stiffness_force_symmetric(stiffness_mat)
                base_name = os.path.splitext(file)[0]

                for angles in rotation_angles:
                    # rotation_matrix = euler_rotation_matrix(*angles)
                    rotation_matrix = get_rotation_matrix(*angles)
                    rotated_stiffness = rotate_tensor(stiffness_mat, rotation_matrix, is_stiffness=True)
                    output_subfolder = os.path.join(output_folder, folder)
                    if not os.path.exists(output_subfolder):
                        os.makedirs(output_subfolder)
                    output_filename = f"{base_name}-Rotated-{angles[0]}-{angles[1]}-{angles[2]}-C.txt"
                    np.savetxt(os.path.join(output_subfolder, output_filename), rotated_stiffness)

            elif file.endswith('_Strength_norm.txt'):
                strength_mat = np.loadtxt(os.path.join(full_path, file))
                base_name = os.path.splitext(file)[0]

                for angles in rotation_angles:
                    # rotation_matrix = euler_rotation_matrix(*angles)
                    rotation_matrix = get_rotation_matrix(*angles)
                    rotated_strength = rotate_tensor(strength_mat, rotation_matrix, is_stiffness=False)
                    output_subfolder = os.path.join(output_folder, folder)
                    if not os.path.exists(output_subfolder):
                        os.makedirs(output_subfolder)
                    output_filename = f"{base_name}-Rotated-{angles[0]}-{angles[1]}-{angles[2]}-Strength.txt"
                    np.savetxt(os.path.join(output_subfolder, output_filename), rotated_strength)


def load_tensors(base_folder='./rotations_FEA/'):
    tensors_dict_c = {}
    tensors_dict_strength = {}
    property_folders = ['1_properties', '2_properties', '3_properties', '4_properties',
                        '1_rand_properties', '2_rand_properties', '3_rand_properties', '4_rand_properties']

    for folder in property_folders:
        full_path = os.path.join(base_folder, folder)
        folder = folder.replace('_properties', '')
        tensors_dict_c[folder] = {}
        tensors_dict_strength[folder] = {}
        for file in os.listdir(full_path):
            if file.endswith('-C.txt'):
                base_name = os.path.splitext(file)[0]
                parts = base_name.split('-')
                sample_index = parts[0].replace('_C_norm', '')
                angle_str = parts[2:-1]
                angle_tuple = tuple(map(int, angle_str))
                if sample_index not in tensors_dict_c[folder]:
                    tensors_dict_c[folder][sample_index] = {}
                if angle_tuple not in tensors_dict_c[folder][sample_index]:
                    tensors_dict_c[folder][sample_index][angle_tuple] = {}
                mat = np.loadtxt(os.path.join(full_path, file))
                stiffness_mat = stiffness_mat2arr(mat)
                stiffness_mat = torch.from_numpy(stiffness_mat)
                tensors_dict_c[folder][sample_index][angle_tuple] = stiffness_mat
            if file.endswith('-Strength.txt'):
                base_name = os.path.splitext(file)[0]
                parts = base_name.split('-')
                sample_index = parts[0].replace('_Strength_norm', '')
                angle_str = parts[2:-1]
                angle_tuple = tuple(map(int, angle_str))
                if sample_index not in tensors_dict_strength[folder]:
                    tensors_dict_strength[folder][sample_index] = {}
                if angle_tuple not in tensors_dict_strength[folder][sample_index]:
                    tensors_dict_strength[folder][sample_index][angle_tuple] = {}
                tensor_data = np.loadtxt(os.path.join(full_path, file))
                strength_mat = strength_mat2arr(tensor_data)
                strength_mat = torch.from_numpy(strength_mat)
                tensors_dict_strength[folder][sample_index][angle_tuple] = strength_mat
    return tensors_dict_c, tensors_dict_strength


# Physics based rotation augmentation
if __name__ == '__main__':
    # Example usage
    rotation_angles = [
        (0, 0, 90), (0, 0, 180), (0, 0, 270),
        (0, 90, 0), (0, 90, 90), (0, 90, 180), (0, 90, 270),
        (0, 180, 0), (0, 180, 90), (0, 180, 180), (0, 180, 270),
        (0, 270, 0), (0, 270, 90), (0, 270, 180), (0, 270, 270),
        (90, 0, 0), (90, 0, 90), (90, 0, 180), (90, 0, 270),
        (270, 0, 0), (270, 0, 90), (270, 0, 180), (270, 0, 270)]

    base_folder = './dataset/'
    output_folder = './rotations_FEA/'

    # Generate and store rotated tensors
    generate_rotated_tensors(base_folder, output_folder, rotation_angles)

    # Load tensors into a dictionary
    tensors_dict_c, tensors_dict_strength = load_tensors(output_folder)
    import pickle


    def save_tensors_dict(tensors_dict, file_path):
        with open(file_path, 'wb') as file:
            pickle.dump(tensors_dict, file)


    save_path = 'dataset/tensors_dict_c.pkl'
    save_tensors_dict(tensors_dict_c, save_path)
    save_path = 'dataset/tensors_dict_strength.pkl'
    save_tensors_dict(tensors_dict_strength, save_path)


    def load_tensors_dict(file_path):
        with open(file_path, 'rb') as file:
            tensors_dict = pickle.load(file)
        return tensors_dict


    load_path = 'dataset/tensors_dict_c.pkl'
    tensors_dict_c = load_tensors_dict(load_path)
    load_path = 'dataset/tensors_dict_strength.pkl'
    tensors_dict_strength = load_tensors_dict(load_path)
    # Access example
    '''sample_index = '0_0_0'
    angle_tuple = (90, 0, 0)
    C_norm = tensors_dict_c['1_properties'][sample_index][angle_tuple]
    Strength_norm = tensors_dict_c['1_properties'][sample_index][angle_tuple]'''
