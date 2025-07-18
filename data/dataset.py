import os
import pickle
import random
from random import choice
import SimpleITK as sitk
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import csv
import pandas as pd
import numpy as np

'''
train&val samples: 608
test samples: 155


All 
CT min: 0.0 max: 3000.0(clipped) mean: 452.3854615715561 std: 641.5060619703477
stiffness min: 0.0 max: 0.32242288841313094 mean: 0.011653846616045732 std: 0.028019400993397635
strength min: 1.9630532997931976e-07 max: 0.0010990439902784906 mean: 0.00012440705382158322 std: 0.0001170220222307021
Tb.Th Mean  min: 0.172993891 max: 0.48745107 mean: 0.2678543871164484 std: 0.051487155832346115
Tb.Sp Mean  min: 0.46356614 max: 4.3293899 mean: 1.0549755854046579 std: 0.5221222943879448
BV/TV  min: 0.06157589 max: 0.544764996 mean: 0.25502733025618635 std: 0.08345706104715979
DA  min: 0.16468553 max: 0.757569405 mean: 0.5587298216841339 std: 0.12941319403867707
Conn.D  min: 0.567479404 max: 10.12285014 mean: 3.2658990725327515 std: 1.3789522452040346
SMI  min: 0.851756335 max: 4.667962104 mean: 2.850510699705968 std: 0.6493118681669655
BS  min: 32.60905501 max: 347.9403078 mean: 196.11393315045123 std: 61.433780692476375


Train&val data
stiffness min: 0.0 max: 0.32242288841313094 mean: 0.012107197028304153 std: 0.029485700668542912
strength min: 2.3939490534279042e-08 max: 0.0010990439902784906 mean: 0.00012853335014441903 std: 0.000124054749297332
Tb.Th Mean  min: 0.161531552 max: 0.467923394 mean: 0.26627217518914476 std: 0.05090218061115085
Tb.Sp Mean  min: 0.477783459 max: 4.122957548 mean: 1.0394956539210527 std: 0.5066010115124502
BV/TV  min: 0.077074528 max: 0.538970947 mean: 0.25621584136184206 std: 0.08570657456051273
DA  min: 0.154175786 max: 0.757569405 mean: 0.5535084261973684 std: 0.12937180607338153
Conn.D  min: 0.567479404 max: 10.12285014 mean: 3.433072652577303 std: 1.4919315953651533
SMI  min: 0.851756335 max: 5.031733504 mean: 2.868076372429276 std: 0.6714801506768617
BS  min: 32.52402118 max: 344.4461411 mean: 196.2272750545395 std: 63.63460934683689

'''
# min, max
TRAIN_DATA_RANGE = {
    'Stiffness Tensor': [0, 0.32242288841313094],
    'Strength Tensor': [2.3939490534279042e-08, 0.0010990439902784906],
    'Tb.Th Mean': [0.161531552, 0.467923394],  #mm
    'Tb.Sp Mean': [0.477783459, 4.122957548],  #mm
    'BV/TV': [0, 1],
    'DA': [0, 1],
    'Conn.D': [0.567479404, 10.12285014],  #mm^-3
    'SMI': [0.851756335, 5.031733504],
    'BS': [32.52402118, 347.4461411],  #mm^2
    'Radius a': [10, 70],
    'Radius b': [10, 70],
    'Radius c': [10, 70],
}
# mean, std
TRAIN_DATASET_MEAN_STD = {
    'Stiffness Tensor': [0.010984674279043923, 0.02692031179791194],
    'Strength Tensor': [0.00011764780790906127, 0.00011531734009925966],
    'Tb.Th Mean': [277.2146802324324, 56.17158436754184],
    'Tb.Sp Mean': [1068.4475027408785, 494.7334262266329],
    'BV/TV': [0.24916554054054055, 0.08338329132389243],
    'DA': [0.5629917603074325, 0.12475040327004197],
    'Conn.D': [3.28439527027027, 1.3325732964392747],
    'SMI': [2.9283378378378377, 0.6933897450700421],
    'BS': [192.33813851351348, 61.77828674056167]
}


def standardization(data, data_type):
    return (data - TRAIN_DATASET_MEAN_STD[data_type][0]) / TRAIN_DATASET_MEAN_STD[data_type][1]


def standardization_inv(data, data_type):
    return data * TRAIN_DATASET_MEAN_STD[data_type][1] + TRAIN_DATASET_MEAN_STD[data_type][0]


CT_MIN = 0
CT_MAX = 3000  # human bone dense part HU value is around 2000, 3000 is in case for metal implants


def CT_norm(data):
    data = np.clip(data, CT_MIN, CT_MAX)
    data = (data - data.min()) / (data.max() - data.min())
    return data


def standardization_key(props):
    res = {}
    for key in props.keys():
        res[key] = standardization(props[key], key)
    return res


def standardization_key_inv(props):
    res = {}
    for key in props.keys():
        res[key] = standardization_inv(props[key], key)
    return res


def normalization(data, data_type):
    return (data - TRAIN_DATA_RANGE[data_type][0]) / (TRAIN_DATA_RANGE[data_type][1] - TRAIN_DATA_RANGE[data_type][0])


def normalization_inv(data, data_type):
    return data * (TRAIN_DATA_RANGE[data_type][1] - TRAIN_DATA_RANGE[data_type][0]) + TRAIN_DATA_RANGE[data_type][0]


def normalization_key(props):
    for key in props.keys():
        props[key] = normalization(props[key], key)
    return props


def normalization_key_inv(props):
    for key in props.keys():
        props[key] = normalization_inv(props[key], key)
    return props


PROP_KEYS = ['Tb.Th Mean', 'Tb.Sp Mean', 'BV/TV', 'DA', 'Conn.D', 'SMI', 'BS']


def stiffness_mat2arr(mat):  #use the mean of both part as stiffness is considered as symmetric
    mat = stiffness_force_symmetric(mat)
    C11, C12, C13 = mat[0][0], mat[0][1], mat[0][2]
    C21, C22, C23 = mat[1][0], mat[1][1], mat[1][2]
    C31, C32, C33 = mat[2][0], mat[2][1], mat[2][2]
    C44, C55, C66 = mat[3][3], mat[4][4], mat[5][5]
    return np.array([C11, C12, C13,
                     C22, C23,
                     C33,
                     C44, C55, C66])


stiffness_names = [
    'C11', 'C12', 'C13',
    'C22', 'C23',
    'C33',
    'C44', 'C55', 'C66',
]
all_tasks = {
    'Properties': len(PROP_KEYS),  # 7,
    'Stiffness Tensor': 9,
    'Strength Tensor': 6,
}


def stiffness_force_symmetric(mat):
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


def strength_mat2arr(mat):
    S11, S12, S31, S22, S23, S33 = mat[0][0], mat[0][1], mat[0][2], mat[1][1], mat[1][2], mat[2][2]
    return np.array([S11, S12, S31, S22, S23, S33])


strength_names = [
    'S11', 'S12', 'S31', 'S22', 'S23', 'S33'
]


def prop_mapping(prop):
    res = []
    for key in PROP_KEYS:
        res.append(prop[key])
    res = torch.stack(res, dim=1)
    return res


# prop has N x len(PROP_KEYS) shape
def prop_unmapping(prop):
    res = {}
    for i in range(len(PROP_KEYS)):
        temp = []
        for items in prop:
            temp.append(items[i])
        res[PROP_KEYS[i]] = torch.Tensor(temp)
    return res


spacing = (0.03585899993777275, 0.03585899993777275, 0.03585899993777275)

'''Stiffness = np.array([[C11, C12, C13, 0, 0, 0],
              [C21, C22, C23, 0, 0, 0],
              [C31, C32, C33, 0, 0, 0],
              [0, 0, 0, C44, 0, 0],
              [0, 0, 0, 0, C55, 0],
              [0, 0, 0, 0, 0, C66]])

Strength = np.array([[S11, S12, S31],
                     [S12, S22, S23],
                     [S31, S23, S33]])'''


def process_all_items(data, norm_fun, norm_type):
    for folder in data.keys():
        for sample_index in data[folder].keys():
            for angle_tuple in data[folder][sample_index].keys():
                data[folder][sample_index][angle_tuple] = norm_fun(data[folder][sample_index][angle_tuple], norm_type)
    return data


def generate_rotation_matrices_with_angles(device='cuda'):
    """
    Generate rotation matrices for the 24 unique cubic rotations using PyTorch, along with their angles.

    Returns:
    - rotation_dict: Dictionary where keys are tuples of angles (x_angle, y_angle, z_angle) in degrees,
                     and values are the corresponding rotation matrices.
    """
    # angles = [0, 90, 180, 270]
    rotation_dict = {}

    # Predefined unique rotations for a cube
    unique_rotations = [  # (0, 0, 0),
        (0, 0, 90), (0, 0, 180), (0, 0, 270),
        (0, 90, 0), (0, 90, 90), (0, 90, 180), (0, 90, 270),
        (0, 180, 0), (0, 180, 90), (0, 180, 180), (0, 180, 270),
        (0, 270, 0), (0, 270, 90), (0, 270, 180), (0, 270, 270),
        (90, 0, 0), (90, 0, 90), (90, 0, 180), (90, 0, 270),
        (270, 0, 0), (270, 0, 90), (270, 0, 180), (270, 0, 270)
    ]

    for x_angle, y_angle, z_angle in unique_rotations:
        rx = torch.tensor([[1, 0, 0],
                           [0, torch.cos(torch.deg2rad(torch.tensor(x_angle))),
                            -torch.sin(torch.deg2rad(torch.tensor(x_angle)))],
                           [0, torch.sin(torch.deg2rad(torch.tensor(x_angle))),
                            torch.cos(torch.deg2rad(torch.tensor(x_angle)))]], device=device)
        ry = torch.tensor(
            [[torch.cos(torch.deg2rad(torch.tensor(y_angle))), 0, torch.sin(torch.deg2rad(torch.tensor(y_angle)))],
             [0, 1, 0],
             [-torch.sin(torch.deg2rad(torch.tensor(y_angle))), 0, torch.cos(torch.deg2rad(torch.tensor(y_angle)))]],
            device=device)
        rz = torch.tensor(
            [[torch.cos(torch.deg2rad(torch.tensor(z_angle))), -torch.sin(torch.deg2rad(torch.tensor(z_angle))), 0],
             [torch.sin(torch.deg2rad(torch.tensor(z_angle))), torch.cos(torch.deg2rad(torch.tensor(z_angle))), 0],
             [0, 0, 1]], device=device)
        rotation_matrix = torch.matmul(rz, torch.matmul(ry, rx))
        rotation_matrix[abs(rotation_matrix) < 1e-6] = 0
        rotation_dict[(x_angle, y_angle, z_angle)] = rotation_matrix

    return rotation_dict


def euler_transform(ct_scan, rotation_matrix):
    """
    Apply an affine transformation to a 3D CT scan using a given rotation matrix.

    Parameters:
    - ct_scan: torch.Tensor of shape (1, 1, D, H, W), the input CT scan.
    - rotation_matrix: torch.Tensor of shape (3, 3), the rotation matrix.

    Returns:
    - rotated_ct: torch.Tensor, the rotated CT scan.
    """
    # d, h, w = ct_scan.shape[2], ct_scan.shape[3], ct_scan.shape[4]
    # identity_grid = F.affine_grid(torch.eye(3, 4).unsqueeze(0), ct_scan.size(), align_corners=True).to(ct_scan.device)

    rotation_matrix_4x4 = torch.eye(4, device=rotation_matrix.device)
    rotation_matrix_4x4[:3, :3] = rotation_matrix
    rotation_matrix_4x4 = rotation_matrix_4x4[:3, :]
    grid = F.affine_grid(rotation_matrix_4x4.unsqueeze(0), ct_scan.size(), align_corners=True)

    rotated_ct = F.grid_sample(ct_scan, grid.double(), mode='nearest', align_corners=True)
    return rotated_ct


class MicroBoneDataset(Dataset):
    def __init__(self,
                 path,
                 property_file,
                 split_path,
                 transform=None,
                 data_prep='norm',  # or 'standardize'
                 load_cache=True,
                 load_rotated_augmentations=False,
                 random_rotation_prob=0.,
                 prefix='./data/dataset/'
                 ):
        assert data_prep in ['norm', 'standardize']
        self.transform = transform
        self.CT_data = []
        if load_cache:
            self.CT_data = torch.load(prefix + split_path.split('/')[-1].replace('.csv', '.pt'), weights_only=True)
        self.stiffness_mats = []
        self.strength_mats = []
        self.names = []
        self.split_path = split_path
        with open(split_path, newline='') as split:
            reader = csv.reader(split)
            for file in reader:
                file = file[0]
                self.names.append(file)
                folder = file.split('__')[0]
                file = file.split('__')[1]
                if not load_cache:
                    img = sitk.GetArrayFromImage(sitk.ReadImage(path + '/' + folder + '/' + file + '.nii.gz'))
                    img = np.array(img)
                    img = CT_norm(img)
                    self.CT_data.append(torch.from_numpy(img))
                stiffness_mat = np.loadtxt(path + '/' + folder + '_properties/' + file + '_C_norm.txt')
                stiffness_mat = stiffness_mat2arr(stiffness_mat)
                stiffness_mat = torch.from_numpy(stiffness_mat)

                strength_mat = np.loadtxt(path + '/' + folder + '_properties/' + file + '_Strength_norm.txt')
                strength_mat = strength_mat2arr(strength_mat)
                strength_mat = torch.from_numpy(strength_mat)

                self.stiffness_mats.append(stiffness_mat)
                self.strength_mats.append(strength_mat)
        self.stiffness_mats = torch.stack(self.stiffness_mats)
        self.strength_mats = torch.stack(self.strength_mats)
        csv_data = pd.read_csv(property_file).set_index('FileName').loc[
            [name.replace('__', '_seg/') + '.nii.gz' for name in self.names]].to_dict('list')
        # mapping
        self.properties = {key: torch.tensor(csv_data[key]) for key in PROP_KEYS}

        if data_prep == 'norm':
            self.stiffness_mats = normalization(self.stiffness_mats, 'Stiffness Tensor')
            self.strength_mats = normalization(self.strength_mats, 'Strength Tensor')
            self.properties = normalization_key(self.properties)
        elif data_prep == 'standardize':
            self.stiffness_mats = standardization(self.stiffness_mats, 'Stiffness Tensor')
            self.strength_mats = standardization(self.strength_mats, 'Strength Tensor')
            self.properties = standardization_key(self.properties)

        self.properties = prop_mapping(self.properties)  # dict to tensor

        self.total_pred_var = len(self.stiffness_mats[0]) + len(self.strength_mats[0]) + len(
            PROP_KEYS)
        print('Total number of variables is:', self.total_pred_var)

        self.rotation_data = None
        self.random_rotation_prob = random_rotation_prob
        if load_rotated_augmentations:
            tensors_dict_c, tensors_dict_strength = self.get_rotation_data(prefix)
            if data_prep == 'norm':
                tensors_dict_c = process_all_items(tensors_dict_c, normalization, 'Stiffness Tensor')
                tensors_dict_strength = process_all_items(tensors_dict_strength, normalization, 'Strength Tensor')
            elif data_prep == 'standardize':
                tensors_dict_c = process_all_items(tensors_dict_c, standardization, 'Stiffness Tensor')
                tensors_dict_strength = process_all_items(tensors_dict_strength, standardization, 'Strength Tensor')
            self.rotation_data = {
                'Stiffness': tensors_dict_c,
                'Strength': tensors_dict_strength,
            }
            self.rotation_dict = generate_rotation_matrices_with_angles(device='cpu')

    def get_rotation_data(self, prefix='./data/'):
        file_path = prefix + 'tensors_dict_c.pkl'
        with open(file_path, 'rb') as file:
            tensors_dict_c = pickle.load(file)
        file_path = prefix + 'tensors_dict_strength.pkl'
        with open(file_path, 'rb') as file:
            tensors_dict_strength = pickle.load(file)
        return tensors_dict_c, tensors_dict_strength

    def get_pred_var_num(self):
        return self.total_pred_var

    def save_cache(self):
        torch.save(torch.stack(self.CT_data), self.split_path.split('/')[-1].replace('.csv', '.pt'))

    def set_rotation_prob(self, prob):
        self.random_rotation_prob = prob

    def __len__(self):
        return len(self.CT_data)

    def __getitem__(self, idx):
        ct = self.CT_data[idx].unsqueeze(0)
        prop = self.properties[idx]
        name = self.names[idx]
        if self.rotation_data is not None and self.random_rotation_prob != 0 and random.random() < self.random_rotation_prob:
            ct, stiffness_mat, strength_mat = self.CT_random_rotation(ct, name)
        else:
            stiffness_mat = self.stiffness_mats[idx]
            strength_mat = self.strength_mats[idx]
        return ct, stiffness_mat, strength_mat, prop, name

    def CT_random_rotation(self, ct, name):
        # Extract folder and file name
        folder, file_name = name.split('__')

        # Generate a random rotation angle that is not (0, 0, 0)
        angles = choice(list(self.rotation_dict.keys()))

        # Rotate the CT scan
        rotated_ct = euler_transform(ct.unsqueeze(0), self.rotation_dict[angles])
        rotated_ct = rotated_ct.squeeze(0)

        # Fetch the corresponding rotated properties
        rotated_stiffness_mat = self.rotation_data['Stiffness'][folder][file_name][angles]
        rotated_strength_mat = self.rotation_data['Strength'][folder][file_name][angles]

        return rotated_ct, rotated_stiffness_mat, rotated_strength_mat


def identity_func(x): return x


base_shape = 128


# turn off align_corners to match tio case
def down_2x_up_batch(x):
    # Downsample by 2
    downsampled = F.interpolate(x, size=[base_shape // 2, base_shape // 2, base_shape // 2], mode='trilinear')
    # Upsample back to original size
    upsampled = F.interpolate(downsampled, size=[base_shape, base_shape, base_shape], mode='trilinear')
    return upsampled


def down_4x_up_batch(x):
    # Downsample by 4
    downsampled = F.interpolate(x, size=[base_shape // 4, base_shape // 4, base_shape // 4], mode='trilinear')
    # Upsample back to original size
    upsampled = F.interpolate(downsampled, size=[base_shape, base_shape, base_shape], mode='trilinear')
    return upsampled


def down_8x_up_batch(x):
    # Downsample by 8
    downsampled = F.interpolate(x, size=[base_shape // 8, base_shape // 8, base_shape // 8], mode='trilinear')
    # Upsample back to original size
    upsampled = F.interpolate(downsampled, size=[base_shape, base_shape, base_shape], mode='trilinear')
    return upsampled


def down_16x_up_batch(x):
    # Downsample by 16
    downsampled = F.interpolate(x, size=[base_shape // 16, base_shape // 16, base_shape // 16], mode='trilinear')
    # Upsample back to original size
    upsampled = F.interpolate(downsampled, size=[base_shape, base_shape, base_shape], mode='trilinear')
    return upsampled


def full_res(x):  #placeholder
    return x


def down_6x_up_batch(x):
    # Downsample by 4
    downsampled = F.interpolate(x, size=[base_shape // 6, base_shape // 6, base_shape // 6], mode='trilinear')
    # Upsample back to original size
    upsampled = F.interpolate(downsampled, size=[base_shape, base_shape, base_shape], mode='trilinear')
    return upsampled


def down_10x_up_batch(x):
    # Downsample by 4
    downsampled = F.interpolate(x, size=[base_shape // 10, base_shape // 10, base_shape // 10], mode='trilinear')
    # Upsample back to original size
    upsampled = F.interpolate(downsampled, size=[base_shape, base_shape, base_shape], mode='trilinear')
    return upsampled


def down_12x_up_batch(x):
    # Downsample by 4
    downsampled = F.interpolate(x, size=[base_shape // 12, base_shape // 12, base_shape // 12], mode='trilinear')
    # Upsample back to original size
    upsampled = F.interpolate(downsampled, size=[base_shape, base_shape, base_shape], mode='trilinear')
    return upsampled


def down_14x_up_batch(x):
    # Downsample by 4
    downsampled = F.interpolate(x, size=[base_shape // 14, base_shape // 14, base_shape // 14], mode='trilinear')
    # Upsample back to original size
    upsampled = F.interpolate(downsampled, size=[base_shape, base_shape, base_shape], mode='trilinear')
    return upsampled


class AugmentedRepeatDataset(Dataset):  # dataset permutation for all downsamples
    def __init__(self, original_dataset,
                 noise_add=identity_func,
                 augmentations=None):
        super().__init__()
        if augmentations is None:
            augmentations = [full_res,
                             down_2x_up_batch,
                             down_4x_up_batch,
                             down_8x_up_batch,
                             down_16x_up_batch, ]
        self.noise_add = noise_add
        self.original_dataset = original_dataset
        self.augmentations = augmentations
        self.repeats = len(augmentations)

    def __len__(self):
        # Extend the length to account for repeats and augmentations
        return len(self.original_dataset) * self.repeats

    def __getitem__(self, idx):
        # Calculate the original index and the augmentation index
        original_idx = idx // self.repeats
        aug_idx = idx % self.repeats

        # Get the original data and label
        ct, stiffness_mat, strength_mat, prop, name = self.original_dataset[original_idx]
        ct = ct.unsqueeze(0)
        ct = self.noise_add(ct)

        # Apply the corresponding augmentation
        augmented_data = self.augmentations[aug_idx](ct)

        return augmented_data.squeeze(0), stiffness_mat, strength_mat, prop, name, aug_idx

    def set_rotation_prob(self, prob):
        self.original_dataset.random_rotation_prob = prob

    def set_noise_add(self, noise_add):
        self.noise_add = noise_add


class CT_batch:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.ct = torch.stack(transposed_data[0], 0).float()
        self.stiffness_mat = torch.stack(transposed_data[1], 0).float()
        self.strength_mat = torch.stack(transposed_data[2], 0).float()
        self.props = torch.stack(transposed_data[3], 0).float()
        self.name = transposed_data[4]
        self.size = self.ct.size(0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.ct = self.ct.pin_memory()
        self.stiffness_mat = self.stiffness_mat.pin_memory()
        self.strength_mat = self.strength_mat.pin_memory()
        self.props = self.props.pin_memory()
        return self

    def cuda(self, device=None):
        self.ct = self.ct.cuda(device)
        self.stiffness_mat = self.stiffness_mat.cuda(device)
        self.strength_mat = self.strength_mat.cuda(device)
        self.props = self.props.cuda(device)
        return self

    def get_gt_dict(self):
        return {
            'Properties': self.props,
            'Stiffness Tensor': self.stiffness_mat,
            'Strength Tensor': self.strength_mat,
        }


def CT_batch_wrapper(batch):
    return CT_batch(batch)


class CT_batch_Augmented:
    def __init__(self, data):
        transposed_data = list(zip(*data))
        self.ct = torch.stack(transposed_data[0], 0).float()
        self.stiffness_mat = torch.stack(transposed_data[1], 0).float()
        self.strength_mat = torch.stack(transposed_data[2], 0).float()
        self.props = torch.stack(transposed_data[3], 0).float()
        self.name = transposed_data[4]
        self.resolution_index = torch.tensor(transposed_data[5])
        self.size = self.ct.size(0)

    # custom memory pinning method on custom type
    def pin_memory(self):
        self.ct = self.ct.pin_memory()
        self.stiffness_mat = self.stiffness_mat.pin_memory()
        self.strength_mat = self.strength_mat.pin_memory()
        self.props = self.props.pin_memory()
        self.resolution_index = self.resolution_index.pin_memory()
        return self

    def cuda(self, device=None):
        self.ct = self.ct.cuda(device)
        self.stiffness_mat = self.stiffness_mat.cuda(device)
        self.strength_mat = self.strength_mat.cuda(device)
        self.props = self.props.cuda(device)
        self.resolution_index = self.resolution_index.cuda(device)
        return self

    def get_gt_dict(self):
        return {
            'Properties': self.props,
            'Stiffness Tensor': self.stiffness_mat,
            'Strength Tensor': self.strength_mat,
        }


def CT_batch_Augmented_wrapper(batch):
    return CT_batch_Augmented(batch)


if __name__ == '__main__':
    prefix = './'
    dataset = MicroBoneDataset('./dataset/',
                               './dataset/data.csv',
                               './splits/test.csv',
                               prefix='./dataset/',
                               load_cache=False,  #run save cache to enable this
                               load_rotated_augmentations=False,  # run PBRA at first to enable this
                               random_rotation_prob=0,
                               transform=None)
    #dataset.save_cache() #do each for train0.csv, val0.csv and test.csv to enable load cache
    test_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=CT_batch_wrapper, pin_memory=True)
    for batch_ndx, batch in enumerate(test_loader):
        print(batch.size)
        break
    '''dataset = MicroBoneDataset('./dataset/',
                               './dataset/data.csv',
                               './splits/test.csv',
                               prefix='./dataset/',
                               load_cache=True,
                               load_rotated_augmentations=True,
                               random_rotation_prob=1,
                               transform=None)
    dataset = AugmentedRepeatDataset(dataset)
    test_loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=CT_batch_Augmented, pin_memory=True)
    print(len(test_loader))
    for batch_ndx, batch in enumerate(test_loader):
        print(batch.size)
        break
        '''
