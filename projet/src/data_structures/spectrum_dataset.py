from __future__ import annotations
import os
from typing import Dict, List, Optional, Self, Iterable
import numpy as np
from torch import Tensor, tensor
from torch.utils.data import Dataset, WeightedRandomSampler
from tqdm import tqdm


class PatientDataset(Dataset):
    """
    This class implements a simple h5py dataset that easily loads and stores h5py files conveniently. It is assumed that
    the data contains only two classes.
    """
    def __init__(self, data: list[tuple[Tensor]]):
        """
        Constructs a PatientDataset object with the corresponding information.

        Parameters
        ----------
        data : list[tuple[Tensor]]
            List of (images, targets) tensors.
        """
        super().__init__()
        self.data = data

    @classmethod
    def load(cls, filename: str) -> Self:
        """
        Loads a PatientDataset from a file.

        Parameters
        ----------
        filename : str
            Name of the file from which to load the PatientDataset.
        """
        data = []
        with h5py.File(filename, "r") as f:
            for patient_info in f.values():
                images = Tensor(np.array(patient_info["0"]["feature_0"]))
                targets = Tensor(np.array(patient_info["0"]["target_0"]))
                data.append((images, targets))
        return cls(data)

    @classmethod
    def create(
        cls,
        dicom_filename: str,
        hdf_filename: str,
        transforms: Optional[Dict[str, List[Transformation]]] = {"feature": [], "target": []},
        *args,
        return_dataset: Optional[bool] = True
    ) -> Optional[Self]:
        """
        Creates an HDF5 file from a folder containing DICOM files and, if desired, returns it as a PatientDataset.

        Parameters
        ----------
        dicom_filename : str
            Path to the the folder containing the patients and their DICOM files. The architecture of the directory will
            be replicated within the HDF5 file.
        hdf_filename : str
            Path of the hdf5 file to be created. Both .h5 and .hdf5 formats are supported.
        transforms : Optional[Dict[str, List[Transformation]]]
            A dictionary containing "feature" and "target" keys, each with a list of transformations for the associated
            Dicom series.
        return_dataset : Optional[bool]
            If True, returns the PatientDataset created, otherwise returns nothing. Defaults to True.
        
        Returns
        -------
        patient_dataset : PatientDataset
            The PatientDataset created from the generated HDF5 file.
        """
        target_transforms = Compose(transforms["target"])
        feature_transforms = Compose(transforms["feature"])
        root = dicom_filename
        dataset =  h5py.File(hdf_filename, "w")

        for patient in tqdm(os.listdir(root)):
            patient_group = dataset.create_group(str(patient))

            for i, serie in enumerate(os.listdir(f"{root}/{patient}")):
                series_group = patient_group.create_group(str(i))
                feature_dirs = []
                target_dirs = []

                for dicom_dir in os.listdir(os.path.join(root, patient, serie)):
                    if len(os.listdir(os.path.join(root, patient, serie, dicom_dir))) > 1:
                        feature_dirs.append(dicom_dir)
                    if len(os.listdir(os.path.join(root, patient, serie, dicom_dir))) == 1:
                        target_dirs.append(dicom_dir)

                for i, feature_dir in enumerate(feature_dirs):
                    folder_path = os.path.join(root, patient, serie, feature_dir)
                    file_list = os.listdir(folder_path)
                    first_file = dcmread(os.path.join(folder_path, list(file_list)[0]))
                    shape = (len(file_list), first_file.Rows, first_file.Columns)
                    pos = [
                        dcmread(os.path.join(folder_path, file)).ImagePositionPatient[2] for file in file_list
                    ]
                    start_image_position, thickness = np.max(pos), abs((np.max(pos) - np.min(pos))/(shape[0]-1))

                    features = np.zeros(shape)
                    for dicom_file in file_list:
                        feature_file = dcmread(os.path.join(folder_path, dicom_file))
                        position = round((start_image_position - feature_file.ImagePositionPatient[2]) / thickness)
                        features[position] = feature_file.pixel_array

                    feature_dataset = series_group.create_dataset(f"feature_{i}", data=feature_transforms(features))

                for i, target_dir in enumerate(target_dirs):
                    folder_path = os.path.join(root, patient, serie, target_dir)
                    target_file = dcmread(os.path.join(folder_path, os.listdir(folder_path)[0]))
                    shape, targets = features.shape, np.zeros_like(features)
                    depth = len(target_file.PerFrameFunctionalGroupsSequence)
                    pos = []
                    for frame in target_file.PerFrameFunctionalGroupsSequence:
                        if "PlanePositionSequence" in frame:
                            pos.append(frame.PlanePositionSequence[0].ImagePositionPatient[2])
                    start_target_position, thickness = np.max(pos), abs((np.max(pos) - np.min(pos))/(depth-1))

                    start_i = round((start_image_position - start_target_position) / thickness)
                    n_frames = target_file.NumberOfFrames
                    targets[start_i:start_i+n_frames] = target_file.pixel_array[::-1]

                    target_dataset = series_group.create_dataset(f"target_{i}", data=target_transforms(targets)) 
        
        if return_dataset:
            return cls.load(hdf_filename)

    def __getitem__(self, key: int) -> tuple[Tensor]:
        return self.data[key]
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __add__(self, other: PatientDataset) -> Self:
        new_data = self.data + other.data
        return self.__class__(new_data)
    
    @property
    def weights(self) -> Tensor:
        """
        Gives the weights to stabilize the asymetric classes distribution.

        Returns
        -------
        Tensor
            Weights to apply to each class to enhance stability.
        """
        classes = []
        for image, targets in self:
            classes.append(int(targets.max()))
        weights_per_class = len(classes) / np.bincount(classes)
        return tensor(weights_per_class[0] / weights_per_class[1])

    def flatten(self) -> PatientDataset:
        """
        Flattens the PatientDataset's data to single layer scans instead of 3D ones. This allows to apply a model to
        2D images instead of a 3D cube.

        Returns
        -------
        PatientDataset
            Newly flattened 2D PatientDataset.
        """
        assert self[0][0].dim() == 3, "Cannot flatten as the PatientDataset already has 2D image data"
        new_data = []
        for images, targets in self:
            for image_slice, target_slice in zip(images, targets):
                new_data.append((image_slice[None,:,:], target_slice[None,:,:]))
        return self.__class__(new_data)
    
    def get_weighted_random_sampler(self, num_samples: int) -> WeightedRandomSampler:
        """
        Gives the WeightedRandomSampler corresponding to the class's features. This allows for a proper model training
        in the event that the number of instances of each class are highly different. This method is only useful for 2D
        data, which does not have segmentation at every image. It is useless for 3D scans as every datum has 
        segmentation.

        Parameters
        ----------
        num_samples : int
            Number of samples to draw with the sampler. This value is given to the WeightedRandomSampler constructor.

        Returns
        -------
        WeightedRandomSampler
            Sampler initialized with the right weights to counterbalance the classes distribution.
        """
        classes = []
        for image, targets in self:
            classes.append(int(targets.max()))
        weights_per_class = len(classes) / np.bincount(classes)
        weights = weights_per_class[classes]
        sampler = WeightedRandomSampler(weights, num_samples)
        return sampler

    def random_split(self, lengths: Iterable[float | int]) -> tuple[PatientDataset]:
        """
        Splits the PatientDataset into random Datasets of the given lengths. This method is equivalent to the
        torch.utils.data.random_split method, but returns PatientDataset instances instead of Subset instances. It also
        allows to specify the size of each dataset directly.

        Parameters
        ----------
        lengths : Iterable[float | int]
            If the values are floats, proportion of each dataset relative to the total length of the input dataset.
            If the values are ints, exact number of samples to take from each dataset.
        
        Returns
        -------
        tuple[PatientDataset]
            PatientDataset instances of the given lengths.
        """
        shuffled_data = self.data.copy()
        np.random.shuffle(shuffled_data)
        total_length = len(self)

        if sum(lengths) == 1.0:
            split_quantities = (np.array(lengths) * total_length).astype(int)   # get the desired length of each dataset
        else:
            split_quantities = lengths
            
        split_indices = [0] + list(np.cumsum(split_quantities))
        slices = [slice(split_indices[i], split_indices[i+1]) for i in range(len(lengths))]
        new_datasets = (self.__class__(shuffled_data[s]) for s in slices)
        return new_datasets
