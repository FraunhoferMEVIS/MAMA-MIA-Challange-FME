import argparse
import json
import os


class Dataset:
    def __init__(self, dataset_folder: str):
        self.patient_id_list = []
        self.patient_dce_mri_paths = {}
        self.patient_json_files = {}
        images_folder = os.path.join(dataset_folder, 'images')
        patient_info_folder = os.path.join(dataset_folder, 'patient_info_files')
        for patient_id_upper in os.listdir(images_folder):
            patient_id_lower = patient_id_upper.lower()
            self.patient_id_list.append(patient_id_lower)
            patient_image_folder = os.path.join(images_folder, patient_id_lower)
            dce_mri_paths = []
            for dce_mir_path in os.listdir(patient_image_folder):
                dce_mri_paths.append(os.path.join(patient_image_folder, dce_mir_path))
            self.patient_dce_mri_paths[patient_id_lower] = dce_mri_paths
            patient_info_json = os.path.join(patient_info_folder, f'{patient_id_lower}.json')
            self.patient_json_files[patient_id_lower] = patient_info_json
        
    def get_patient_id_list(self) -> list[str]:
        return self.patient_id_list

    def get_dce_mri_path_list(self, patient_id: str) -> list[str]:
        return self.patient_dce_mri_paths[patient_id]

    def read_json_file(self, patient_id: str) -> dict:
        json_path = self.patient_json_files[patient_id]
        with open(json_path, 'r') as file:
            patient_info = json.load(file)
        return patient_info


def main(dataset_folder: str, output_folder: str):
    dataset = Dataset(dataset_folder)
    print(dataset.get_patient_id_list())



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Test inference script for code submission of MAMA-MIA challenge."
    )
    parser.add_argument("dataset_folder", help="Path to the input folder containing patient files.")
    parser.add_argument("output_folder", help="Path to the output folder where inference results will be stored.")
    args = parser.parse_args()

    main(args.dataset_folder, args.output_folder)