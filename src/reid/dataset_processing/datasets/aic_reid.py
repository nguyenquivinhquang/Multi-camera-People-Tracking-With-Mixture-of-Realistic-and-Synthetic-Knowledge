import os.path as osp

from .bases import BaseImageDataset
from torch.utils.data import DataLoader

from .dataset_loader import ImageDataset, AICDataset
import torchvision.transforms as T
import os


class AIC_reid(BaseImageDataset):
    def __init__(
        self, root="./", label_file=None, verbose=True, relabel=False, **kwargs
    ):
        f = open(label_file, "r")
        f = f.readlines()

        self.root = root
        self.file = f
        self.num_persons = None
        self.dataset = self._process_dir(relabel=relabel)
        return

    def _process_dir(self, relabel):
        persons = dict()
        dataset = []
        for line in self.file:
            img_path, pid = line.split(" ")
            pid = pid.replace("\n", "")
            try:
                scene, camid, frame_id, person_id = img_path.split("_")
            except:
                # print(img_path.split("_")[-4:])
                scene, camid, frame_id, person_id = img_path.split("_")[-4:]
                # exit(0)
            if pid not in persons:
                persons[pid] = len(persons)
            dataset.append(
                (
                    osp.join(self.root, "person_reid", img_path),
                    pid,
                    int(camid.replace("c", "")),
                )
            )
        _dataset = []
        for path, pid, camid in dataset:
            if relabel:
                _dataset.append((path, persons[pid], camid))
            else:
                _dataset.append((path, int(pid), camid))
        self.num_persons = len(persons)

        return _dataset


class AIC_Feature_Extractor(BaseException):
    def __init__(
        self, root="./", label_folder=None, verbose=True, relabel=False, **kwargs
    ):
        self.root = root
        self.label_folder = label_folder
        self.camera_bboxes = {}
        for file in os.listdir(label_folder):
            # label folder name has format: Sxx_Cxx.txt
            scene_cam = file.split(".")[0]
            self.camera_bboxes[scene_cam] = self._read_file(
                osp.join(label_folder, file)
            )
        self._summarize()

        self.dataset = self._process_dir()
        return

    def _read_file(self, file_path):
        f = open(file_path, "r")
        f = f.readlines()
        bbox = []
        for line in f:
            frame_id, _, x, y, w, h, _, _, _, _ = line.split(",")
            bbox.append((frame_id, x, y, w, h))
        return bbox

    def _process_dir(self) -> dict:
        """
        Processes scene and camera folders to build a dataset of 
            (image_path, camera_view, bounding_box) tuples.

        Returns:
            dict: A dictionary containing a list of tuples for each camera view in the dataset.
        """
        camera_person = {}
        for scene_cam in self.camera_bboxes: 
            scene, camera = scene_cam.split("_")
            folder_path = osp.join(self.root, scene, camera, "img1")
            
            scene_cam_data = []
            for frame_id, x, y, w, h in self.camera_bboxes[scene_cam]:
                # frame id format is 000001.jpg
                frame_id = str(frame_id).zfill(6) + ".jpg"
                img_path = osp.join(folder_path, frame_id) 
                bbox = (x, y, w, h)  
                scene_cam_data.append((img_path, scene_cam, bbox))
            camera_person[scene_cam] = scene_cam_data

        return camera_person

    def _summarize(self):
        print("Total cameras: {}".format(len(self.camera_bboxes)))
        for cam in self.camera_bboxes:
            print(
                "Camera {} has {} objects".format(
                    cam.split("/")[-1], len(self.camera_bboxes[cam])
                )
            )


if __name__ == "__main__":
    dataset_veri = AIC_Feature_Extractor(
        "/mnt/Data/dataset/ReiD/AIC23_Track1_MTMC_Tracking/test",
        "/mnt/Data/CVIP-Lab-Work/segmentation/OneFormer/outputs",
    )
    transform = T.Compose(
        [
            T.Resize((112, 112)),
            T.ToTensor(),
        ]
    )
    train_loader = DataLoader(AICDataset(dataset_veri.dataset, transform), batch_size=2)
    import matplotlib.pyplot as plt

    for img, camid, path in train_loader:
        print(camid, img[0].shape, path)
        plt.imshow(img[0].permute(1, 2, 0))
        # plt.savefig("img.png")
        plt.show()

        break
    pass
