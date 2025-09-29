## [Waymo]()
  - We download the dataset following the instruction of [NeuralSim](https://github.com/PJLab-ADG/neuralsim).


## [KITTI](https://www.cvlibs.net/datasets/kitti/)
  - Please download `raw_data` and from [KITTI](https://www.cvlibs.net/datasets/kitti/) website.
  - We use `sync` dataset (10 Hz IMU), we upload the processed IMU data on [huggingface](https://huggingface.co/datasets/Promethe-us/VINGS-Mono-Dataset).
  - Download the data and list them as below:
      ```cmd
      ├── metadata
      ├── image_00
      │   └── data
      ├── image_01
      │   └── data
      ├── image_02
      │   ├── data
      ├── image_03
      │   └── data
      └── oxts
          └── data
      ```    

## [KITTI360](https://www.cvlibs.net/datasets/kitti-360/)
  - Please download `raw_data` and from [KITTI360](https://www.cvlibs.net/datasets/kitti-360/) website.
  - We use `unsync` dataset (100 Hz IMU), c2i is calibrated using openvins, The c2i and intrinsic are same (You can find them in our configs), we upload the processed IMU data on [huggingface](https://huggingface.co/datasets/Promethe-us/VINGS-Mono-Dataset) and put it to `metadata`.
  - Download the data and list them as below:
      ```cmd
      ├── metadata
      ├── image_00
      │   └── data_rgb
      ├── image_01
      │   └── data_rgb
      └── oxts
          └── data
      ``` 

## Self-Collected

- We will upload the data we collected ourselves **after** we have blurred out faces and license plates.