# Surf_keypoints_detection_and_pointcloud_registration
使用SURF特征对两帧RGBD数据进行配准

### 依赖项
- Opencv with contrib
- PCL

### 数据
- data文件夹下为两帧kinect v2 rgbd数据

### SURF特征检测与RANSAC提纯（Surf_detection_with_Ransac.cpp）
- 对两张图片进行SURF特征提取并使用Ransac提纯，根据匹配点计算两帧之间的相机位姿
- 使用示例
```
./Surf_detection_with_Ransac rgb1filename rgb2filename 100
```
- 可使用`./data/`下的`r-1.png`与`r-2.png`进行实验

### 点云配准（Surf_rgbd_pointcloud_Registration.cpp）
- 对两帧RGBD数据进行配准，首先生成点云，然后，提取SURF特征，RANSAC提纯，计算相机位姿，最后，对点云进行旋转平移，得到配准后点云
- 使用示例
```
./Surf_detection_with_Ransac rgb1filename rgb2filename depth1filename depth2filename
```
- 可使用`./data/`下的两帧数据进行实验
