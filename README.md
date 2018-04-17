# Deep-Convolutional-Network-Cascade-for-Facial-Point-Detection

## 1 Data Preprocessing and Environmental Configuration

#### 1.1 Pre-Work of Environment

First you could download the `raw dataset` provided by paper author [dataset web](https://mmlab.ie.cuhk.edu.hk/archive/CNN/). I save those pictures to desktop, which is easy for me to cite. And the URLs in my programs are relavant to `desktop`.
</br>
#### 1.2 Install tensorflow-gpu

First of all, you need to bulid a new tensorflow-gpu environment with `python = 3.5`. And then the important libs like PIL, matplotlib, pandas, xlrd and xlwt libs should be installed. I used `Anaconda3` to finish these tasks so I recommend you use it to build relavant environment, which is very convenient.
</br>
If your gpu is NVIDIA product, you should also install `CUDA` and `cuDNN`, because they are the necessities of tensorflow-gpu. Before your installing, you need to examine the `computing capability` of your NVIDIA gpu [GPU computing capability](https://blog.csdn.net/real_myth/article/details/44308169). The capability should be larger that `3.0`, and you could not install gpu version of tensorflow if it is smaller. It is important that the specific version of tensorflow-gpu correspond to a certain version of CUDA or cuDNN. For example, my computer is `Win10 + CUDA9.1 + CUDNN7.1 + TensorFlow 1.6`.
</br>
You can reference [how to build a tensorflow-gpu environment 1](https://blog.csdn.net/lwplwf/article/details/54894364) and [how to build a tensorflow-gpu environment 2](https://blog.csdn.net/lwplwf/article/details/54896088). They are the detailed steps of installing tensorflow-gpu.
</br>
You should know that I take the experiment on my `Win10` computer with `NVIDIA GTX860m GPU`. So my programs are writen according to the configuration of my computer, and you could rename and change some of the variables, like URLs. And the versions of [CUDA](https://developer.nvidia.com/cuda-toolkit-archive) and [cuDNN](https://developer.nvidia.com/rdp/cudnn-download) should be related to your GPU, which is really important. Before you download the cuDNN, you should register a NVIDIA account. There are some reference: [how to install cuDNN](https://blog.csdn.net/xuyanan3/article/details/73866883), [some questions when installing CUDA & cuDNN](https://blog.csdn.net/mm644706215/article/details/78797782). Besides, I strongly recommend that you prepare a computer with more than 8G RAM, because there are a lot of parameters.
</br>
#### 1.3 Get Level 1 Dataset

Then you can run `read_pictures_for_level1.py` to get the training dataset. You should know that the dataset is divided into two parts: training dataset (10000 pictures after facebounding) and testing dataset (3466 pictures after facebounding), which are provided by paper author and you can find how to use them in the paper. When you train the network, use 10000 pictures for training and 3466 pictures for testing, because the 13466 pictures are all labeled with the x/y coordinates of five key points. And when you use your trained network for detection, I mean the parameters (learning rate, dropout, L2 regularization and so on) of trained network are defined, you could use the whole 13466 pictures to train the network and see the results of other input images.
</br>
At last, you can find the new folders in the downloaded train folder, which are the results: `lfw_5590_F1`, `lfw_5590_EN1`, `lfw_5590_NM1`, `lfw_5590_F1_test`, `lfw_5590_EN1_test`, `lfw_5590_NM1_test`, `net_7876_F1`, `net_7876_EN1`, `net_7876_NM1`, `net_7876_F1_test`, `net_7876_EM1_test`, `net_7876_NM1_test`. The reason of my building so many folders is that it is convenient for me to train the level 1 network. Indeed you may not write the program `read_pictures_for_level1.py` individually, just put the image preprocessing and network training together in one program.
</br>
The `read_pictures_for_level2_level3.py` is a small test. You can run it to see what the inputs of level 2 & 3 like and it does not help the implemention.
</br>

## 2 Take a small Experiment

I prepare some programs for your test: `F1.py`, `EN1.py`, `NM1.py`, `LE21.py`, `LE31.py`. Please run some of them and you can see the results, which are the `err` definition in the paper. It is important that the PATH of the folders (see Section 1.2) are related to desktop, if your saving URLs are not desktop, you need to replace the variables in the programs.
</br>
The number of iterations may be big and you can adjust it (in Session's for loop). The outputs are all percentages. You can change the hyper parameters and see whether the result will be better. Every 500 iterations approximately costs 20-25 mins (GTX860m). But when the number of iterations reach 5000, it costs approximately 6 hours because the laptop is really hot.
</br>

## 3 Training Method

#### 3.1 Something Important

All the related programs are well trained by me. If you want to train again, the test_images is writen in the program. I set the iteration as 1000, for F1 network as an example, it means that each of the 10000 input images is used 1000 times. If you want to get a more precise result or you do not want to run the following networks, you can set a higher iteration.
</br>
All the input images are all normalized as grey image and 39*39 pixel. The pixel values are all divided by 255.
</br>
In order to enlarge the outputs, I multiply 39 and the relative coordinates of face bounding box, which is easy for me to evaluate the results.
</br>
The end condition of training is that the euclidean distance of the prediction position and the ground truth position is less than 5%. In the paper, the euclidean distance is called `err`.
</br>
#### 3.2 Level 1 Training Method

Related programs are: `F1.py`, `EN1.py` and `NM1.py`.
</br>
In dataset preprocessing (see Section 1.2), the input pictures are well prepared: F1 pictures are -0.05 left, +1.05 right, -0.05 top and +1.05 bottom; EN1 pictures are -0.05 left, +1.05 right, -0.04 top and +0.84 bottom; F1 pictures are -0.05 left, +1.05 right, +0.18 top and +1.05 bottom. The four boundary positions are relative to the normalized face bounding box with boundary positions (0,1,0,1). And the pictures are all reshaped with (39,39), (31,39) and (31,39) according to the paper.
</br>
As for outputs, they can be found in `trainImageList.xlsx` and `testImageList.xlsx`, which are provided by paper author (please see the Section 1.1) [dataset web](https://mmlab.ie.cuhk.edu.hk/archive/CNN/). Of course the 10 or 6 outputs of each network are `the x/y coordinates of five key points (LE, RE, N, LM, RM)`.
</br>
#### 3.3 Level 2 & 3 Training Method

Related programs are: `LE21.py`, `LE21.py`, `RE21.py`, `RE22.py`, `N21.py`, `N22.py`, `LM21.py`, `LM22.py`, `RM21.py`, `RM22.py`, `LE31.py`, `LE32.py`, `RE31.py`, `RE32.py`, `N31.py`, `N32.py`, `LM31.py`, `LM32.py`, `RM31.py` and `RM32.py`.
</br>
The dataset processing and network training works are put together in one program, because at the following two levels we should take training patches centered at positions `randomly shifted` from the ground truth position. So the two works writen in one program is easier for training, because the input region is random and we need not to save the randomly shifted pictures every training time.
</br>
So the inputs of networks are randomly shifted pictures. The size of the regions are defined: *21 with ±0.16, *22 with ±0.18, *31 with ±0.12, *32 with ±0.11. For networks at level 2 and level 3, the four boundary positions are relative to the predicted
facial point position by level 1. The maximum shift in both horizontal and vertical directions is 0.05 at the second level, and 0.02 at the third level, where the distances are normalized with the face bounding box.
</br>
As for outputs, for single network like LE21, they are `the shifted x/y coordinates of key points (LE)`. The definition of the randomly shifted numbers are `rx` and `ry` in the programs. And I put the relative coordinates of level 2 bounding box as the outputs: (1-rx)/2 and (1-ry)/2.
</br>

## 4 Testing Method

#### 4.1 Something Important

All the related programs are tested by CASIA test dataset. The raw pictures of CASIA dataset is normalized as 144*144 pixel.
</br>
You can only run the `_run.py` to see the results, because the hyper parameters are all well trained in Section 3.
</br>
#### 4.2 Run Level 1

Related programs are: `F1_run.py`, `EN1_run.py` and `NM1_run.py`. First you should run these 3 '.py' files.
</br>
All the results are saved to excels: `F1.xlsx`, `EN1.xlsx` and `NM1.xlsx`. Then run `get_level1_keypoints.py`: put the outputs together (average) and get `level1.xlsx`.
</br>
#### 4.3 Run Level 2

Related programs are: `LE21_run.py`, `LE22_run.py`, `RE21_run.py`, `RE22_run.py`, `N21_run.py`, `N22_run.py`, `LM21_run.py`, `LM22_run.py`, `RM21_run.py` and `RM22_run.py`. If you successfully get 'level1.xlsx', you can run these 10 '.py' files.
</br>
All the results are saved to excels: `LE21.xlsx`, `LE22.xlsx`, `RE21.xlsx`, `RE22.xlsx`, `N21.xlsx`, `N22.xlsx`, `LM21.xlsx`, `LM22.xlsx`, `RM21.xlsx` and `RM22.xlsx`. Then run `get_level2_keypoints.py`: put the outputs together (average) and get `level2.xlsx`.
</br>
#### 4.4 Run Level 3

Related programs are: `LE31_run.py`, `LE32_run.py`, `RE31_run.py`, `RE32_run.py`, `N31_run.py`, `N32_run.py`, `LM31_run.py`, `LM32_run.py`, `RM31_run.py` and `RM32_run.py`. If you successfully get 'level2.xlsx', you can run these 10 '.py' files.
</br>
All the results are saved to excels: `LE31.xlsx`, `LE32.xlsx`, `RE31.xlsx`, `RE32.xlsx`, `N31.xlsx`, `N32.xlsx`, `LM31.xlsx`, `LM32.xlsx`, `RM31.xlsx` and `RM32.xlsx`. Then run `get_level3_keypoints.py`: put the outputs together (average) and get `level3.xlsx`.
</br>
The 'level3.xlsx' is the final result. Plot it, and see whether the result is great.
</br>

## 5 Results and Reference
