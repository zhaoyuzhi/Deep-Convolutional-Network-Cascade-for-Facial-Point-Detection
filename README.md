# Deep-Convolutional-Network-Cascade-for-Facial-Point-Detection

1 Data Preprocessing and Environmental Configuration
------
First you could download the raw dataset provided by author [dataset web](http://mmlab.ie.cuhk.edu.hk/archive/CNN/). I save those pictures to desktop, which is easy for me to cite. And the URLs in my programs are relavant to desktop.
</br>
Then you need to bulid a new tensorflow-gpu environment with python = 3.5. And PIL, matplotlib, pandas, xlrd and xlwt libs should be installed. I used Anaconda3 to finish these tasks so I recommend you use it to build relavant environment. You can reference [how to build a tensorflow-gpu environment 1](https://blog.csdn.net/lwplwf/article/details/54894364) and [how to build a tensorflow-gpu environment 2](https://blog.csdn.net/lwplwf/article/details/54896088).
</br>
You should know that I take the experiment on my Win10 computer with GTX860m GPU. So my programs are writen according to my computer, and you could rename and change some of the variables.
</br>
Then you can run `read_pictures_for_level1.py` and `read_pictures_for_level2_level3.py` to get the training dataset. The dataset is divided into two parts: training dataset and testing dataset, which are provided by author and you can find how to use them in the paper.
</br>
At last, you can find the new folders in the downloaded train folder, which are the results.
</br>
2 Take a small Experiment
------
  Run the `F1.py`, `EN1.py` and `NM1.py`. You can see that the results that are the `err` in the paper.
