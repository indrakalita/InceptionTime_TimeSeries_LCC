# Inception time DCNN for land cover classification by analyzing multi-temporal remotely sensed images
Indrajit Kalita, Graduate Student Member, IEEE and Moumita Roy, Member, IEEE

Publication:
Kalita, I., & Roy, M. (2022, July). Inception time DCNN for land cover classification by analyzing multi-temporal remotely sensed images. In IGARSS 2022-2022 IEEE International Geoscience and Remote Sensing Symposium (pp. 5736-5739). IEEE.


Instruction:
1. finetuneVGG16.py: Run the script to finetune the pre-trained model using the original images.
2. InceptionTime.py: InceptionTime script to obtain the final result.

Model:
The weights of the trained models.


Dataset:
The experimentation using two temporal image data sets from different regions evaluates the performance of the proposed approach. Here, the images are captured using Google earth pro software from the entire subcontinent which is then divided into two sets zone-wise. The images collected over Eastern India and Bangladesh are grouped to form the Eastern sub-continent dataset (ESD); whereas, those from Western India and Pakistan are grouped into the Western sub-continent dataset (WSD). Both datasets are a combination of 6 land cover classes. The total number of images in ESD and WSD are 956 and 1215, respectively.

![alt text](https://github.com/indrakalita/InceptionTime_TimeSeries_LCC/blob/main/Dataset/datasetImage.jpg)

#The datasets are available on request via mail to indrakalita09@gmail.com/moumita2009.roy@gmail.com

Note: Please cite the work if you are using the code in your work.
Thank you
![alt text](https://github.com/indrakalita/InceptionTime_TimeSeries_LCC/blob/main/Architecture/ProposedModel.jpg)
