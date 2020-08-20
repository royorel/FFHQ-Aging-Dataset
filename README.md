# FFHQ-Aging Dataset
### [Project Page](https://grail.cs.washington.edu/projects/lifespan_age_transformation_synthesis/) | [Paper](https://arxiv.org/pdf/2003.09764.pdf) | [Lifespan Age Transformation Synthesis Code](https://github.com/royorel/Lifespan_Age_Transformation_Synthesis)
<div align="center"><img src=./images/dataset_samples_github.png></div>

## Overview
FFHQ-Aging is a Dataset of human faces designed for benchmarking age transformation algorithms as well as many other possible vision tasks.

This dataset is an extention of the NVIDIA [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset), on top of the 70,000 original FFHQ images, it also contains the following information for each image:
1. Gender information (male/female with confidence score)
2. Age group information (10 classes with confidence score)
3. Head pose (pitch, roll & yaw)
4. Glasses type (none, normal or dark)
5. Eye occlusion score (0-100, different score for each eye)
6. Full semantic map (19 classes, based on CelebAMask-HQ labels)

If you use this dataset for your work, please cite our paper:
> **Lifespan Age Transformation Synthesis**<br>
> Roy Or-El, Soumyadip Sengupta, Ohad Fried, Eli Shechtman, Ira Kemelmacher-Shlizerman<br>
> ECCV 2020<br>
> https://arxiv.org/pdf/2003.09764.pdf

## Dataset Statistics
The following histogram shows the age class distribution per gender.

<div align="center"><img src=./images/age_distribution.png></div>

Gender labels & confidence, age class labels & confidence score, head pose, glasses type and left & right eye occlusion scores for each individual image are stored in **ffhq_aging_labels.csv**.

## Pre-Requisits
You must have a **GPU with CUDA support** in order to run the segmentation code.

This code requires **PyTorch** to be installed, please go to [Pytorch.org](https://pytorch.org/) for installation info.<br>
In addition, the following python packages should be installed:
1. requests
2. pillow
3. numpy
4. scipy
5. PyDrive

If any of these packages are not installed on your computer, you can install them using the supplied `requirements.txt` file:<br>
```pip install -r requirements.txt```

**Note for windows users:** make sure that you have a 64bit python version installed. Otherwise you might get a memory error when reading the FFHQ JSON file.

## Usage

### Default download method
To download the dataset in the default resolution (256x256) run:<br>
Linux & Mac: ```./get_ffhq_aging.sh```<br>
Windows: ```get_ffhq_aging.bat```<br>

If you encounter a "quota exceeded" error, see [Downloading with PyDrive](#downloading-with-pydrive)

### Downloading with PyDrive
Google drive enforces a quota on file download by anonymous users.
If you encounter a "quota exceeded" error, either wait 24 hours for the quota limit to reset and try again, or follow the procedure below.

#### Step 1: Add the original FFHQ dataset to the "Shared With Me" section of your Google Drive
Note: this step does *not* count against your Google Drive storage limit.

* Login to your Google Drive
* Visit [ffhq-dataset](https://drive.google.com/drive/folders/1u2xu7bSrWxrbUxk-dT-UvEJq8IjdmNTP)

#### Step 2: Enable the Google Drive API
Note: this only applies to *your* download script, and does not give access to other users.
Nevertheless, we recommend revoking the script's access after the download is complete.

* Go to : https://developers.google.com/drive/api/v3/quickstart/python
* Click on enable drive API
* Select Desktop app
* Download client configuration
* Rename this file to `client_secrets.json` and place it in the same folder as the download script (`download_ffhq_aging.py`).

#### Step 3: Run the script
* In order to run the code with authntication, edit the `get_ffhq_aging.sh/bat` script, and add the `--pydrive` flag when invoking `download_ffhq_aging.py`. This will open a browser authentication window. Log in to your account and allow access.
* If you have no display (like when running from a remote compute server), edit the `get_ffhq_aging.sh/bat` script, and also add the `--cmd_auth` flag when invoking `download_ffhq_aging.py`. This will print a Google authentication link to the screen. Open the link in any browser, allow access, and paste the Google authentication token back to the command line.  

**Important Note**: using this will let the code access your Google Drive, which might pose a security risk.
We recommend using it only in cases when the default interface consistently returns a quota exceeded error.
In addition, we recommend to disable the drive API and delete `client_secrets.json` after the dataset download is complete.

### Optional Arguments
**download_ffhq_aging.py**<br>
```
  --debug              run in debug mode, download 50 random images (default: False)
  --pydrive            use pydrive interface to download files. It can override google drive quota limitation
                       this requires google credentials (default: False)
  --cmd_auth           use command line google authentication when using pydrive interface
                       this is good when running on a server with no display (default: False)
  --resolution         final resolution of saved images (default: 256)
  --num_threads NUM    number of concurrent download threads (default: 32)
  --num_attempts NUM   number of download attempts per file (default: 10)
```

**run_deeplab.py**<br>
```
  --resolution         segmentation output size (default: 256)
  --workers            number of data loading workers (default: 4)
 ```

 Please make sure that the `--resolution` option for both scripts is the same


## License & Privacy
The dataset is made available under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license by University of Washington. You can **use, redistribute, and adapt it for non-commercial purposes**, as long as you (a) give appropriate credit by **citing our paper**, (b) **indicate any changes** that you've made, and (c) distribute any derivative works **under the same license**.

The individual images were published in Flickr by their respective authors under either [Creative Commons BY 2.0](https://creativecommons.org/licenses/by/2.0/), [Creative Commons BY-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/), [Public Domain Mark 1.0](https://creativecommons.org/publicdomain/mark/1.0/), [Public Domain CC0 1.0](https://creativecommons.org/publicdomain/zero/1.0/), or [U.S. Government Works](http://www.usa.gov/copyright.shtml) license. All of these licenses allow **free use, redistribution, and adaptation for non-commercial purposes**. However, some of them require giving **appropriate credit** to the original author, as well as **indicating any changes** that were made to the images. The license and original author of each image are indicated in the metadata.

* [https://creativecommons.org/licenses/by/2.0/](https://creativecommons.org/licenses/by/2.0/)
* [https://creativecommons.org/licenses/by-nc/2.0/](https://creativecommons.org/licenses/by-nc/2.0/)
* [https://creativecommons.org/publicdomain/mark/1.0/](https://creativecommons.org/publicdomain/mark/1.0/)
* [https://creativecommons.org/publicdomain/zero/1.0/](https://creativecommons.org/publicdomain/zero/1.0/)
* [http://www.usa.gov/copyright.shtml](http://www.usa.gov/copyright.shtml)

The JSON metadata is made available under [Creative Commons BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) license by NVIDIA Corporation.

The individual images and JSON metadata are hosted on NVIDIA's Google Drive, please see the original [FFHQ dataset](https://github.com/NVlabs/ffhq-dataset) for more details.

To find out whether your photo is included in the original Flickr-Faces-HQ dataset and/or get it removed from both this dataset and the original FFHQ dataset please go to the privacy section in the original [FFHQ Dataset website](https://github.com/NVlabs/ffhq-dataset) and follow the instructions.

## Acknowledgements
We wish to thank Thevina Dokka for helping us collecting the dataset.

Original face images were collected in the [NVIDIA FFHQ dataset](https://github.com/NVlabs/ffhq-dataset).
> **A Style-Based Generator Architecture for Generative Adversarial Networks**<br>
> Tero Karras, Samuli Laine, Timo Aila, CVPR 2019<br>
> http://openaccess.thecvf.com/content_CVPR_2019/papers/Karras_A_Style-Based_Generator_Architecture_for_Generative_Adversarial_Networks_CVPR_2019_paper.pdf

Age & gender labels and confidence scores were collected using the [Appen](https://www.appen.com/) platform.

Head pose, glasses type and eye occlusion score were extraceted using the [Face++](https://www.faceplusplus.com/) platform.

Face Semantic maps were acquired by training a pytorch implementation of [DeepLabV3](https://github.com/chenxi116/DeepLabv3.pytorch) network on the [CelebAMASK-HQ](https://github.com/switchablenorms/CelebAMask-HQ) dataset.
> **Encoder-Decoder with Atrous Separable Convolution for Semantic Image Segmentation**<br>
> Liang-Chieh Chen, Yukun Zhu, George Papandreou, Florian Schroff, Hartwig Adam, ECCV 2018<br>
> http://openaccess.thecvf.com/content_ECCV_2018/papers/Liang-Chieh_Chen_Encoder-Decoder_with_Atrous_ECCV_2018_paper.pdf

> **MaskGAN: Towards Diverse and Interactive Facial Image Manipulation**<br>
> Cheng-Han Lee, Ziwei Liu, Lingyun Wu, Ping Luo, CVPR 2020<br>
> https://arxiv.org/pdf/1907.11922.pdf
