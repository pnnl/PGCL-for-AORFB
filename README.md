# Physics-Guided Continual Learning for Aqueous Organic Redox Flow Battery
This repository contians the dataset and Physics-Guided Continual Learning (PGCL) code reported in [ACS Energy Letters](https://pubs.acs.org/doi/10.1021/acsenergylett.4c00493).

## [DataProcessing](DataProcessing/) Folder 
The folder contians the raw dataset (`ID 780 Cell Dataset.xlsx`) documenting the aqueous organic redox flow battery (AORFB) performance with various sampled Aqueous soluble organic (ASO) materials. The data is generated using a 3D interdigiated (ID) cell model with a area of 780 cm^2. It also comes with files (`Data_Processing_Voltage.m`and`Data_Processing_EE.m`) for data processing and visualization (input parameters vs. Voltage/Energy Efficiency(EE)).


## [PGCL](PGCL/) Folder
This folder contains following items:
- Demonstration of catastrophic forgetting for machine learning algorithm
- Comparison of continual learning (CL) performance with different task creation strategies
- PGCL performance evaluations
- PGCL tested with DHP isomerstesting

The codes are developed and tested using Python 3.9 and PyTorch 1.12.1 in the Spyder IDE. The Elastic Weight Consolidation (EWC) and Learning without Forgetting (LwF) methods implemented in this project are adapted from Vincezo Lomonaco et al. [1].

## Authors
    - Yucheng Fu
    - Amanda Howard
    - Panos Stinis

## Reference
1. Vincenzo Lomonaco, Lorenzo Pellegrini, Andrea Cossu, et al. "[Avalanche: An End-to-End Library for Continual Learning](https://openaccess.thecvf.com/content/CVPR2021W/CLVision/html/Lomonaco_Avalanche_An_End-to-End_Library_for_Continual_Learning_CVPRW_2021_paper.html)," in _Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops_, June 2021, pp. 3600-3610.











## DISCLAIMER
This material was prepared as an account of work sponsored by an agency of the United States Government. Neither the United States Government nor the United States Department of Energy, nor Battelle, nor any of their employees, nor any jurisdiction or organization that has cooperated in the development of these materials, makes any warranty, express or implied, or assumes any legal liability or responsibility for the accuracy, completeness, or usefulness or any information, apparatus, product, software, or process disclosed, or represents that its use would not infringe privately owned rights.

Reference herein to any specific commercial product, process, or service by trade name, trademark, manufacturer, or otherwise does not necessarily constitute or imply its endorsement, recommendation, or favoring by the United States Government or any agency thereof, or Battelle Memorial Institute. The views and opinions of authors expressed herein do not necessarily state or reflect those of the United States Government or any agency thereof.




