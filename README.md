# KKR Facial Affect Reader

## About
KKR Facial Affect Reader (KKR-FAR) is a GUI-based system to estimate valence and arousal from a facial video.
KKR-FAR estimates and visualizes the intensity of valence and arousal in real time from a video of a single face.
Either a video file or a video data captured by a web camera can be used for the facial video.

The following image shows the GUI-based system of KKR-FAR. 
Top left, input video; top right, current valence and arousal intensity values, represented as a point in two-dimensional space; bottom, graph showing changes in intensity values.

<br><img width="640" alt="Screenshot of the GUI-based system" src="https://github.com/user-attachments/assets/db374dbf-12bc-419a-99d4-733ea711b452" /><br>

After the estimation, KKR-FAR summarizes the estimation results as shown in the following figure, including temporal changes in valence and arousal intensity throughout the entire video, and the distribution and frequency of intensity values, indicated by color changes.

<br><img width="480" alt="Image" src="https://github.com/user-attachments/assets/333f9b7e-91c8-4014-8dec-6ed7aea5d103" /><br>

KKR-FAR is easy to use and will be helpful for the analysis of the estimation result of valence and arousal.
"KKR" stands for the affiliations of the developers; Kyoto Institute of Technology, KOHINATA Limited Liability Company, and RIKEN.

## How to use
Currently, KKR-FAR works only on 64-bit Windows (Windows 10 or later).

**KKR-FER was updated on April 25, 2025.**
The latest Windows binary of KKR-FER can be found [here](https://gp1.work/prgshare/dl.php?download=3).
Estimated valence/arousal values and intensity values of 17 AUs can be exported as a CSV file.

**KKR-FER was updated on March 7, 2025.**
The previous version of Windows binary of KKR-FER can be found [here](https://gp1.work/prgshare/dl.php?download=2) (We are preparing to provide source codes of KKR-FER).
The new version of KKR-FER requires two model files for the estimation of valence and arousal. The original version of KKR-FER (requires a single model file) is still available from [here](https://gp1.work/prgshare/dl.php?download=1).

Because KKR-FAR depends on [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), the following materials are required:

- [64-bit Windows binary of OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/releases/download/OpenFace_2.2.0/OpenFace_2.2.0_win_x64.zip)
- [Additional model files (*.dat files)](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Model-download)

KKR-FAR works by placing all files contained in KKR-FER-r*.zip in the folder containing the OpenFace executables.

## Citation

If you use KKR-FER in your research, we ask you to cite the following work.

[H. Nomiya, K. Shimokawa, S. Namba, M. Osumi, and W. Sato, An Artificial Intelligence Model for Sensing Affective Valence and Arousal from Facial Images, Sensors 2025, 25(4), 1188.](https://www.mdpi.com/1424-8220/25/4/1188)

## License
For the latest version of the license, see [LICENSE](/LICENSE)

## OpenFace Integration Notice

This project includes OpenFace, a facial behavior analysis toolkit developed by Carnegie Mellon University, University of Cambridge, and University of Southern California.

Important Notice: Non-Commercial and Academic Use Only  
This software incorporates OpenFace under a "special license agreement with Carnegie Mellon University", allowing for its inclusion in the binary distribution. However, the following restrictions apply:

- "Commercial use is strictly prohibited" unless a separate commercial license is obtained from Carnegie Mellon University.
- This software is provided for "academic research and non-commercial purposes only".
- Redistribution of this software is permitted "only under the same license conditions".

For more details regarding OpenFace’s license, please visit:  
[OpenFace License](https://github.com/TadasBaltrusaitis/OpenFace/blob/master/OpenFace-license.txt)

If you intend to use OpenFace for commercial purposes, please contact Carnegie Mellon University to obtain the necessary licensing:  
[Flintbox - OpenFace Licensing](https://cmu.flintbox.com/technologies/5c5e7fee-6a24-467b-bb5f-eb2f72119e59)

## Contact
If you have any problems, questions, or requests about KKR-FER, please contact us by email: kkr_support#kohinet.com (please replace # with @).
