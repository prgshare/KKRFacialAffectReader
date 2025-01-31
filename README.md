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
Windows binary of KKR-FAR can be found [here](http://mmde.is.kit.ac.jp/KKR-FER.zip).
We are preparing to provide source codes of KKR-FER.

Because KKR-FAR depends on [OpenFace](https://github.com/TadasBaltrusaitis/OpenFace), the following materials are required:

- [64-bit Windows binary of OpenFace](https://github.com/TadasBaltrusaitis/OpenFace/releases/download/OpenFace_2.2.0/OpenFace_2.2.0_win_x64.zip)
- [Additional model files (*.dat files)](https://github.com/TadasBaltrusaitis/OpenFace/wiki/Model-download)

KKR-FAR works by placing all files contained in KKR-FER.zip in the folder containing the OpenFace executables.

## License
