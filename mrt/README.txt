The Taipei MRT dataset contains the ridership information in terms of number of riders enter/exit each station form November 2015 to March 2017. The dataset is used in [1] to demonstrate the capability of the real-Time Golden Batch monitoring as an attention focusing method.

The downloaded csv files are renamed and stored in the csv folder. The file name is renamed to YYYYMM_enter.csv or YYYYMM_exit.csv. YYYY is the year and MM is the month.

The csv files are parsed and stored in the mat folder. The "data" variable stored in each mat file consists a matrix where the 1st column is the day, the 2nd column is the hour of the day, and rest of columns are readings from all 108 stations.

* all_data.mat consists a matrix which is produced by combining all the matrix in the mat folder. The 1st column is the YYYYMM, the 2nd column is the day, the 3rd column is the hour of the day, and 4th to 101st columns consist readings from all 108 stations (number of rider entering), and the 112nd to 219th columns consist readings from all 108 stations (number of rider exiting).
* station_name_ch.txt consists each station's name (Traditional Chinese) in the storing order (the first station in the file is the third column in each mat files in the mat folder and fourth column in the all_data.mat).
* station_name_en.txt consists each station's name (English) in the storing order (the first station in the file is the third column in each mat files in the mat folder and fourth column in the all_data.mat).

The original data is downloaded from:
http://data.taipei/

[1] Chin-Chia Michael Yeh, Yan Zhu, Hoang Anh Dau, Amirali Darvishzadeh, Mikhail Noskov, and Eamonn Keogh. "Online Amnestic Dynamic Time Warping to Allow Real-Time Golden Batch Monitoring."
