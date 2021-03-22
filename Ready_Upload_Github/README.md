=================================================================================
Instructions for running the R codes:
Note: These files are developed and tested in R 4.0.4.

The following packages are required to run the codes for the R part:
ggplot2
tidyr
gridExtra
MASS
tidyverse
caret
leaps

These packages can be installed by issuing the following command "install.packages()"
"Basic Analysis.Rmd”:
Required input document:
"Basic_analysis_data"

Output:
Performing the EDA and basic regression model fitting, model selection for Exxon's stock.

One may also run the "Basic Analysis BP" and "Basic Analysis - chevron" if interested; those results
are not included in the report because they are almost identical.


=================================================================================
### Required dependencies to run Python code:
* python(3.8)
* pytorch(1.8.0)
* torchvision(0.9.0)
* matplotlib(3.3.4)
* numpy(1.19.2)
* pandas(1.2.3)
* scikit-learn(0.24)

###  Folder Structure
```bash
│...
│   ├── Py
│   │   ├── data
│   │   │   ├── ExxonWholeYear.csv
│   │   │   ├── news.csv
│   │   ├── Exxon_Prediction.py
│   │   ├── Performance_Score.py
```
### Usage
With required packages, run the execution file in your python IDE or in terminal:

    python3 Exxon_Prediction.py

and

    python3 Performance_Score.py 
    
    
================================= GitHub Repository =============================
We will also make the raw python scripts available via our GitHub repository: 




================================= Contact Info ==================================
Yuchen Cao: yc.cao@knights.ucf.edu
Chang Li: changli@knights.ucf.edu
Yukun Li: yukun.li@ucf.edu
Guanqian Wang: guanqian.wang@ucf.edu
Gerrit Welper: Gerrit.Welper@ucf.edu
Feng Yu: feng.yu@ucf.edu

