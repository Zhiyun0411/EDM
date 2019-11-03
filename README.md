### This is the source code for our paper "Grade Prediction Based on Cumulative Knowledge and Co-taken Courses."

To run the code, type


`./run.sh`


in the terminal.

There are two input files: (1) the student-course grades data, (2) parameter file.

A simple example of student-course grades data is as follows:

```
cohort,     stdID, TERMBNR,    TMAJOR, crsID, instrID,  grade
2010Spring, 001,   2012Spring, cs,     cs101, Marry,    4.0
```

where "cohort" is the term when student starts college;
"TERMBNR" is the current term. "stdID", "crsID" and "instrID" are codes for student, course and instructor, respectively. 
"TMAJOR" is the student's major at TERMBNR. 
"grade" is student's garde on the course taught by a specific instructor. 

For example, the first line means student 001 gets a grade 4.0 on course cs101 taught by Prof. Marry at term 2012 spring. Student 001 starts college at term 2010 spring. 


The parameter file has the format as follows:
{"isS": 1, "isbc": 0, "K": 20, "isP": 1, "isT": 1, "FFbatchSize": 100, "w2": 1, "w1": 1, "maxIter": 100, "isbs": 0, "l2-r": 0.0001, "lr_bias": 1e-05, "hiddenSize": [7, 1], "l1-r": 0.0001, "decayRate": 0.001, "inputSize": 20, "l2": 0.1, "l1": 0.001, "lr": 0.0003}

userFile, itemFile, instrFile, trainFile, testFile and logFile are output files.

Please modify the input and output file paths in the code accordingly.
