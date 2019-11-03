#!/bin/sh

logFile="xxx/xxx/logfile.txt"
echo -e "\n\n\n" >> "$logFile"

date >> "$logFile"


echo -e "\n\n\n" >> "$logFile"
cat "$paraDictPath" >> "$logFile"
echo -e "\n\n\n" >> "$logFile"

echo ''
cat "$paraDictPath"


#201610

python3 main.py \
--dataPath input.csv \
--islinear 0 \
--model ALE \
--majorVec $majorVec \
--trainingTerms ['200970','201010','201070','201110','201170','201210','201270','201310','201370','201410','201470','201510','201570'] \
--testTerms ['201610'] \
--paraDictPath "$paraDictPath" \
--userFile studentFile.csv \
--itemFile courseFile.csv \
--instrFile instrFile.csv \
--trainFile trainFile.json \
--testFile testFile.json \
--ifRead 0 \
--ifFTF 1 \
--logFile "$logFile" \
--lf 0.8 \
--coCrsSum 0





echo -e "\n\n\n" >> "$logFile"

date >> "$logFile"

echo -e "\n\n\n" >> "$logFile"
