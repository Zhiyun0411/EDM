#!/bin/sh

logFile="/Users/jessicaren/Downloads/research/Project04/03 myCode/organizeCode/logFile-ftf-ale.txt"
echo -e "\n\n\n" >> "$logFile"

date >> "$logFile"


echo -e "\n\n\n" >> "$logFile"
cat "$paraDictPath" >> "$logFile"
echo -e "\n\n\n" >> "$logFile"

echo ''
cat "$paraDictPath"


#201610

python3 main.py \
--dataPath /Users/jessicaren/Downloads/research/Data/dataOrganize/ftf-crsDict.csv \
--islinear 0 \
--model ALE \
--majorVec $majorVec \
--trainingTerms ['200970','201010','201070','201110','201170','201210','201270','201310','201370','201410','201470','201510','201570'] \
--testTerms ['201610'] \
--paraDictPath "$paraDictPath" \
--userFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/userFile-ftf-ale.csv \
--itemFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/itemFile-ftf-ale.csv \
--instrFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/instrFile-ftf-ale.csv \
--trainFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/trainFile-ftf-ale.json \
--testFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/testFile-ftf-ale.json \
--ifRead 0 \
--ifFTF 1 \
--logFile "$logFile" \
--lf 0.8 \
--coCrsSum 0





echo -e "\n\n\n" >> "$logFile"

date >> "$logFile"

echo -e "\n\n\n" >> "$logFile"
