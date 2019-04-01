#!/bin/sh



logFile="/Users/jessicaren/Downloads/research/Project04/03 myCode/organizeCode/logFile-ftf-ckcc.txt"




echo -e "\n\n\n" >> "$logFile"

date >> "$logFile"

cat "$paraDictPath"

echo -e "\n\n\n" >> "$logFile"
cat "$paraDictPath" >> "$logFile"
echo -e "\n\n\n" >> "$logFile"

echo ''


python3 main.py \
--dataPath /Users/jessicaren/Downloads/research/Data/dataOrganize/ftf-crsDict.csv \
--islinear 0 \
--model CKCC \
--majorVec $majorVec \
--trainingTerms ['200970','201010','201070','201110','201170','201210','201270','201310','201370','201410','201470','201510','201570'] \
--testTerms ['201610'] \
--paraDictPath "$paraDictPath" \
--userFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/userFile-ftf-ck.csv \
--itemFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/itemFile-ftf-ck.csv \
--trainFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/trainFile-ftf-ck.json \
--testFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/testFile-ftf-ck.json \
--ifRead 0 \
--ifFTF 1 \
--logFile "$logFile" \
--lf 0.2 \
--coCrsSum 0




echo -e "\n\n\n" >> "$logFile"

date >> "$logFile"

echo -e "\n\n\n" >> "$logFile"
