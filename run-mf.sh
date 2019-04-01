#!/bin/sh

logFile="/Users/jessicaren/Downloads/research/Project04/03 myCode/organizeCode/logFile-ftf-mf.txt"
echo -e "\n\n\n" >> "$logFile"

date >> "$logFile"


echo -e "\n\n\n" >> "$logFile"
cat "$paraDictPath" >> "$logFile"
echo -e "\n\n\n" >> "$logFile"

echo ''
cat "$paraDictPath"


#201810

python3 main.py \
--dataPath /Users/jessicaren/Downloads/research/Data/dataOrganize/ftf-crsDict.csv \
--islinear 0 \
--model MF \
--majorVec ['CHEM','MATH','PHYS','IT','CS','BIOL'] \
--trainingTerms ['200970','201010','201070','201110','201170','201210','201270','201310','201370','201410','201470','201510','201570','201610','201670','201710','201770'] \
--testTerms ['201810'] \
--paraDictPath "$paraDictPath" \
--userFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/userFile-ftf-temp.csv \
--itemFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/itemFile-ftf-temp.csv \
--instrFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/instrFile-ftf-temp.csv \
--trainFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/trainFile-ftf-temp.json \
--testFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/testFile-ftf-temp.json \
--ifRead 1 \
--ifFTF 1 \
--logFile "$logFile" \
--lf 0.8 \
--coCrsSum 0





#201810
echo -e "\n\n\n" >> "$logFile"

echo ''


python3 main.py \
--dataPath /Users/jessicaren/Downloads/research/Data/dataOrganize/ftf-crsDict.csv \
--islinear 0 \
--model MFCC \
--majorVec ['CHEM','MATH','PHYS','IT','CS','BIOL'] \
--trainingTerms ['200970','201010','201070','201110','201170','201210','201270','201310','201370','201410','201470','201510','201570','201610','201670','201710','201770'] \
--testTerms ['201810'] \
--paraDictPath "$paraDictPath" \
--userFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/userFile-ftf-temp.csv \
--itemFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/itemFile-ftf-temp.csv \
--instrFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/instrFile-ftf-temp.csv \
--trainFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/trainFile-ftf-temp.json \
--testFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/testFile-ftf-temp.json \
--ifRead 1 \
--ifFTF 1 \
--logFile "$logFile" \
--lf 0.8 \
--coCrsSum 0






#201810
echo -e "\n\n\n" >> "$logFile"

echo ''


python3 main.py \
--dataPath /Users/jessicaren/Downloads/research/Data/dataOrganize/ftf-crsDict.csv \
--islinear 1 \
--model MFCC \
--majorVec ['CHEM','MATH','PHYS','IT','CS','BIOL'] \
--trainingTerms ['200970','201010','201070','201110','201170','201210','201270','201310','201370','201410','201470','201510','201570','201610','201670','201710','201770'] \
--testTerms ['201810'] \
--paraDictPath "$paraDictPath" \
--userFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/userFile-ftf-temp.csv \
--itemFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/itemFile-ftf-temp.csv \
--instrFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/instrFile-ftf-temp.csv \
--trainFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/trainFile-ftf-temp.json \
--testFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/testFile-ftf-temp.json \
--ifRead 1 \
--ifFTF 1 \
--logFile "$logFile" \
--lf 0.8 \
--coCrsSum 0



echo -e "\n\n\n" >> "$logFile"

date >> "$logFile"

echo -e "\n\n\n" >> "$logFile"












#201770

echo -e "\n\n\n" >> "$logFile"

echo ''
cat "$paraDictPath"

python3 main.py \
--dataPath /Users/jessicaren/Downloads/research/Data/dataOrganize/ftf-crsDict.csv \
--islinear 0 \
--model MF \
--majorVec ['CHEM','MATH','PHYS','IT','CS','BIOL'] \
--trainingTerms ['200970','201010','201070','201110','201170','201210','201270','201310','201370','201410','201470','201510','201570','201610','201670','201710'] \
--testTerms ['201770'] \
--paraDictPath "$paraDictPath" \
--userFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/userFile-ftf-temp.csv \
--itemFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/itemFile-ftf-temp.csv \
--instrFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/instrFile-ftf-temp.csv \
--trainFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/trainFile-ftf-temp.json \
--testFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/testFile-ftf-temp.json \
--ifRead 1 \
--ifFTF 1 \
--logFile "$logFile" \
--lf 0.8 \
--coCrsSum 0






echo -e "\n\n\n" >> "$logFile"

echo ''

#201770
python3 main.py \
--dataPath /Users/jessicaren/Downloads/research/Data/dataOrganize/ftf-crsDict.csv \
--islinear 0 \
--model MFCC \
--majorVec ['CHEM','MATH','PHYS','IT','CS','BIOL'] \
--trainingTerms ['200970','201010','201070','201110','201170','201210','201270','201310','201370','201410','201470','201510','201570','201610','201670','201710'] \
--testTerms ['201770'] \
--paraDictPath "$paraDictPath" \
--userFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/userFile-ftf-temp.csv \
--itemFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/itemFile-ftf-temp.csv \
--instrFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/instrFile-ftf-temp.csv \
--trainFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/trainFile-ftf-temp.json \
--testFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/testFile-ftf-temp.json \
--ifRead 1 \
--ifFTF 1 \
--logFile "$logFile" \
--lf 0.8 \
--coCrsSum 0







echo -e "\n\n\n" >> "$logFile"

echo ''

#201770
python3 main.py \
--dataPath /Users/jessicaren/Downloads/research/Data/dataOrganize/ftf-crsDict.csv \
--islinear 1 \
--model MFCC \
--majorVec ['CHEM','MATH','PHYS','IT','CS','BIOL'] \
--trainingTerms ['200970','201010','201070','201110','201170','201210','201270','201310','201370','201410','201470','201510','201570','201610','201670','201710'] \
--testTerms ['201770'] \
--paraDictPath "$paraDictPath" \
--userFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/userFile-ftf-temp.csv \
--itemFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/itemFile-ftf-temp.csv \
--instrFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/instrFile-ftf-temp.csv \
--trainFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/trainFile-ftf-temp.json \
--testFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/testFile-ftf-temp.json \
--ifRead 1 \
--ifFTF 1 \
--logFile "$logFile" \
--lf 0.8 \
--coCrsSum 0



echo -e "\n\n\n" >> "$logFile"

date >> "$logFile"

echo -e "\n\n\n" >> "$logFile"












echo -e "\n\n\n" >> "$logFile"

echo ''
cat "$paraDictPath"

#201710
python3 main.py \
--dataPath /Users/jessicaren/Downloads/research/Data/dataOrganize/ftf-crsDict.csv \
--islinear 0 \
--model MF \
--majorVec ['CHEM','MATH','PHYS','IT','CS','BIOL'] \
--trainingTerms ['200970','201010','201070','201110','201170','201210','201270','201310','201370','201410','201470','201510','201570','201610','201670'] \
--testTerms ['201710'] \
--paraDictPath "$paraDictPath" \
--userFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/userFile-ftf-temp.csv \
--itemFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/itemFile-ftf-temp.csv \
--instrFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/instrFile-ftf-temp.csv \
--trainFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/trainFile-ftf-temp.json \
--testFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/testFile-ftf-temp.json \
--ifRead 1 \
--ifFTF 1 \
--logFile "$logFile" \
--lf 0.8 \
--coCrsSum 0






echo -e "\n\n\n" >> "$logFile"

echo ''

#201710
python3 main.py \
--dataPath /Users/jessicaren/Downloads/research/Data/dataOrganize/ftf-crsDict.csv \
--islinear 0 \
--model MFCC \
--majorVec ['CHEM','MATH','PHYS','IT','CS','BIOL'] \
--trainingTerms ['200970','201010','201070','201110','201170','201210','201270','201310','201370','201410','201470','201510','201570','201610','201670'] \
--testTerms ['201710'] \
--paraDictPath "$paraDictPath" \
--userFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/userFile-ftf-temp.csv \
--itemFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/itemFile-ftf-temp.csv \
--instrFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/instrFile-ftf-temp.csv \
--trainFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/trainFile-ftf-temp.json \
--testFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/testFile-ftf-temp.json \
--ifRead 1 \
--ifFTF 1 \
--logFile "$logFile" \
--lf 0.8 \
--coCrsSum 0







echo -e "\n\n\n" >> "$logFile"

echo ''

#201710
python3 main.py \
--dataPath /Users/jessicaren/Downloads/research/Data/dataOrganize/ftf-crsDict.csv \
--islinear 1 \
--model MFCC \
--majorVec ['CHEM','MATH','PHYS','IT','CS','BIOL'] \
--trainingTerms ['200970','201010','201070','201110','201170','201210','201270','201310','201370','201410','201470','201510','201570','201610','201670'] \
--testTerms ['201710'] \
--paraDictPath "$paraDictPath" \
--userFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/userFile-ftf-temp.csv \
--itemFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/itemFile-ftf-temp.csv \
--instrFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/instrFile-ftf-temp.csv \
--trainFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/trainFile-ftf-temp.json \
--testFile /Users/jessicaren/Downloads/research/Project04/03\ myCode/organizeCode/testFile-ftf-temp.json \
--ifRead 1 \
--ifFTF 1 \
--logFile "$logFile" \
--lf 0.8 \
--coCrsSum 0



echo -e "\n\n\n" >> "$logFile"

date >> "$logFile"

echo -e "\n\n\n" >> "$logFile"
