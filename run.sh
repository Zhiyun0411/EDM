mainmajorVec=(AIT BIOL CEIE CPE CS PSYC)
for i in "${mainmajorVec[@]}"
do
  majorVec="[$i]"
  paraDictPath="/Users/jessicaren/Downloads/research/Project04/03 myCode/organizeCode/paraDict/paraDict-K3-decay-2-l2-2-l1-2.json"
  #. run-mf.sh
  #. run-ck.sh
  #. run-tr-mf.sh
  #. run-tr-ck.sh
  . run-ale.sh
  #. run-ckcc.sh
  #. run-ncf.sh
done
