#!/bin/bash

BASE_DIR="$(cd -P "$(dirname "${BASH_SOURCE[0]}")" > /dev/null 2>&1 && pwd)"
BASE_DIR="$( realpath "${BASE_DIR}/../..")"


#ls $PWD/*.png | awk -F'[/\\.]' 'BEGIN{OFS=",";print "Path,ID,Y"}{print $0,$6,1}'
python3 DevideTrainTest.py -i /mnt/e/myCode/Data/List/mnist/Data_info_mnist_test \
    -B 0 \
    -F 1 \
    -R "4:1:1" \
    -N mnist_test \
    -O /mnt/e/myCode/Model_save/MNIST2/List \


exit

ls $PWD/SF/*/* | grep -i "DCM$" | awk -F'[/\\.]' 'BEGIN{OFS=",";print "Path,ID,Y"}{Y=0; if(/1_dcm/){Y=1}; print $0,$7,Y}' > List/Data_info_raw

for i in $(ls $PWD/SF/*/* | grep -i "json$" );do
    cp "$i" /soxhenry/Scaphoid/Data/JSON/
done

for i in $(ls $PWD/CropImage/* | grep -i "png$"); do
    NewName=$(echo $i | awk -F'[/_]' '{print $6}')
    mv $i $PWD/CropImage/Seg_${NewName}.png
done

ls $PWD/CropImage/* | grep -i "png$" | awk -F'[/_\\.]' 'BEGIN{OFS=",";print "Path,ID,Y"}{Y=$7;if(/\/[RL]_/){Y=0};print $0,$0,Y}' > List/Data_info_seg





