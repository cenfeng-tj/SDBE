#!/bin/bash
mkdir train
tar xvf ILSVRC2012_img_train.tar -C train
find train -name "*.tar" | while read NAME ; do mkdir -p "${NAME%.tar}"; tar -xvf "${NAME}" -C "${NAME%.tar}"; rm -f "${NAME}"; done
mkdir val
tar xvf ILSVRC2012_img_val.tar -C val
#wget https://raw.githubusercontent.com/jkjung-avt/jkjung-avt.github.io/master/assets/2017-12-01-ilsvrc2012-in-digits/valprep.sh
cd val
bash ../valprep.sh
cd ..

