#!/bin/bash
sudo rm -R lib
rm ./phantoms/lowdose/*.vol.bin
cd build
make && make install
