#!/bin/sh

keras="$1"

cd train
if [ -d framework ]; then
    rm framework
fi
ln -s $keras framework
cd ..

cd model
if [ -d framework ]; then
    rm framework
fi
ln -s $keras framework
cd ..

cd utils
if [ -d framework ]; then
    rm framework
fi
ln -s $keras framework
cd ..

cd dataloader
if [ -d framework ]; then
    rm framework
fi
ln -s $keras framework
cd ..

