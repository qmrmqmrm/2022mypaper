#!/bin/sh

cd train
if [ -d framework ]; then
    rm -d framework
fi
cd ..

cd model
if [ -d framework ]; then
    rm -d framework
fi

cd ..

cd utils
if [ -d framework ]; then
    rm -d framework
fi
cd ..

cd dataloader
if [ -d framework ]; then
    rm -d framework
fi
cd ..

