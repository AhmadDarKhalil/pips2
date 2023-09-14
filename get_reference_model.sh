#!/bin/bash

echo "downloading the model from dropbox..."
wget https://www.dropbox.com/scl/fi/8jaym8sycfmst46cinshj/reference_model.tar.gz?rlkey=ec9igxl3i57llwxubb294syf9

echo "cleaning the filename..."
mv reference_model.tar.gz?rlkey=ec9igxl3i57llwxubb294syf9 reference_model.tar.gz

echo "extracting from tar..."
tar -xvf reference_model.tar.gz

echo "deleting the tar..."
rm -v reference_model.tar.gz

echo "done"