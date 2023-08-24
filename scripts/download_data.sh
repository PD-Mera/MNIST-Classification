mkdir -p ./data

git clone https://github.com/myleott/mnist_png

tar -xvf mnist_png/mnist_png.tar.gz -C ./data

rm -rf mnist_png