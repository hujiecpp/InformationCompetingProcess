mkdir -p data
cd data

if [ "$1" = "dsprites" ]; then
    git clone https://github.com/deepmind/dsprites-dataset.git
    cd dsprites-dataset
    rm -rf .git* *.md LICENSE *.ipynb *.gif *.hdf5

elif [ "$1" = "CelebA" ]; then
    unzip img_align_celeba.zip
    mkdir CelebA
    mv img_align_celeba CelebA
fi
