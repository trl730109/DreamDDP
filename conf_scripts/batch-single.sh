#dnns=( "resnet50" "googlenet" "vgg16i" "alexnet" "inceptionv4" )
#dnns=( "densenet161" "densenet201" )
dnns=( "resnet152" )
for dnn in "${dnns[@]}"
do
    dnn=$dnn max_epochs=1 ./single.sh
done
