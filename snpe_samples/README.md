# Samples for SNPE - Snapdragon Neural Processing Engine SDK (https://developer.qualcomm.com/docs/snpe/)

## Hello classification sample
Sample demonstrates the inference of Image classification model. Requires Model and picture
For loading and pre-processing of picture it uses OpenCV

### How to build
Example of build OpenCV for Android
```
cd ~
git clone git@github.com:opencv/opencv.git
cd opencv
mkdir build_arm_android
export ANDROID_SDK_ROOT=~/Android/Sdk
export ANDROID_NDK_ROOT=~/Android/Sdk/ndk/21.1.6352462/
cmake  -DCMAKE_TOOLCHAIN_FILE=~/Android/Sdk/ndk/21.1.6352462/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_ARM_NEON=ON -DANDROID_NATIVE_API_LEVEL=android-23 -DANDROID_STL=c++_static -DBUILD_SHARED_LIBS=ON -DBUILD_ANDROID_EXAMPLES=OFF -DBUILD_ANDROID_PROJECTS=OFF -DBUILD_JAVA=OFF -DBUILD_JASPER=OFF -DBUILD_ZLIB=ON -DBUILD_opencv_dnn=OFF -DBUILD_opencv_apps=OFF -DBUILD_opencv_java=OFF -DBUILD_opencv_java_bindings_gen=OFF ..
```

To build SNPE classificatino sample you need to set up dependency on OpenCV and SNPE
Been in the directory of current README.md
```
export SNPE_ROOT=~/snpe-1.36.0.746
export OpenCV_DIR=~/opencv/build_arm_android
cd snpe_hello_classification
mkdir build_arm_android
cd build_arm_android
cmake  -DCMAKE_TOOLCHAIN_FILE=~/Android/Sdk/ndk/21.1.6352462/build/cmake/android.toolchain.cmake -DANDROID_ABI=armeabi-v7a -DANDROID_ARM_NEON=ON -DANDROID_NATIVE_API_LEVEL=android-23 -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=ON ..
make
```

Convert model following SNPE guide and prepare imege, deploy to ARM Android device
```
adb shell "mkdir -p /data/local/tmp/snpeexample/bin"
adb shell "mkdir -p /data/local/tmp/snpeexample/lib"
adb push ~/opencv/build_arm_android/lib/*.so /data/local/tmp/snpeexample/lib
adb push ~/snpe-1.36.0.746/lib/arm-android-clang6.0/*.so /data/local/tmp/snpeexample/lib
adb push ~/dldt_tools/snpe_samples/snpe_hello_classification/build_arm_android/snpe_hello_classification /data/local/tmp/snpeexample/bin
# copy model and image to the /data/local/tmp/snpeexample/bin, like
adb push inception_v3_2016_08_28_frozen.dlc /data/local/tmp/snpeexample/bin
adb push ILSVRC2012_val_00000001.JPEG /data/local/tmp/snpeexample/bin
adb shell
cd /data/local/tmp/snpeexample/bin
export LD_LIBRARY_PATH=/data/local/tmp/snpeexample/lib:$LD_LIBRARY_PATH
# preparation is completed
./snpe_hello_classification inception_v3_2016_08_28_frozen.dlc ILSVRC2012_val_00000001.JPEG dsp
```

The result of the work - list of class id and "probabilitied" as well the timing over the sample execution
```
Image parameters: Channels=3, width=500, height=375
Resized image parameters: Channels=3, width=224, height=224
819    0.983893
917    0.0038584
1000    0
Read model: 0.500843 ms
Load of the model to SNPE: 3103.6 ms
Load of the picture: 11.7576 ms
Resize of the picture: 1.59196 ms
Copying data from U8 to float array for input: 2.62635 ms
Createo of the ITensor: 2.80369 ms
Inference: 107.776 ms
Post processing of output results: 4.75613 ms
```