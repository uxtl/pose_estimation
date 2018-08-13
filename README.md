# Pose Estimation
some accelerating methods with less accuracy loss

## Method 1
Use mobileNetV2 instead of ResNet. 

## Method 2
Use optical flow or tracking to get more information to increase accuracy with little overhead. Then we can use less layers while training and inferencing.

## Method 3
In real-time video pose estimation, calculate mean pose and modify a little each frame with a simple network
