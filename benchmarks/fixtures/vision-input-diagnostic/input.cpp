#include <yolo_ros/yolo/utils.hpp>
#include <opencv2/opencv.hpp>
#include <fstream>
#include <iostream>
int main(int argc,char**argv){
 if(argc!=3)return 2;
 auto original=cv::imread(argv[1]);
 auto image=yolo_ros::yolo::utils::letterbox(original,cv::Size(640,640),cv::Scalar(114,114,114));
 cv::imwrite(std::string(argv[2])+"/decoded.png",original);
 cv::imwrite(std::string(argv[2])+"/letterboxed.png",image);
 image=yolo_ros::yolo::utils::bgr_to_rgb(image);
 image.convertTo(image,CV_32FC3,1.0/255.0);
 std::vector<float> data(3*640*640);
 std::vector<cv::Mat> channels(3);
 for(int c=0;c<3;c++)channels[c]=cv::Mat(640,640,CV_32FC1,data.data()+c*640*640);
 cv::split(image,channels);
 std::ofstream out(std::string(argv[2])+"/input.bin",std::ios::binary);
 out.write(reinterpret_cast<char*>(data.data()),data.size()*sizeof(float));
 std::cout<<"OpenCV="<<CV_VERSION<<" width="<<original.cols<<" height="<<original.rows<<std::endl;
 return out?0:1;
}
