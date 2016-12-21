/*
 * Usage of the CUDA version:
 * 
 * TestVideo: https://goo.gl/KDyGDZ
 * wget https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades_cuda/haarcascade_frontalface_default.xml
 * mkdir output
 * g++ face_detect_and_save_cuda.cpp -o face_detect_and_save_cuda `pkg-config --cflags --libs opencv` -std=c++11 -DDEBUG
 * cd output
 * ../face_detect_and_save_cuda ../TestVideo.mkv label
 * check output dir for images and output video
 */

/**
 * @brief face_detect_and_save_cuda Fast prototype for face extraction from videos
 *
 * @file face_detect_and_save_cuda.cpp
 * @author  Renátó Besenczi <renato.besenczi@gmail.com>
 * @version 0.0.3
 *
 * @section LICENSE
 *
 * Copyright (C) 2016 Renátó Besenczi, besenczi.renato@inf.unideb.hu
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * @section DESCRIPTION
 * face_detect_and_save_cuda
 *
 * desc
 *
 * CUDA version
 */

#include "opencv2/opencv.hpp"
#include "opencv2/cudaobjdetect.hpp"
#include <string>

using namespace cv;

String cascade_file = "../haarcascade_frontalface_default.xml";

int main(int argc, char** argv){

  if (argc != 3){
    std::cout << "./appname videofile label" << std::endl;
    exit(0);
  }
  
  VideoCapture cap;
  cap.open(argv[1]);

  if(!cap.isOpened()){
    std::cout << "Can't open video" << std::endl;
    return -1;
  }

int image_counter = 0;
cuda::printShortCudaDeviceInfo (cv::cuda::getDevice());

#ifdef DEBUG
  std::cout << "Frame size: " << cap.get(CV_CAP_PROP_FRAME_WIDTH) << "x"
    << cap.get(CV_CAP_PROP_FRAME_HEIGHT) << '\n'
    << "FPS: " << cap.get(CV_CAP_PROP_FPS) << '\n'
    << "Codec in FourCC: " << cap.get(CV_CAP_PROP_FOURCC) << '\n'
    << "Total frames: " <<cap.get(CV_CAP_PROP_FRAME_COUNT) << '\n'
    << "Files: " << std::endl;
      
    VideoWriter vwr ("TestVideo_output.mkv", VideoWriter::fourcc('D','A','V','C'), 
      cap.get(CV_CAP_PROP_FPS), Size ( cap.get(CV_CAP_PROP_FRAME_WIDTH), cap.get(CV_CAP_PROP_FRAME_HEIGHT) ), true);
#endif

  Ptr<cuda::CascadeClassifier> cascadeClassifier = cuda::CascadeClassifier::create(cascade_file);
  cascadeClassifier->setMinObjectSize(Size(250,250));

  while(1){
    Mat frame, grey;
    std::vector<Rect> detected_objects;

    cap >> frame;

    if(frame.empty()){
      break;
    }

    cvtColor(frame, grey, COLOR_BGR2GRAY);

    cuda::GpuMat image_gpu(grey);
    cuda::GpuMat objbuf;
    cascadeClassifier->detectMultiScale(image_gpu, objbuf);
    cascadeClassifier->convert(objbuf, detected_objects);

    for(int i = 0; i < (int)detected_objects.size(); i++){

      std::string filename(argv[2]);
      filename += std::to_string(image_counter++);
      filename += ".jpg";

#ifdef DEBUG
      std::cout << filename << " ";
      Mat debugframe(frame);
      for(int i = 0; i < (int)detected_objects.size(); i++)
	       rectangle(debugframe, detected_objects.at(i), 255, 2);
      vwr << debugframe;
#endif

      Mat selected(frame,detected_objects[i]);
      Mat image;
      resize(selected,image,Size(300,300));

      imwrite(filename, image);
    }

#ifdef DEBUG
    if (detected_objects.size()==0)
      vwr<<frame;
#endif
    
  }
  std::cout << "\nExtracted frames: " << image_counter << std::endl;
  return 0;
}
