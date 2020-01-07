#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <algorithm>
#include <queue>
#include <vector>
#include <fstream>
#include <list>
#include <iomanip>
#include "CellDetector.hpp"


int main(int argc, char **argv)
{
  using namespace cv;
  //std::ios_base::sync_with_stdio(false);

  CellDetector detector;

  std::ifstream config("config.txt");

  if (config.is_open())
  {
    std::cout << "Reading from config file" << std::endl;

    std::string tmp;
    std::getline(config, tmp);

    CellDetector::Parameters params;
    config >> params.minimumCellPixels;
    config >> params.resizeAmount;
    config >> params.binaryThreshold;
    //TODO: read seperate blue / red thresholds
    detector.setParameters(params);
  }
  else
  {
    std::cout << "No config file found, using default configuration" << std::endl;
  }
	
  std::ofstream log("log.txt");


  for (int i = 1; i <= 92; i++)
  {
    std::string filename = "1 (" + std::to_string(i) + ").jpg";
    std::ifstream imgFile(filename);

    if (imgFile.is_open())
    {
      imgFile.close();
      Mat src_img = imread(filename);
      detector.detect(src_img, log, filename);
    }
  }

	return 0;
}