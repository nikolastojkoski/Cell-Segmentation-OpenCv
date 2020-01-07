#ifndef CELLDETECTOR_HPP
#define CELLDETECTOR_HPP

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <list>
#include <iostream>

using namespace cv;

struct point_t
{
  int row, col;
};

enum Color_t{RED, BLUE};

struct CellGroup
{
  double cells, cellsRed, cellsBlue;
  int minRow, maxRow, minCol, maxCol;
  size_t size;
  unsigned int sizeRed;
  double redToBlueRatio;
  point_t center;
  std::list<point_t> *pixels;
  bool detectedWithWatershed = false;
};

class CellDetector
{
public:
	struct Parameters
	{
		double resizeAmount = 0.2; //deprecated
		double binaryThreshold = 16.0; //deprecated
		int minimumCellPixels = 20;
	};

	CellDetector() = default;
	CellDetector(const CellDetector::Parameters &params);
	virtual ~CellDetector() = default;

	void setParameters(const CellDetector::Parameters &params);

  void detect(const Mat &image, std::ostream & log, const std::string &imgFileName);
	//std::list<CellGroup> *detect(const Mat &image, std::ostream &log);


  //void watershedMethod(std::list<CellGroup>::iterator &it);


private:
	CellDetector::Parameters params_;
  cv::Mat binaryRGB_, binaryRED_, rgb_resized_, rgb_gray_resized_, rgb_resized_blackBackground_;
  cv::Mat groupLabels_, rgb_resized_contours_;
  unsigned int medianCellArea_;
  int __blockSize, __offset;
  std::string _imgFileName;

	void processImage(const Mat &src_rgb, std::ostream &log);
  Mat getBinaryImage(const Mat &gray);
  std::list<CellGroup> *findPixelGroups();
  void filterDetections(std::list<CellGroup> *groups, std::ostream &log);
  void calculateCellCenters(std::list<CellGroup> *groups);
  void displayDetections(const std::list<CellGroup> * groups);
  void watershedDetection(std::list<CellGroup> *groups);

  void labelDetections(const std::list<CellGroup> *groups, std::ostream &log);

  std::list<point_t> *findBackground();
  std::list <point_t> *backgroundPixels_;
  Mat isFromWatershed_;

};




#endif
