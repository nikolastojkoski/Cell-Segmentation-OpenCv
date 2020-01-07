#include "CellDetector.hpp"
#include <iostream>
#include <algorithm>
#include <queue>
#include <vector>
#include <fstream>
#include <list>
#include <numeric>
#include <iomanip>
#include <set>

#define IMG_T1
//#define SHOW_DEBUG

void CellDetector::processImage(const cv::Mat &src_rgb, std::ostream &log)
{
  using namespace cv;

  //Get Binary Image from Red and Green Channel
  //note: red will mean red and green

  Mat BGRChannels[3], src_red;
  split(src_rgb, BGRChannels); 
  BGRChannels[0] = Mat::zeros(src_rgb.rows, src_rgb.cols, CV_8UC1);  //remove blue channel
  merge(BGRChannels, 3, src_red);

  //Convert both images to grayscale
  Mat gray_rgb_big, gray_red_big;
  cvtColor(src_rgb, gray_rgb_big, CV_RGB2GRAY);
  cvtColor(src_red, gray_red_big, CV_RGB2GRAY);

  //Resize both images
  //double resizeAmount = 1 / ((double)gray_rgb_big.rows / 640.0);
  double resizeAmount = 1 / ((double)gray_rgb_big.rows / 1024.0);

  std::cout << "Resize amount = " << resizeAmount << std::endl;
  log << "Resize amount = " << resizeAmount << std::endl;

  Mat gray_rgb, gray_red;
  resize(gray_rgb_big, gray_rgb, Size(), resizeAmount, resizeAmount);
  resize(gray_red_big, gray_red, Size(), resizeAmount, resizeAmount);

  //Save image for further use (showing results)
  rgb_gray_resized_ = gray_rgb;
  resize(src_rgb, rgb_resized_, Size(), resizeAmount, resizeAmount);

  //Get binaries

  binaryRGB_ = getBinaryImage(gray_rgb);
  binaryRED_ = getBinaryImage(gray_red);

  //imshow("SourceRGB", rgb_resized_);
  //imshow("binaryRGB", binaryRGB_);
  //imshow("binaryRED", binaryRED_);

}
Mat CellDetector::getBinaryImage(const Mat &gray)
{
  Mat bin, blured;
#ifdef IMG_T1 //for normal images
  GaussianBlur(gray, blured, Size(3, 3), 0);
  adaptiveThreshold(blured, bin, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 131, -1);
  Mat tmp;
  erode(bin, tmp, Mat());
  dilate(tmp, bin, Mat());
#else // for noisy images
  medianBlur(gray, blured, 7);
  adaptiveThreshold(blured, bin, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, 131, -5);
#endif

  //static int i = 1;
  //imwrite("binary" + std::to_string(i++) + ".jpg", bin);
  return bin;
}


void CellDetector::detect(const Mat &image, std::ostream & log, const std::string &imgFileName)
{
  log << " - - - - - - - - - - - - - - - -  - - -" << std::endl;
  _imgFileName = imgFileName;

  //Binarize the source image
  processImage(image, log);

  //Find groups of connected pixels
  std::cout << "FPG1" << std::endl;
  std::list<CellGroup> *groups = findPixelGroups();

  //Analyze groups
  std::cout << "FILTER" << std::endl;
  filterDetections(groups, log);

  //Find background pixel group
  //todo: grupite na pixeli sho se pomali od mozebi 0.2*medianArea popolni gi so bela boja
  //findBackground (which colors the inside of cells)
  //must be after filterDetections because it depends on medianCellArea
  std::cout << "FINDBACKGROUND" << std::endl;
  backgroundPixels_ = findBackground();


  //cells now dont contain any more holes, groups need to be updated
  //for correct usage in watershed
  //easy way (slow):
  //delete groups;
  std::cout << "FPG2" << std::endl;
  groups = findPixelGroups();
  std::cout << "NEW GROUPS SIZE: " << groups->size() << std::endl;
  filterDetections(groups, log);
  //hard way:....

  std::cout << "WATERSHED" << std::endl;
  watershedDetection(groups);

  //Calculate cell centers
  std::cout << "CENTERS" << std::endl;
  calculateCellCenters(groups);
  
  std::cout << "LABELDETECTIONS" << std::endl;
  labelDetections(groups, log);

  //Show centers on image
  std::cout << "DISPLAY" << std::endl;
  displayDetections(groups);

}
std::list<CellGroup> *CellDetector::findPixelGroups()
{
  Mat bin = binaryRGB_;
  Mat binRed = binaryRED_;

  //visited pixel matrix
  std::vector<std::vector<bool> > visited(bin.rows);
  for (size_t i = 0; i < visited.size(); i++)
    visited[i].resize(bin.cols, false);

  //directions for BFS
  int rdir[4] = { 0, 0,  1, -1 };
  int cdir[4] = { -1, 1, 0,  0 };

  //check if coordinate is inside img
  auto inRange = [&](int row, int col)
  {
    if (row < 0 || row >= bin.rows || col < 0 || col >= bin.cols)
      return false;
    return true;
  };

  std::list<CellGroup> *groups = new std::list<CellGroup>();

  for (int i = 0; i < bin.rows; i++)
  {
    std::cout << ".";
    for (int j = 0; j < bin.cols; j++)
    {
      if (visited[i][j] || bin.at<uchar>(i, j) == 0) //pixel is black (background)
        continue;

      groups->emplace_back();
      groups->back().pixels = new std::list<point_t>();

      groups->back().minRow = i;
      groups->back().maxRow = i;
      groups->back().minCol = j;
      groups->back().maxCol = j;


      std::queue<point_t> q;
      q.push(point_t{ i, j });

      while (!q.empty())
      {
        point_t cur = q.front();
        q.pop();

        if (!inRange(cur.row, cur.col))
          continue;
        if (visited[cur.row][cur.col])
          continue;
        if (bin.at<uchar>(cur.row, cur.col) == 0)
          continue;

        visited[cur.row][cur.col] = true;
        groups->back().pixels->push_back(cur);

        groups->back().minRow = min(groups->back().minRow, cur.row);
        groups->back().maxRow = max(groups->back().maxRow, cur.row);
        groups->back().minCol = min(groups->back().minCol, cur.col);
        groups->back().maxCol = max(groups->back().maxCol, cur.col);

        if (binRed.at<uchar>(cur.row, cur.col) != 0) //If pixel is red
        {
          groups->back().sizeRed++;
        }

        for (size_t k = 0; k < 4; k++)
        {
          q.push(point_t{ cur.row + rdir[k], cur.col + cdir[k] });
        }
      }
      groups->back().size = groups->back().pixels->size();
    }
  }

  std::cout << "finished bfs" << std::endl;

  return groups;
}

void CellDetector::filterDetections(std::list<CellGroup> *groups, std::ostream &log)
{
  std::cout << "Getting Statistics" << std::endl;
  std::cout << "minimumCellPixels: " << params_.minimumCellPixels << std::endl;
  log << "Getting Statistics" << std::endl;
  log << "minimumCellPixels: " << params_.minimumCellPixels << std::endl;

  //Remove cells below minimum
  auto it = groups->begin();
  while (it != groups->end())
  {
    //if (it->size < params_.minimumCellPixels)
    if(it->size < 50)
    {
      it = groups->erase(it);
      continue;
    }
    it++;
  }
  std::cout << "Erased groups below minimum" << std::endl;

  //Sort the cellGroups in order to find median
  size_t numGroups = groups->size();
  groups->sort([](const CellGroup &a, const CellGroup &b) { return a.size < b.size; });
  //TEST:
  groups->reverse();
  
  unsigned int medianGroupSize = std::next(groups->begin(), numGroups / 2)->size;
  std::cout << "Median Cell Pixels: " << medianGroupSize << std::endl;
  log << "Median Cell Pixels: " << medianGroupSize << std::endl;
  medianCellArea_ = medianGroupSize;
  
  //Calculate cell color
  for (auto it = groups->begin(); it != groups->end(); it++)
  {
    it->cells = (double)it->size / (double)medianGroupSize;
    //it->cellsRed = it->sizeRed / ((double)medianGroupSize * 0.8);
    it->cellsRed = it->sizeRed / (double)it->size;
    //it->cellsRed = std::min(it->cellsRed, it->cells);
    it->cellsBlue = it->cells - it->cellsRed;
    it->redToBlueRatio = (double)it->sizeRed / (double)it->size;
  }

}
void CellDetector::calculateCellCenters(std::list<CellGroup> *groups)
{
  std::cout << "Calculating cell centers" << std::endl;
  for(auto it=groups->begin(); it != groups->end(); it++)
  {
    it->center.row = std::accumulate(it->pixels->begin(), it->pixels->end(), 0,
      [](unsigned int total, const point_t &pt) { return total + pt.row; }) / it->size;

    it->center.col = std::accumulate(it->pixels->begin(), it->pixels->end(), 0,
      [](unsigned int total, const point_t &pt) { return total + pt.col; }) / it->size;
  }
}
void CellDetector::displayDetections(const std::list<CellGroup> *groups)
{
  std::cout << "Displaying Detections" << std::endl;
  cv::Mat img = rgb_resized_;

  for(auto it = groups->begin(); it != groups->end(); it++)
  {
    std::string information;
    //int red = (int)std::round(it->cellsRed);
    //int blue = (int)std::round(it->cellsBlue);
    //
    //if (red > 0 && blue > 0)
    //  information = std::to_string(red) + "/" + std::to_string(red + blue);
    //else if (blue > 0)
    //  information = std::to_string(blue);
    //else
    //  information = std::to_string(red) + "*";


    information = "+";
    CvScalar color;

    if (it->cellsRed > 0.5)
      color = cvScalar(0, 163, 244);
    else
      color = cvScalar(255, 255, 255);


    //todo: remove (for debugging)
    if (it->detectedWithWatershed) 
    {
      color = cvScalar(0, 255, 0);
    }

    cv::putText(img, information,
      cvPoint(it->center.col, it->center.row),
      cv::FONT_HERSHEY_COMPLEX, 0.3, 
      color, //cvScalar(255,255,255),
      1, CV_AA);
  }


  static int idx = 0;
  idx++;
  std::string filename = "img_detections_" + std::to_string(idx) + ".jpg";
  cv::imwrite(filename, img);

  //cv::namedWindow("Detected Cells", cv::WINDOW_NORMAL);
  //cv::imshow("Detected Cells", img);

}
void non_maxima_suppression(const cv::Mat& image, cv::Mat& mask, bool remove_plateaus) {
  // find pixels that are equal to the local neighborhood not maximum (including 'plateaus')
  cv::dilate(image, mask, cv::Mat());
  cv::dilate(image, mask, cv::Mat());
  cv::compare(image, mask, mask, cv::CMP_GE);

  // optionally filter out pixels that are equal to the local minimum ('plateaus')
  if (remove_plateaus) {
    cv::Mat non_plateau_mask;
    cv::erode(image, non_plateau_mask, cv::Mat());
    cv::compare(image, non_plateau_mask, non_plateau_mask, cv::CMP_GT);
    cv::bitwise_and(mask, non_plateau_mask, mask);
  }
  //imshow("mask1", mask);
}

void CellDetector::watershedDetection(std::list<CellGroup> *groups)
{
  // create matrix for labeling if pixel is part of cell found with watershed
  //todo: mozebi nema potreba ovie raboti da se prat tuka voobspto, proveri pak!
  Mat tmp(rgb_resized_.rows, rgb_resized_.cols, CV_16UC1, cv::Scalar(0));
  isFromWatershed_ = tmp;

  std::list<CellGroup> newCellGroups;
  //make the background of resized src_rgb image black
  rgb_resized_blackBackground_;
  rgb_resized_.copyTo(rgb_resized_blackBackground_);
  for (int i = 0; i < binaryRGB_.rows; i++)
  {
    for (int j = 0; j < binaryRGB_.cols; j++)
    {
      if (binaryRGB_.at<uchar>(i, j) == 0)
      {
        rgb_resized_blackBackground_.at<Vec3b>(Point(j, i)) = Vec3b(0, 0, 0);
      }
    }
  }

  //// Create a window
  //namedWindow("DT Thresh", 1);
  //// create a toolbar
  //createTrackbar("blockSize", "DT Thresh", &_blockSize, 100 , onTrackbarChanged, this);
  //createTrackbar("offset", "DT Thresh", &_offset, 200, onTrackbarChanged, this);
  //createTrackbar("method", "DT Thresh", &_method, 1, onTrackbarChanged, this);
  //createTrackbar("nextGroup", "DT Thresh", &_changed, 1, onNextImage, this);

  for (auto it = groups->begin(); it != groups->end(); )
  {
    //if (it->cells > 1.5)
    if(it->cells > 1.0)
    {
      //_it = it;
      //_end = groups->end();
      //watershedMethod(_it);

      //Create binary image from group pixels with black border
      int originalWidth = it->maxCol - it->minCol + 1;
      int originalHeight = it->maxRow - it->minRow + 1;

      int width = originalWidth + 2*10; // +20 for black frame around img
      int height = originalHeight + 2*10;

      Size size(width, height);
      Mat bin = Mat::zeros(size, CV_8U);
      int offsetRow = it->minRow - 10;
      int offsetCol = it->minCol - 10;
      for (auto it2 = it->pixels->begin(); it2 != it->pixels->end(); it2++)
      {
        bin.at<uchar>(Point(it2->col - offsetCol, it2->row - offsetRow)) = 255;
      }

      //Fill holes inside of cells
      //TODO: find changes between this and binary, if hole is too big, then dont fill it
      //std::vector<std::vector<Point> > gcontours;
      //findContours(bin, gcontours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
      //for (int i = 0; i < gcontours.size(); i++)
      //  drawContours(bin, gcontours, i, 255, -1);

      //Get image block for current group from src_rgb with black border
      Mat img = Mat::zeros(size, rgb_resized_blackBackground_.type());
      for (auto it2 = it->pixels->begin(); it2 != it->pixels->end(); it2++)
      {
        img.at<Vec3b>(Point(it2->col - offsetCol, it2->row - offsetRow)) 
          = rgb_resized_blackBackground_.at<Vec3b>(Point(it2->col, it2->row));
      }

      //imshow("binary filled", bin);
      //imshow("added border", img);

      /* * * Watershed algorithm * * */

      //Perform the distance transform algorithm
      Mat dist;
      distanceTransform(bin, dist, CV_DIST_L2, 5);
      normalize(dist, dist, 0.0, 1.0, NORM_MINMAX);


      Mat tmp1;
      dist.convertTo(tmp1, CV_8UC1, 255.0);
      dist = tmp1;
      
#ifdef SHOW_DEBUG
      imshow("distance transform cellgroup", dist);
#endif
      //Get peaks
      //double min, max;
      //Point pmin, pmax;
      //minMaxLoc(dist, &min, &max, &pmin, &pmax);
      Mat tmp;
      //threshold(dist, tmp, 0.4*max, 255, THRESH_BINARY);
      
      //ORIGINAL:
      //adaptiveThreshold(dist, tmp, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 31, -45);
      //dist = tmp;

      //TEST:
      //adaptiveThreshold(dist, tmp, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, 21, -45);
      //dist = tmp;

      //TEST2:
      //if (_blockSize % 2 == 0)
      //  _blockSize++;

      //if(_method == 0)
      //  adaptiveThreshold(dist, tmp, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, _blockSize, -_offset);
      //else
      //  adaptiveThreshold(dist, tmp, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY, _blockSize, -_offset);
      //dist = tmp;

      //TEST3: calculated blocksize, interpolated offset
      const int minBlockSize = 21, maxBlockSize = 79;
      int additionalOffset = 0;
      int blockSize = std::min(maxBlockSize, 2 * (int)std::sqrt((double)medianCellArea_ / 3.14));
      if (blockSize < minBlockSize)
      {
        additionalOffset = minBlockSize - blockSize;
        blockSize = minBlockSize;
      }
      if (blockSize % 2 == 0)
        blockSize++;

      //lagrange interpolation by points (blockSize, offset): (20,40) (27,60) (48,80) (60, 81)
      auto fOffset = [](double x){
        return ((809.0 * x*x*x) / 776160.0) - ((25931.0 * x*x) / (155232.0))
          + ((82871.0 * x) / 9240.0) - (43602.0 / 539.0);
      };
      int offset = (int)fOffset(blockSize);

      __blockSize = blockSize;
      __offset = offset;
      //std::cout << "blockSize, offset : " << blockSize << " " << offset << std::endl;

      adaptiveThreshold(dist, tmp, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY, blockSize, -offset);
      dist = tmp;

      erode(dist, dist, Mat());
      dilate(dist, dist, Mat());

#ifdef SHOW_DEBUG
      imshow("threshold", dist);
#endif

      //imshow("thresh", dist);
      // Create the CV_8U version of the distance image
      // It is needed for findContours()
      Mat dist_8u;
      dist.convertTo(dist_8u, CV_8U);

      // Find total markers
      std::vector<std::vector<Point> > contours;
      findContours(dist_8u, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

      // Create the marker image for the watershed algorithm
      Mat markers = Mat::zeros(dist.size(), CV_32SC1);

      // Draw the foreground markers
      for (size_t i = 0; i < contours.size(); i++)
        drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i) + 1), -1);
      circle(markers, Point(2, 2), 1, CV_RGB(255,255,255), -1);

      //// Draw the foreground markers
      //for (size_t i = 0; i < contours.size(); i++)
      //  drawContours(markers, contours, i, CV_RGB(i+2, i+2, i+2), -1);
      //  //drawContours(markers, contours, static_cast<int>(i), Scalar::all(static_cast<int>(i) + 1), -1);
      //// Draw the background marker
      //todo: fill backogrund with rgb(1,1,1) ?
      //circle(markers, Point(2, 2), 1, CV_RGB(1, 1, 1), -1);

      Mat mark1 = Mat::zeros(markers.size(), CV_8UC1); //
      markers.convertTo(mark1, CV_8UC1);               //
      //imshow("Before watershed", mark1*10000);

      // Perform the watershed algorithm
      watershed(img, markers);
      
      //todo:remove, just for testing
      Mat mark = Mat::zeros(markers.size(), CV_8UC1); //
      markers.convertTo(mark, CV_8UC1);               //
      //imshow("mark", mark*10000);
      //imshow("img", img);

      std::vector<Vec3b> colors;
      for (size_t i = 0; i < contours.size()+10; i++)
      {
        int b = theRNG().uniform(0, 255);
        int g = theRNG().uniform(0, 255);
        int r = theRNG().uniform(0, 255);
        colors.push_back(Vec3b((uchar)b, (uchar)g, (uchar)r));
      }



      std::vector<CellGroup> newGroups(contours.size());
      for (int i = 0; i < newGroups.size(); i++)
      {
        newGroups[i].pixels = new std::list<point_t>();
      }
      // Create the result image
      Mat dst = Mat::zeros(markers.size(), CV_8UC3);
      // Fill labeled objects with random colors
      for (int i = 0; i < markers.rows; i++)
      {
        for (int j = 0; j < markers.cols; j++)
        {
          int index = markers.at<int>(i, j);
          //std::cout << index << " ";
          //if (index > 0 && index <= static_cast<int>(contours.size()))
          if (index > 0 && index <= contours.size())
          {
            dst.at<Vec3b>(i, j) = colors[index - 1];
            newGroups[index - 1].pixels->push_back(point_t{ i + offsetRow, j + offsetCol});

            if (binaryRED_.at<uchar>(i + offsetRow, j + offsetCol) != 0) //If pixel is red
            {
              newGroups[index - 1].sizeRed++;
            }
          }
          else
            dst.at<Vec3b>(i, j) = Vec3b(0, 0, 0);

          ////ADD AREAS BETWEEN CELLS (-1s) TO CONTROUR IMAGE
          //if (index == -1) //area between cells and whole background?
          //{
          //  if (i > 0 && i < markers.rows - 1 && j>0 && j < markers.cols - 1)
          //  {
          //    std::set<int> neighbours;
                //todo: treba da bidi 8-connected t.e. +dijagonalni
          //    neighbours.insert(markers.at<int>(i - 1, j));
          //    neighbours.insert(markers.at<int>(i + 1, j));
          //    neighbours.insert(markers.at<int>(i, j - 1));
          //    neighbours.insert(markers.at<int>(i, j + 1));

          //    if (neighbours.size() >= 3)
          //    {
          //      rgb_resized_contours_.at<Vec3b>(i + offsetRow, j + offsetCol) = Vec3b(255, 255, 255);
          //    }

          //  }
          //}
          /////////////////////////////////////////////////////
        }
      }

#ifdef SHOW_DEBUG
      imshow("binary filled", bin);
      imshow("added border", img);
      // Visualize the final image
      imshow("Final Result", dst);
      waitKey();
#endif
      //waitKey();

      for (int i = 0; i < newGroups.size(); i++)
      {
        newGroups[i].size = newGroups[i].pixels->size();
        newGroups[i].detectedWithWatershed = true;
        newCellGroups.push_back(newGroups[i]);
      }
      if (newGroups.size() > 0)
      {
        //label group pixels as found with watershed
        //todo: mozebi nema potreba od ova ovde?, proveri pak!
        std::for_each(it->pixels->begin(), it->pixels->end(), [&](const point_t &pt) {
          isFromWatershed_.at<ushort>(pt.row, pt.col) = 1;
        });

        //erase group from list
        it = groups->erase(it);
        continue;
      }
    }
    it++;
  }

  filterDetections(&newCellGroups, std::cout);

  for (auto it = newCellGroups.begin(); it != newCellGroups.end(); it++)
  {
    groups->push_back(*it);
  }

}
void CellDetector::labelDetections(const std::list<CellGroup> *groups, std::ostream &log)
{
  Mat labels(rgb_resized_.rows, rgb_resized_.cols, CV_16UC1, cv::Scalar(0));
  groupLabels_ = labels;

  Mat cellColors;
  groupLabels_.copyTo(cellColors);

  Mat isFromWatershed = isFromWatershed_;
  //groupLabels_.copyTo(isFromWatershed);

  //Label full groups
  //TODO: also take cell colors
  ushort groupNum = 1;
  for (auto it = groups->begin(); it != groups->end(); it++)
  {
    std::for_each(it->pixels->begin(), it->pixels->end(), [&](const point_t &pt) {
      groupLabels_.at<ushort>(pt.row, pt.col) = groupNum;
      
      if (it->cellsRed > 0.5)
        cellColors.at<ushort>(pt.row, pt.col) = 2; //red
      else
        cellColors.at<ushort>(pt.row, pt.col) = 1; //blue

      //if (it->detectedWithWatershed)
      //  isFromWatershed.at<ushort>(pt.row, pt.col) = 1;
    });
    groupNum++;
  }

  //Extract contours
  //todo: optimize by not going through whole image, but only group pixels

  //ORIGINAL: * * ** ** ** * ** * * ** * ** * * ** ** * ** ** 
  rgb_resized_.copyTo(rgb_resized_contours_);
  for (int row = 1; row < groupLabels_.rows - 1; row++)
  {
    for (int col = 1; col < groupLabels_.cols - 1; col++)
    {
      if (groupLabels_.at<ushort>(row, col) != 0 && isFromWatershed.at<ushort>(row, col) == 1
        && (groupLabels_.at<ushort>(row - 1, col) == 0 ||
          groupLabels_.at<ushort>(row + 1, col) == 0 ||
          groupLabels_.at<ushort>(row, col - 1) == 0 ||
          groupLabels_.at<ushort>(row, col + 1) == 0))
      {
        if (cellColors.at<ushort>(row, col) == 1)
          rgb_resized_contours_.at<Vec3b>(row, col) = Vec3b(255, 255, 255);
        else
          rgb_resized_contours_.at<Vec3b>(row, col) = Vec3b(0, 152, 234);
      }
    }
  }
  // * * * ** ** ** * ** * * ** * ** * * ** ** * ** ** * ** *

  //TEST1:
  //rgb_resized_.copyTo(rgb_resized_contours_);
  for (auto it = backgroundPixels_->begin(); it != backgroundPixels_->end(); it++)
  {
    int row = it->row;
    int col = it->col;

    static int dirs_row[4] = { 1, -1, 0, 0 };
    static int dirs_col[4] = { 0, 0, 1, -1 };

    for (int i = 0; i < 4; i++)
    {
      if (groupLabels_.at<ushort>(row + dirs_row[i], col + dirs_col[i]) > 0 &&
        isFromWatershed.at<ushort>(row + dirs_row[i], col + dirs_col[i]) != 1)
      {
        if (cellColors.at<ushort>(row + dirs_row[i], col + dirs_col[i]) == 1)
          rgb_resized_contours_.at<Vec3b>(row, col) = Vec3b(255, 255, 255);
        else
          rgb_resized_contours_.at<Vec3b>(row, col) = Vec3b(0, 152, 234);
        break;
      }
    }
  }

  //cv::namedWindow("With contours", cv::WINDOW_NORMAL);
  //imshow("With contours", rgb_resized_contours_);
  static int i = 1;
  imwrite("detectionsContours" + std::to_string(i++) + ".jpg", rgb_resized_contours_);
  log << "outputFileName: detectionsContours" + std::to_string(i - 1) + ".jpg" << std::endl;
  log << "blockSize, offset: " << __blockSize << ", " << __offset << std::endl;
  log << "imgFileName: " << _imgFileName << std::endl;
}

CellDetector::CellDetector(const CellDetector::Parameters & params)
{
  params_ = params;
}
void CellDetector::setParameters(const CellDetector::Parameters & params)
{
  params_ = params;
}

std::list<point_t> *CellDetector::findBackground()
{
  std::list< std::list<point_t>* > blackPixelGroups;
  
  //visited pixel matrix
  std::vector<std::vector<bool> > visited(binaryRGB_.rows);
  for (size_t i = 0; i < visited.size(); i++)
    visited[i].resize(binaryRGB_.cols, false);

  //directions for BFS
  int rdir[4] = { 0, 0,  1, -1 };
  int cdir[4] = { -1, 1, 0,  0 };

  //check if coordinate is inside img
  auto inRange = [&](int row, int col)
  {
    if (row < 0 || row >= binaryRGB_.rows || col < 0 || col >= binaryRGB_.cols)
      return false;
    return true;
  };

  std::cout << "Searching for background pixel group" << std::endl;

  for (int row = 0; row < binaryRGB_.rows; row++)
  {
    std::cout << ".";
    for (int col = 0; col < binaryRGB_.cols; col++)
    {
      if (!visited[row][col] && binaryRGB_.at<uchar>(row, col) == 0) //pixel is black
      {
        std::list<point_t> *group = new std::list<point_t>();

        std::queue<point_t> q;
        q.push({ row, col });

        while (!q.empty())
        {
          point_t cur = q.front();
          q.pop();

          if (!inRange(cur.row, cur.col))
            continue;
          if (visited[cur.row][cur.col] || binaryRGB_.at<uchar>(cur.row, cur.col) != 0)
            continue;

          visited[cur.row][cur.col] = true;

          //skipping frame pixels for optimization
          if (cur.row > 0 && cur.row < binaryRGB_.rows - 1 &&
            cur.col > 0 && cur.col < binaryRGB_.cols - 1)
          {
            group->push_back(cur);
          }


          for (size_t k = 0; k < 4; k++)
          {
            q.push(point_t{ cur.row + rdir[k], cur.col + cdir[k] });
          }
        }

        blackPixelGroups.push_back(group);
      }
    }
  }
  std::cout << std::endl;

  //ORIGINAL:
  //std::list< std::list<point_t>*>::iterator maxElem = std::max_element(blackPixelGroups.begin(), 
  //  blackPixelGroups.end(), [](const std::list<point_t> *g_a, const std::list<point_t> *g_b) {
  //  return g_a->size() < g_b->size();
  //});
  ////TODO: delete[] not needed pixel groups
  //std::cout << "background area(pixels) = " << (*maxElem)->size() << std::endl;
  //return *maxElem;

  //TEST1:
  blackPixelGroups.sort([](const std::list<point_t> *g_a, const std::list<point_t> *g_b) {
    return g_a->size() < g_b->size();
  });

  //fill holes inside cell
  for (auto it = blackPixelGroups.begin(); it != blackPixelGroups.end(); it++)
  {
    if ((*it)->size() < 0.1 * (double)medianCellArea_)
    {
      std::for_each((*it)->begin(), (*it)->end(), [&](const point_t &pt) {
        binaryRGB_.at<uchar>(pt.row, pt.col) = 255;
      });
    }
    else
    {
      break;
    }
  }

  static int i = 1;
  imwrite("binaryFilled" + std::to_string(i++) + ".jpg", binaryRGB_);

  //return largest black pixel group (background)
  return blackPixelGroups.back();

}



