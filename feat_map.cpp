//this is code for feature mapping between two images, written in C++
#include <opencv2/opencv.hpp>
#include "opencv2/xfeatures2d.hpp"
#include "opencv2/features2d.hpp"
 
using namespace std;
using namespace cv;
using namespace cv::xfeatures2d;
 
const int MAX_FEATURES = 500;
const float GOOD_MATCH_PERCENT = 0.15f;
 
//these intrinsic camera matrix values are obtained from the results i got from running the camera calibration code
int fx = 725;
int cx = 295;
int fy = 725;
int cy = 230;
//Mat k;//the intrinsic parameter matrix
 
int match_Images(Mat im1, Mat im2, Mat E)
{
   
	// Convert images to grayscale
	Mat im1Gray, im2Gray;
	cvtColor(im1, im1Gray, CV_BGR2GRAY);
	cvtColor(im2, im2Gray, CV_BGR2GRAY);

	// Variables to store keypoints and descriptors
	std::vector<KeyPoint> keypoints1, keypoints2;
	Mat descriptors1, descriptors2;

	// Detect ORB features and compute descriptors.
	Ptr<Feature2D> orb = ORB::create(MAX_FEATURES);
	orb->detectAndCompute(im1Gray, Mat(), keypoints1, descriptors1);
	orb->detectAndCompute(im2Gray, Mat(), keypoints2, descriptors2);

	// Match features.
	std::vector<DMatch> matches;
	Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
	matcher->match(descriptors1, descriptors2, matches, Mat());

	// Sort matches by score
	std::sort(matches.begin(), matches.end());

	// Remove not so good matches
	const int numGoodMatches = matches.size() * GOOD_MATCH_PERCENT;
	matches.erase(matches.begin()+numGoodMatches, matches.end());

	// Draw top matches
	Mat imMatches;
	drawMatches(im1, keypoints1, im2, keypoints2, matches, imMatches);
	imwrite("matches.jpg", imMatches);

	namedWindow("matches",WINDOW_NORMAL);  
	imshow("matches",imMatches);

	// Extract location of good matches
	std::vector<Point2f> points1, points2;

	for( size_t i = 0; i < matches.size(); i++ )
	{
		points1.push_back( keypoints1[ matches[i].queryIdx ].pt );
		points2.push_back( keypoints2[ matches[i].trainIdx ].pt );
	}

	Point2d pp;
	pp.x = cx;
	pp.y = cy;

	Mat R, t, mask;
	//R and t are placeholders for rotational and translational matrices

	E = findEssentialMat(points1, points2, fx , pp, RANSAC, 0.999, 1.0, mask); 
	int a = recoverPose(E, points1, points2, R, t, fx , pp, mask);
	//the rotation matrix(R) and translational matrix(t) have now been flled up with the values and can be used for triangulation
	//hence, the function recoverPose recovers the R and t matrices from the Essential matrix

	/*
	  Note that what we just did only gives us one camera matrix, 
	  and for triangulation, we require two camera matrices. 
	  This operation assumes that one camera matrix is fixed and canonical (no rotation and no translation
	*/
	/*
	  The other camera that we recovered from the essential matrix has moved and rotated in relation to the fixed one. 
	  This also means that any of the 3D points that we recover from these two camera matrices
	  will have the first camera at the world origin point (0, 0, 0).
	*/
    return a;
}
 
 
int main(int argc, char **argv)
{
 
  Mat imReference = imread("query_image.jpeg",CV_LOAD_IMAGE_UNCHANGED);
  
  Mat im = imread("train_image.jpeg",CV_LOAD_IMAGE_UNCHANGED);
 
  if(im.empty())
  {
    cout <<"Could not open or find the image" <<endl;
    cin.get();
    return -1;
  }


  Mat imReg, E;
   
  // Align images
  cout << "Matching the two images ..." << endl; 
  int a = match_Images(im, imReference, E);
  cout << "The number of inliers which passed the check of the function recoverPose is: "<<a<<endl;

  waitKey(0);

  destroyAllWindows();

  return 0; 

}