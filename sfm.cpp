#include <opencv2/features2d.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <vector>
#include <iostream>
#include <map>
#include <fstream>
#include <cassert>

#include <gtsam/geometry/Point2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/slam/PriorFactor.h>
#include <gtsam/slam/ProjectionFactor.h>
#include <gtsam/slam/GeneralSFMFactor.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/DoglegOptimizer.h>
#include <gtsam/nonlinear/Values.h>
//including the GTSAM library

using namespace std;

 
int fx = 725;
double cx = 295;
int fy = 725;
double cy = 230;

const int IMAGE_DOWNSAMPLE = 4; // downsample the image to speed up processing
const double FOCAL_LENGTH = fx / IMAGE_DOWNSAMPLE; // focal length in pixels, after downsampling, guess from jpeg EXIF data
const int MIN_LANDMARK_SEEN = 3; // minimum number of camera views a 3d point (landmark) has to be seen to be used

const std::string IMAGE_DIR = "desk/";

//define IMAGES as a vector of the 5 images
const std::vector<std::string> IMAGES = {
    "img1.JPG",
    "img2.JPG",
    "img3.JPG",
    "img4.JPG",
    "img5.JPG"
};



struct SFM_Helper
{
    struct ImagePose
    {
        cv::Mat img; // down sampled image used for display
        cv::Mat desc; // feature descriptor
        std::vector<cv::KeyPoint> kp; // keypoint

        cv::Mat T; // 4x4 pose transformation matrix, consisting of R(3*3 matrix) and T(3 dimensional vector) 
        cv::Mat P; // 3x4 projection matrix

        // alias to clarify map usage below
        using kp_idx_t = size_t;
        using landmark_idx_t = size_t;
        using img_idx_t = size_t;

        std::map<kp_idx_t, std::map<img_idx_t, kp_idx_t>> kp_matches; // keypoint matches in other images
        std::map<kp_idx_t, landmark_idx_t> kp_landmark; // setpoint to 3d points

        // helper
        kp_idx_t& kp_match_idx(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx][img_idx]; };
        bool kp_match_exist(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx].count(img_idx) > 0; };

        landmark_idx_t& kp_3d(size_t kp_idx) { return kp_landmark[kp_idx]; }
        bool kp_3d_exist(size_t kp_idx) { return kp_landmark.count(kp_idx) > 0; }
    };

    // 3D point
    struct Landmark
    {
        cv::Point3f pt;
        int seen = 0; // how many cameras have seen this point
    };

    std::vector<ImagePose> img_pose;
    std::vector<Landmark> landmark;
};



int main(int argc, char **argv)
{
    SFM_Helper SFM;

    // Find matching features
    {
        using namespace cv;

        //create the AKAZE object, which is a feature extractor
        Ptr<AKAZE> feature = AKAZE::create();
        //use the BruteForce-Hamming to match the features
        Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
//matcher is the object used for the class "DescriptorMatcher"

        namedWindow("img", WINDOW_NORMAL);

        // Extract features
        for (auto f : IMAGES) {
            SFM_Helper::ImagePose a;

            Mat img = imread(IMAGE_DIR + f);
            assert(!img.empty());

            resize(img, img, img.size()/IMAGE_DOWNSAMPLE);
            a.img = img;
            cvtColor(img, img, COLOR_BGR2GRAY);

            feature->detect(img, a.kp);
            feature->compute(img, a.kp, a.desc);
            //in each images we have deteted the features and computed the feature vectors

            SFM.img_pose.emplace_back(a);
            //adds the image "a" at the end of the current vector, img_pose, that is a vector of the type ImagePose
        }

        // Match features between all images
        for (size_t i=0; i < SFM.img_pose.size()-1; i++) {
            auto &img_pose_i = SFM.img_pose[i];
            //auto is a keyword used to automatically detect what data type we are dealing with
            //hence, now img_pose_i is the same as img_pose[i]
            for (size_t j=i+1; j < SFM.img_pose.size(); j++) {
                auto &img_pose_j = SFM.img_pose[j];
                vector<vector<DMatch>> matches;
                vector<Point2f> src, dst;
                vector<uchar> mask;
                vector<int> i_kp, j_kp;

                // 2 nearest neighbour match
                matcher->knnMatch(img_pose_i.desc, img_pose_j.desc, matches, 2);
//matches the descriptors of img-pose_i image with those of img_pose_j


/*
DMatch.trainIdx - Index of the descriptor in train descriptors
DMatch.queryIdx - Index of the descriptor in query descriptors
DMatch.imgIdx - Index of the train image.
*/
//iterates through each match found and only if the matches are "close enough", (that is the score of the features being matched is not too different), the match is pushed into src and dst vectors, which are vectors of points(2f)
                for (auto &m : matches) {
                    if(m[0].distance < 0.7*m[1].distance) {
//src and dst are vectors of 2d points, we push back those points in the images, which pass the ratio test                  
                        src.push_back(img_pose_i.kp[m[0].queryIdx].pt);
                        dst.push_back(img_pose_j.kp[m[0].trainIdx].pt);

//i_kp and j_kp are vectors of integers,we are pushing the index of the "match" that pass the ratio test in this vector
                        i_kp.push_back(m[0].queryIdx);
                        j_kp.push_back(m[0].trainIdx);
                    }
                }

//now src and dst contain the list of points that qualify as good feature points and are present in both the images
               
                // Filter bad matches using fundamental matrix constraint
                findFundamentalMat(src, dst, FM_RANSAC, 3.0, 0.99, mask);

///////////IDK about the code given below
               /*
 cv::Mat::clone (       )   const
Creates a full copy of the array and the underlying data.

The method creates a full copy of the array. The original step[] is not taken into account. 
So, the array copy is a continuous array occupying total()*elemSize() bytes.
               */

                /*
 cv::Mat::push_back (   const _Tp &     elem    )   
Adds elements to the bottom of the matrix.

The methods add one or more elements to the bottom of the matrix. They emulate the corresponding method of the STL vector class.
When elem is Mat , its type and the number of columns must be the same as in the container matrix.
                */
                Mat canvas = img_pose_i.img.clone();
                canvas.push_back(img_pose_j.img.clone());

                for (size_t k=0; k < mask.size(); k++) {
                    if (mask[k]) {
                        img_pose_i.kp_match_idx(i_kp[k], j) = j_kp[k];
                        img_pose_j.kp_match_idx(j_kp[k], i) = i_kp[k];

                        line(canvas, src[k], dst[k] + Point2f(0, img_pose_i.img.rows), Scalar(0, 0, 255), 2);
                    }
                }

                int good_matches = sum(mask)[0];
                assert(good_matches >= 10);

                cout << "Feature matching " << i << " " << j << " ==> " << good_matches << "/" << matches.size() << endl;

                resize(canvas, canvas, canvas.size()/2);

                imshow("img", canvas);
                waitKey(1);
            }
        }
    }

    // Recover motion between previous to current image and triangulate points
    {
        using namespace cv;

        // Setup camera matrix
       /* double cx = SFM.img_pose[0].img.size().width/2;
        double cy = SFM.img_pose[0].img.size().height/2;
*/
        Point2d pp(cx, cy);

        Mat K = Mat::eye(3, 3, CV_64F);

        K.at<double>(0,0) = FOCAL_LENGTH;
        K.at<double>(1,1) = FOCAL_LENGTH;
        K.at<double>(0,2) = cx;
        K.at<double>(1,2) = cy;

        cout << endl << "initial camera matrix K " << endl << K << endl << endl;

        SFM.img_pose[0].T = Mat::eye(4, 4, CV_64F);
        SFM.img_pose[0].P = K*Mat::eye(3, 4, CV_64F);

        for (size_t i=0; i < SFM.img_pose.size() - 1; i++) {
            auto &prev = SFM.img_pose[i];
            auto &cur = SFM.img_pose[i+1];

            vector<Point2f> src, dst;
            vector<size_t> kp_used;

//for loop iterates through all the keypoints of the "prev" image
            for (size_t k=0; k < prev.kp.size(); k++) {    
//bool kp_match_exist(size_t kp_idx, size_t img_idx) { return kp_matches[kp_idx].count(img_idx) > 0; };
               
                //if the kth keypoiny of the prev image(ith image), has a corresponding match in the cur image((i+1)th image) then enter the if condition
                if (prev.kp_match_exist(k, i+1)) {
                    //the index of the matching match in the (i+1)th image is stored in match_idx
                    size_t match_idx = prev.kp_match_idx(k, i+1);

                    src.push_back(prev.kp[k].pt);
                    //hence, push back the kth keypoint in the ith image in the src array
                    //hence, push back the keypoint corresponding to the (match_index)th index om the (i+1)th image in the dst array
                    dst.push_back(cur.kp[match_idx].pt);

//we push back the indices of the keypoints for which matches exist in the kp_used vector of integers
                    kp_used.push_back(k);
                }
            }

            Mat mask;

            // NOTE: Recovering pose from dst to src.....
            Mat E = findEssentialMat(dst, src, FOCAL_LENGTH, pp, RANSAC, 0.999, 1.0, mask);
            Mat local_R, local_t;

/*
	 mask : Input/output mask for inliers in points1 and points2.
	 If it is not empty, then it marks inliers in points1(dst) and points2(src) for then given essential matrix E. 
	 Only these inliers will be used to recover pose.
	 In the output mask only inliers which pass the cheirality check
*/
			//the recoverPose function recovers the R and t matrices from the Essential matrix
            
//if we are in the ith iteration of the for loop, src is the array of points in the img_pose[i] and dst is the corresponding array of points in the img_pose[i+1]
            recoverPose(E, dst, src, local_R, local_t, FOCAL_LENGTH, pp, mask);

            //rowRange is the Range of the m rows to take. 
            /////////*************The range start is inclusive and the range end is exclusive.**************///////// 
            //Use Range::all() to take all the rows.
            
            // local tansform
            Mat T = Mat::eye(4, 4, CV_64F);
            local_R.copyTo(T(Range(0, 3), Range(0, 3)));
            local_t.copyTo(T(Range(0, 3), Range(3, 4)));

            // accumulate transform
            cur.T = prev.T*T;

            // make projection matrix
            Mat R = cur.T(Range(0, 3), Range(0, 3));
            Mat t = cur.T(Range(0, 3), Range(3, 4));

            //P is a (3*4) matrix, which is a product of the intrinsic and the extrinsic camera parameters
            Mat P(3, 4, CV_64F);

            //initially, make P equal to T and then multiply P by K(intrinsic matrix) to get the real P(projection matrix)
            P(Range(0, 3), Range(0, 3)) = R.t();
            P(Range(0, 3), Range(3, 4)) = -R.t()*t;
            
            P = K*P;
            //P, the projection matrix is obtained by multiplying K(intrinsic matrix) by P(only the extrinsic matrix till now)

            cur.P = P;

/*
void triangulatePoints(InputArray projMatr1, InputArray projMatr2, InputArray projPoints1, InputArray projPoints2, OutputArray points4D)
*/
/*
Parameters:	
projMatr1 – 3x4 projection matrix of the first camera.
projMatr2 – 3x4 projection matrix of the second camera.
projPoints1 – 2xN array of feature points in the first image. In case of c++ version it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
projPoints2 – 2xN array of corresponding points in the second image. In case of c++ version it can be also a vector of feature points or two-channel matrix of size 1xN or Nx1.
points4D – 4xN array of reconstructed points(3d points in the real world) in homogeneous coordinates
note that in points4D, ****************HOMOGENUOUS coordinates************ should be used
*/

            Mat points4D;
            //we dont know the scale as of now, because the t vector in T is always a unit vector
            //this is triangulation from 2 images
            triangulatePoints(prev.P, cur.P, src, dst, points4D);

            // Scale the new 3d points to be similar to the existing 3d points (landmark)
            // Use ratio of distance between pairing 3d points
            if (i > 0) {
                double scale = 0;
                int count = 0;

                Point3f prev_camera;//to take the translational position vector of the camera in the previous image(prev) that is the ith image

                prev_camera.x = prev.T.at<double>(0, 3);
                prev_camera.y = prev.T.at<double>(1, 3);
                prev_camera.z = prev.T.at<double>(2, 3);

                vector<Point3f> new_pts;
                vector<Point3f> existing_pts;

                for (size_t j=0; j < kp_used.size(); j++) {
                    size_t k = kp_used[j];
                    //k is the keypoint index(out of all keypoints detected initially) of the jth useful keypoint
                    //mask is the matrix which was a parameter of the recoverPose() function used before in the code
                    if (mask.at<uchar>(j) && prev.kp_match_exist(k, i+1) && prev.kp_3d_exist(k)) {
                        Point3f pt3d;

//this is basically converting from the homogenuous coordinates into non homomgenuous coordinate system(real coord system)
                        pt3d.x = points4D.at<float>(0, j) / points4D.at<float>(3, j);
                        pt3d.y = points4D.at<float>(1, j) / points4D.at<float>(3, j);
                        pt3d.z = points4D.at<float>(2, j) / points4D.at<float>(3, j);
/*
landmark_idx_t& kp_3d(size_t kp_idx) { return kp_landmark[kp_idx]; }
bool kp_3d_exist(size_t kp_idx) { return kp_landmark.count(kp_idx) > 0; }
*/
                        size_t idx = prev.kp_3d(k);
                        Point3f avg_landmark = SFM.landmark[idx].pt / (SFM.landmark[idx].seen - 1);

                        new_pts.push_back(pt3d);
                        existing_pts.push_back(avg_landmark);
                    }
                }

                // ratio of distance for all possible point pairing
                // probably an over kill! can probably just pick N random pairs
                for (size_t j=0; j < new_pts.size()-1; j++) {
                    for (size_t k=j+1; k< new_pts.size(); k++) {
                        double s = norm(existing_pts[j] - existing_pts[k]) / norm(new_pts[j] - new_pts[k]);

                        scale += s;
                        count++;
                    }
                }

                assert(count > 0);

                scale /= count;

                cout << "image " << (i+1) << " ==> " << i << " scale=" << scale << " count=" << count <<  endl;

                // apply scale and re-calculate T and P matrix
                local_t *= scale;

                // local tansform
                Mat T = Mat::eye(4, 4, CV_64F);
                local_R.copyTo(T(Range(0, 3), Range(0, 3)));
                local_t.copyTo(T(Range(0, 3), Range(3, 4)));

                // accumulate transform
                cur.T = prev.T*T;

                // make projection ,matrix
                R = cur.T(Range(0, 3), Range(0, 3));
                t = cur.T(Range(0, 3), Range(3, 4));

                Mat P(3, 4, CV_64F);
                
                // in openCv, matrix multiplication is done by * operator
                P(Range(0, 3), Range(0, 3)) = R.t();
                P(Range(0, 3), Range(3, 4)) = -R.t()*t;
                P = K*P;

                cur.P = P;

                triangulatePoints(prev.P, cur.P, src, dst, points4D);
                //better version of triangulatePoints because now we know the scale also and hence, the actual 3D coordinates of the points
            }

            // Out of the triangulated points, Find good triangulated points
            for (size_t j=0; j < kp_used.size(); j++) {
                if (mask.at<uchar>(j)) {
                    size_t k = kp_used[j];
                    size_t match_idx = prev.kp_match_idx(k, i+1);

                    Point3f pt3d;

                    pt3d.x = points4D.at<float>(0, j) / points4D.at<float>(3, j);
                    pt3d.y = points4D.at<float>(1, j) / points4D.at<float>(3, j);
                    pt3d.z = points4D.at<float>(2, j) / points4D.at<float>(3, j);

                    if (prev.kp_3d_exist(k)) {
                        // Found a match with an existing landmark
                        cur.kp_3d(match_idx) = prev.kp_3d(k);

                        SFM.landmark[prev.kp_3d(k)].pt += pt3d;
                        SFM.landmark[cur.kp_3d(match_idx)].seen++;
                    } else {
                        // Add new 3d point
                        SFM_Helper::Landmark landmark;

                        landmark.pt = pt3d;
                        landmark.seen = 2;

                        SFM.landmark.push_back(landmark);

                        prev.kp_3d(k) = SFM.landmark.size() - 1;
                        cur.kp_3d(match_idx) = SFM.landmark.size() - 1;
                    }
                }
            }
        }

        // Average out the landmark 3d position
        for (auto &l : SFM.landmark) {
            if (l.seen >= 3) {
                l.pt /= (l.seen - 1);
            }
        }
    }

    // Create output files for PMVS2
    {
        using namespace gtsam;

        Matrix3 K_refined = result.at<Cal3_S2>(Symbol('K', 0)).K();

        cout << endl << "final camera matrix K" << endl << K_refined << endl;

        // Convert to full resolution camera matrix
        K_refined(0, 0) *= IMAGE_DOWNSAMPLE;
        K_refined(1, 1) *= IMAGE_DOWNSAMPLE;
        K_refined(0, 2) *= IMAGE_DOWNSAMPLE;
        K_refined(1, 2) *= IMAGE_DOWNSAMPLE;

        system("mkdir -p root/visualize");
        system("mkdir -p root/txt");
        system("mkdir -p root/models");

        ofstream option("root/options.txt");

        option << "timages  -1 " << 0 << " " << (SFM.img_pose.size()-1) << endl;;
        option << "oimages 0" << endl;
        option << "level 1" << endl;

        option.close();

        for (size_t i=0; i < SFM.img_pose.size(); i++) {
            Eigen::Matrix<double, 3, 3> R;
            Eigen::Matrix<double, 3, 1> t;
            Eigen::Matrix<double, 3, 4> P;
            char str[256];

            R = result.at<Pose3>(Symbol('x', i)).rotation().matrix();
            t = result.at<Pose3>(Symbol('x', i)).translation().vector();

            P.block(0, 0, 3, 3) = R.transpose();
            P.col(3) = -R.transpose()*t;
            P = K_refined*P;

            sprintf(str, "cp -f %s/%s root/visualize/%04d.jpg", IMAGE_DIR.c_str(), IMAGES[i].c_str(), (int)i);
            system(str);
            //imwrite(str, SFM.img_pose[i].img);


            sprintf(str, "root/txt/%04d.txt", (int)i);
            ofstream out(str);

            out << "CONTOUR" << endl;

            for (int j=0; j < 3; j++) {
                for (int k=0; k < 4; k++) {
                    out << P(j, k) << " ";
                }
                out << endl;
            }
        }

        cout << endl;
        cout << "You can now run pmvs2 on the results eg. PATH_TO_PMVS_BINARY/pmvs2 root/ options.txt" << endl;
    }

	return 0;
}
