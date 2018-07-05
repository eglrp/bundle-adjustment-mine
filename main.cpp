#include <iostream>
#include <fstream>
#include <ctime>
#include <set>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>
//#include <opencv2/xfeatures2d.hpp>

#include "Eigen.h"
#include "PointCloud.h"
#include "Optimization.h"

using namespace std;
using namespace cv;

// Global variables
std::string DATA_PATH = PROJECT_DIR + std::string("/data/rgbd_dataset_freiburg3_long_office_household/");

// Prototype functions
std::vector<cv::Mat> get_images(std::string);
void detect_features(string method, vector<Mat> images);
void test_detectors(vector<Mat> images);
Matrix4f getExtrinsicsFromQuaternion(std::vector<string>);
std::vector<Vector3f> performTriangulation(std::vector<Vector2f> points2dFrame1, std::vector<Vector2f> points2dFrame2, Matrix4f extrinsicsFrame1, Matrix4f extrinsicsFrame2, Matrix3f intrinsics);

int SKIP_FRAMES = 5;

/*********TO DO****************
- We have to test the different features detector and descriptors (especially many or few features got, matching good or not, speed fast or not)
- Make a diagram with speed of the different algorithms


BUG!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
bug
bug
BUG!!!!
1. TAKE IMAGES first 15 all together for the pose, and the rest skip 5 images each
******************************/

void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}



int main(int argc, char *argv[]){

    std::cout << "PHASE 0: Get data" << std::endl;

    std::vector<std::vector<std::string>> lines;

    //Reading the data to initialize the point clouds for the first 15 frames
    std::ifstream inFile_GT;
    std::string line_file;

    int i = -4, j;

    std::ifstream infile(DATA_PATH + std::string("groundtruth.txt"));
    std::string line;
    int initCounter = 0;
    while (initCounter < 15) {
        std::getline(infile, line);
        if (line.find("#") != std::string::npos) {

        }
        else {
            std::vector<std::string> row_values;
            split(line, ' ', row_values);
            lines.push_back(row_values);
            initCounter++;
        }
    }
    infile.close();
    
    Matrix3f intrinsicMatrix;
    intrinsicMatrix <<  525.0f, 0.0f, 319.5f,
                        0.0f, 525.0f, 239.5f,
                        0.0f, 0.0f, 1.0f;

    // Get rgb & depth images from directory
    std::vector<cv::Mat> rgb_images = get_images("rgb.txt");
    std::vector<cv::Mat> depthImages = get_images("depth.txt");


    //****************DEBUG*************************: BACKPROJECTION WORKING
	/*
    Matrix4f extrinsicMatrix = getExtrinsicsFromQuaternion(lines[0]);
    std::vector<Vector2f> pointCoordinates;
    std::vector<float> depthValues;
    std::vector<cv::Vec3b> colorValues;
    for(i = 0; i < 480; i++){
        for(j = 0; j < 640; j++){
            cv::Point2f point(j,i);
            pointCoordinates.push_back(Vector2f(j, i));
            if ((int)depthImages[0].at<uint16_t>(point) == 0){
                depthValues.push_back(MINF);
                cv:Vec3b tmp(255, 255, 255);
                colorValues.push_back(tmp);
            }
            else{
                cv::Vec3b colors = rgb_images[0].at<cv::Vec3b>(i,j);
                depthValues.push_back((depthImages[0].at<uint16_t>(point) * 1.0f) / 5000.0f);
                colorValues.push_back(colors);
            }
        }       
    }
    PointCloud pointCloud = PointCloud(intrinsicMatrix, extrinsicMatrix);
    pointCloud.setPoints2d(pointCoordinates);
    pointCloud.setDepthMap(depthValues);
    pointCloud.setPoints3d(pointCloud.points2dToPoints3d(intrinsicMatrix, extrinsicMatrix, pointCoordinates, depthValues));
	pointCloud.setColorValues(colorValues);
    pointCloud.generateOffFile("verify");
	*/


    std::cout << "PHASE 1: Finding, matching, discarding outlier keypoints" << std::endl;
    // To test detectors
    //test_detectors(rgb_images);
    
    // Detect keypoints
    std::vector<std::vector<cv::KeyPoint>> keypointsAllImgs;
    cv::Ptr<cv::ORB> detectorORB = cv::ORB::create(2000);

    // Descriptors for keypoints
    std::vector<cv::Mat> descriptorsAllImgs;
    cv::Ptr<cv::BRISK> detectorBRISK = cv::BRISK::create();

    // Matching descriptors, we use NORM_HAMMING because the descriptors are BINARY and so it is faster with this norm
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    int nearest_neighbors = 2;
    const float ratio = 0.5;    //parameter to be tuned

    std::vector<std::vector<std::vector<cv::DMatch>>> allMatches;

    // Detect keypoints and calculate descriptors
    for (i=0; i<rgb_images.size(); i++){
        std::vector<cv::KeyPoint> keypointsImg;
        cv::Mat descriptorsImg;

        detectorORB->detect(rgb_images[i], keypointsImg);
        detectorBRISK->compute(rgb_images[i], keypointsImg, descriptorsImg);

        keypointsAllImgs.push_back(keypointsImg);
        descriptorsAllImgs.push_back(descriptorsImg);

        if (i > 0){
            // Match with kNN algorithm between pair of frames
            std::vector<std::vector<cv::DMatch>> matches;
            matcher.knnMatch(descriptorsAllImgs[i-1], descriptorsAllImgs[i], matches, nearest_neighbors);
            allMatches.push_back(matches);
        }
    }


    //Creating a vector to store goodMatches for first 15 frames
    //This vector is used to initialize point clouds using corresponding depths
    std::vector<std::vector<cv::DMatch>> goodMatchesInitialization;
    //Indices to skip for each frame
    std::vector<std::set<int>> dupIdx;
	//Save good matches among 3 frames for BA
	std::vector<std::vector<Vector3d>> allMatchesBetween3Frames;

    for (int i = 0; i < rgb_images.size(); i++) {
        std::set<int> empty_set;
        dupIdx.push_back(empty_set);
    }

    for (i=0; i<(rgb_images.size()-2); i++){
        std::vector<cv::DMatch> good_matches;

        std::vector<std::vector<cv::DMatch>> matchesFirstThirdFrame;
        matcher.knnMatch(descriptorsAllImgs[i], descriptorsAllImgs[i+2], matchesFirstThirdFrame, nearest_neighbors);
		std::vector<Vector3d> matchesBetween3Frames;
        for (j=0; j<allMatches[i].size(); j++){
            
            // Discard possible outliers, between first and second frame, if the 2 Nearest Neighbors are too close each other
            if (allMatches[i][j][0].distance < ratio * allMatches[i][j][1].distance){

                // Discard possible outliers, between first and third frame, if the 2 Nearest Neighbors are too close each other
                if(matchesFirstThirdFrame[j][0].distance < ratio * matchesFirstThirdFrame[j][1].distance){

                    // Find the same keypoint both in (frame+1) and (frame+2), so that I know it is not noise
                    // Check if the correspondence of first-second and second-third, is equivalent to the first-third one
                    if (allMatches[i + 1][allMatches[i][j][0].trainIdx][0].trainIdx == matchesFirstThirdFrame[j][0].trainIdx) {

                        // Skip duplicate points that already stored in the preview frame
                        if (i == 0 || dupIdx[i].find(allMatches[i][j][0].queryIdx) == dupIdx[i].end()) {                            
                            good_matches.push_back(allMatches[i][j][0]);
                            dupIdx[i + 1].insert(allMatches[i][j][0].trainIdx);
							dupIdx[i + 2].insert(matchesFirstThirdFrame[j][0].trainIdx);
							matchesBetween3Frames.push_back(Vector3d(matchesFirstThirdFrame[j][0].queryIdx, allMatches[i][j][0].trainIdx, matchesFirstThirdFrame[j][0].trainIdx));
                        }
                    }
                }
            }
        }
        goodMatchesInitialization.push_back(good_matches);
		allMatchesBetween3Frames.push_back(matchesBetween3Frames);
        // Visualize matches after removing outliers
        //cv::Mat imgMatches;
        //cv::drawMatches(rgb_images[i], keypointsAllImgs[i], rgb_images[i+1], keypointsAllImgs[i+1], goodMatchesInitialization[i], imgMatches);

        //cv::imshow("Good Matches", imgMatches);
        //cv::waitKey(0);
        //cv::destroyWindow("Good Matches");

    }
 
    std::cout << "PHASE 2: Backprojection" << std::endl;
	//Vector of all unique 3d points
	std::vector<Vector3f> allPoints3D;
	int pointCounter = 0;

    // Initializing the Point Clouds for first 15 frames using good matches and checking their depth values
    std::vector<PointCloud> pointClouds;
	Matrix4f extrinsicMatrix;
	for (i = 0; i < allMatchesBetween3Frames.size(); i++) {
		// Take frame 1 as groundtruth, that we know the camera pose
		if (i == 0 || i == 1) {
			extrinsicMatrix = getExtrinsicsFromQuaternion(lines[i]);
		}
		else {
			extrinsicMatrix = Matrix4f::Identity();
		}
		PointCloud pointCloud = PointCloud(intrinsicMatrix, extrinsicMatrix);
		std::vector<Vector2f> pointCoordinates;
		std::vector<float> depthValues;
		std::vector<cv::Vec3b> colorValues;
		std::vector<int> globalIndices;
		std::vector<Vector2f> pointCoordinates_nextFrame;
		std::vector<Vector2f> pointCoordinates_thirdFrame;
		for (j = 0; j < allMatchesBetween3Frames[i].size(); j++) {
			cv::Point2f point_0 = keypointsAllImgs[i][allMatchesBetween3Frames[i][j][0]].pt;
			cv::Point2f point_1 = keypointsAllImgs[i + 1][allMatchesBetween3Frames[i][j][1]].pt;
			cv::Point2f point_2 = keypointsAllImgs[i + 2][allMatchesBetween3Frames[i][j][2]].pt;
			// Only points where depth is VALID
			if (depthImages[i].at<uint16_t>(point_0) != 0 && 
				depthImages[i + 1].at<uint16_t>(point_1) != 0 &&
				depthImages[i + 2].at<uint16_t>(point_2) != 0) {
				// 2d points
				pointCoordinates.push_back(Vector2f(point_0.x, point_0.y));
				pointCoordinates_nextFrame.push_back(Vector2f(point_1.x, point_1.y));
				//pointCoordinates_thirdFrame.push_back(Vector2f(point_2.x, point_2.y));
				// depth values
				depthValues.push_back((depthImages[i].at<uint16_t>(point_0) * 1.0f) / 5000.0f);
				// color values
				colorValues.push_back(rgb_images[i].at<cv::Vec3b>(point_0));
				// global positions
				globalIndices.push_back(pointCounter++);
			}
		}
		pointCloud.setPoints2d(pointCoordinates);
		pointCloud.setPoints2dNextFrame(pointCoordinates_nextFrame);
		pointCloud.setPoints2dThirdFrame(pointCoordinates_thirdFrame);
		pointCloud.setDepthMap(depthValues);
		pointCloud.setColorValues(colorValues);
		std::vector<Vector3f> points3d = pointCloud.points2dToPoints3d(intrinsicMatrix, extrinsicMatrix, pointCoordinates, depthValues);
		pointCloud.setPoints3d(points3d);
		pointCloud.setGlobalPosition(globalIndices);
		pointClouds.push_back(pointCloud);
		allPoints3D.insert(allPoints3D.end(), points3d.begin(), points3d.end());
	}
	/*Test
	std::cout << pointClouds[5].points3d[6] << endl;
	std::cout << allPoints3D[pointClouds[5].getGlobalPosition(6)] << endl;
	std::cout << endl;
	std::cout << pointClouds[5].points3d[7] << endl;
	std::cout << allPoints3D[pointClouds[5].getGlobalPosition(7)] << endl;
	std::cout << endl;
	std::cout << pointClouds[5].points3d[8] << endl;
	std::cout << allPoints3D[pointClouds[5].getGlobalPosition(8)] << endl;
	std::cout << endl;
	*/

    //TO DO: We have to write everything in ONE UNIQUE FILE OFF
    //std::cout << "PHASE 3: Write on file .off" << std::endl;
    
    //for (int i = 0; i < pointClouds.size(); i++)
        //pointClouds[i].generateOffFile(to_string(i));

    std::cout << "Check if triangulation is working fine " << std::endl;
    PointCloud pointCloudM = pointClouds[0];
    PointCloud pointCloudN = pointClouds[1];

    std::vector<Vector2f> points2d1;
    std::vector<Vector2f> points2d2;

    std::vector<float> groundTruthDepth;

    for ( int p = 0; p < goodMatchesInitialization[0].size(); p++){

        cv::Point2f pointQuery = keypointsAllImgs[0][goodMatchesInitialization[0][p].queryIdx].pt;
        cv::Point2f pointTrain = keypointsAllImgs[1][goodMatchesInitialization[0][p].trainIdx].pt;

//        cv::Mat imgMatches;
//        cv::drawMatches(rgb_images[0], keypointsAllImgs[0], rgb_images[1], keypointsAllImgs[1], goodMatchesInitialization[0], imgMatches);
//        cv::imshow("Good Matches", imgMatches);
//        cv::waitKey(0);
//        cv::destroyWindow("Good Matches");

        if(depthImages[0].at<uint16_t>(pointQuery) != 0){
            groundTruthDepth.push_back(((depthImages[0].at<uint16_t>(pointQuery)* 1.0f) / 5000.0f));
            points2d1.push_back(Vector2f(pointQuery.x, pointQuery.y));
            points2d2.push_back(Vector2f(pointTrain.x, pointTrain.y));
        }

    }

    std::cout << pointCloudM.getCameraExtrinsics() << std::endl;
    std::cout << pointCloudN.getCameraExtrinsics() << std::endl;

    std::vector<Vector3f> triangulatedPoints = performTriangulation(
            points2d1, points2d2, pointCloudM.getCameraExtrinsics(), pointCloudN.getCameraExtrinsics(), pointCloudM.getCameraIntrinsics());

//    for ( int p=0; p < triangulatedPoints.size(); p++){
//        //std::cout << groundTruthDepth[p] << "           " << triangulatedPoints[p].z() << "            " << (triangulatedPoints[p].z()-groundTruthDepth[p]) <<std::endl;
//        std::cout << (triangulatedPoints[p].z()-groundTruthDepth[p]) <<std::endl;
//    }

    // Project the points back in the image 1
    std::vector<Vector2f> projectedAfter;
    Vector3f pointAfter;
    Vector2f point2dAfter;
    Matrix3f rotation = pointCloudM.getCameraExtrinsics().block(0, 0, 3, 3);
    Vector3f translation = pointCloudM.getCameraExtrinsics().block(0, 3, 3, 1);

    for(int p = 0; p < triangulatedPoints.size(); p++){
        pointAfter = rotation * triangulatedPoints[p];
        pointAfter = pointAfter + translation;
        pointAfter = pointCloudM.getCameraIntrinsics() * pointAfter;
        point2dAfter = Vector2f(pointAfter.x()/pointAfter.z(), pointAfter.y()/pointAfter.z());
        projectedAfter.push_back(point2dAfter);
    }

    for ( int p = 0; p < projectedAfter.size(); p++){
        std::cout << points2d1[p].x() << "           " << points2d1[p].y() << std::endl;
        std::cout << projectedAfter[p].x() << "           " << projectedAfter[p].y() << std::endl;
        std::cout << "" << std::endl;
    }


    exit(0);


    std::cout << "PHASE 3: Optimization (NOW ONLY ON POSE AND BETWEEN TWO FRAMES)" << std::endl;
	
    Optimization optimizer;
    optimizer.setNbOfIterations(10);
	for (int i = 0; i < pointClouds.size() - 2; i++) {
		std::cout << "Optimizing frame " << i + 2 << std::endl;
		Matrix4f finalPose = optimizer.estimatePose(pointClouds[i]);
		pointClouds[i + 1].setCameraExtrinsics(finalPose);
		pointClouds[i + 1].computeNewPoints3D();
		std::cout << pointClouds[i + 1].getCameraExtrinsics() << std::endl;
	}
	for (int i = 0; i < pointClouds.size() - 1; i++) {
		pointClouds[i].generateOffFile(to_string(i));
	}

  //  i = 0;
  //  int frames_to_consider = 10;
  //  while (i < (pointClouds.size() - frames_to_consider)) {

  //      std::vector<PointCloud> pointClouds_optimization;
  //      for(j=i; j < (i+frames_to_consider); j++){
  //          PointCloud tmp = pointClouds[j];
  //          pointClouds_optimization.push_back(tmp);
  //      }
  //      std::vector<PointCloud> pointClouds_optimized = optimizer.estimatePose(pointClouds_optimization);
		//for (int k = 0; k < pointClouds_optimized.size(); k++) {
		//	pointClouds_optimized[k].setPoints3d(pointClouds_optimized[k].points2dToPoints3d(pointClouds_optimized[k].getCameraIntrinsics(), pointClouds_optimized[k].getCameraExtrinsics(), pointClouds_optimized[k].getPoints2d(), pointClouds_optimized[k].getDepthMap()));
		//	pointClouds_optimized[k].generateOffFile("opt" + to_string(k));
		//}
  //      i++;
  //  }
}


std::vector<cv::Mat> get_images(std::string dir_base){

    std::vector<cv::Mat> images;
    std::ifstream inFile;
    std::string fileName;

    int i = -1;

    inFile.open(DATA_PATH + dir_base);

    if (!inFile) {
        std::cerr << "Unable to open file" << std::endl;
        exit(1);
    }

    while (std::getline(inFile, fileName)) {
        i++;

        // skip header
        if (i < 3)
            continue;

        // skip frame, since we take one frame every 5 to have larger baseline
        if ((i-3) % 5 != 0)
            continue;            

        // Split string
        std::istringstream stringstream(fileName);
        for(std::string str; stringstream >> fileName; );

        //std::cout << fileName << std::endl;

        cv::Mat img = cv::imread(DATA_PATH + fileName, IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);

        if (!img.data) {
            std::cout << "Image not found." << std::endl;
            exit(1);
        }

        images.push_back(img.clone());

        // TAKE FEW IMAGES DURING DEBUG
        if (i > 200)
            break;
    }

    inFile.close();

    return images;
}


void detect_features(string method, vector<Mat> images) {
    Ptr<Feature2D> detector;
    ofstream outFile(method + "test.txt");
    if (!outFile.is_open()) return;
   /* if (method.compare("SIFT") == 0) {
        detector = xfeatures2d::SIFT::create();
    }
    if (method.compare("SURF") == 0) {
        detector = xfeatures2d::SURF::create();
    }*/
    if (method.compare("ORB") == 0) {
        detector = cv::ORB::create();
    }
    else if (method.compare("FAST") == 0) {
        detector = FastFeatureDetector::create();
    }
    /*else if (method.compare("STAR") == 0) {
        detector = xfeatures2d::StarDetector::create();
    }*/
    else if (method.compare("BRISK") == 0) {
        detector = cv::BRISK::create();
    }
    outFile << "Method: " << method << endl;

    vector<vector<KeyPoint>> keypointsAllImgs;

    // Detection time test
    clock_t start = clock();
    for (int i = 0; i < images.size(); i++) {
        Mat image = images[i];
        vector<KeyPoint> keypoints;
        detector->detect(image, keypoints);
        keypointsAllImgs.push_back(keypoints);
        outFile << "Frame " << i << " Keypoints: " << keypoints.size() << endl;
    }
    clock_t end = clock();
    double time = double(end - start) / CLOCKS_PER_SEC;

    outFile << "Detect Time: " << time << endl;
    outFile.close();
}

void test_detectors(vector<Mat> images) {
    // Compare 6 methods
    string methods[6] = {"ORB", "BRISK", "SURF", "FAST", "STAR", "SIFT" };
    for (int i = 0; i < 6; i++) {
        detect_features(methods[i], images);
    }
    return;
}


Matrix4f getExtrinsicsFromQuaternion(std::vector<string> poses){
    int i, j;
    Matrix4f extrinsics;
    extrinsics.setIdentity();

    Eigen::Quaterniond q(Eigen::Vector4d(std::stod(poses[4]), std::stod(poses[5]), std::stod(poses[6]), std::stod(poses[7])));
    Eigen::Matrix<double, 3, 3> rotation = q.normalized().toRotationMatrix();

    for(i = 0; i < 3; i++){
        for(j = 0; j < 3; j++)
            extrinsics(i,j) = rotation(i,j);
    }
    extrinsics(0,3) = std::stod(poses[1]);
    extrinsics(1,3) = std::stod(poses[2]);
    extrinsics(2,3) = std::stod(poses[3]);

    return extrinsics;
}

std::vector<Vector3f> performTriangulation(std::vector<Vector2f> points2dFrame1, std::vector<Vector2f> points2dFrame2, Matrix4f extrinsicsFrame1, Matrix4f extrinsicsFrame2, Matrix3f cameraIntrinsics){
    // Run the triangulation here
    // Find the matching points for which we do not have the 3d points
    // Triangulate them and add them to the point cloud

    std::cout << "PHASE 4.1: Running the triangulation" <<std::endl;

    // std::vector<Vector2f> points2dFrame1 stores the 2d coordinates of the points in the frame1
    // for which we do not have corresponding 3d point coordinate
    // but the 2d point is a good match w.r.t features

    // std::vector<Vector2f> points2dFrame2 stores the 2d coordinates of the points in the frame2
    // for which we do not have corresponding 3d point coordinate
    // but the 2d point is a good match w.r.t features

    // extrinsicsFrame1 stores the world to camera transformation for the frame1
    // extrinsicsFrame2 stores the world to camera transformation for the frame2
    // intrinsics stores the camera intrinsic parameters

    // This vector stores the 3D point coordinates that we will calculate during the triangulation
    std::vector<Vector3f> points3dTriangulation;

    if(points2dFrame2.size() > 0){

        // Defining Ax=b
        // Defining the A and b for the linear least square solver
        MatrixXf A(4, 3);
        A <<    0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0,
                0.0, 0.0, 0.0;

        VectorXf x(3);
        x <<    0.0, 0.0, 0.0;

        VectorXf b(4);
        b << 0.0, 0.0, 0.0, 0.0;

        std::cout << "PHASE 3.1: Number of points to triangulate: " << points2dFrame2.size() << std::endl;

        // Perform the triangulation to get the 3dPoints for points2dPreviousFrame and points2dCurrentFrame
        for( int g = 0; g < points2dFrame2.size(); g++){

            // Calculate the values to be inserted inside A and b for Ax=b
            A(0, 0) = ((cameraIntrinsics(0,0) * extrinsicsFrame1(0,0)) + (cameraIntrinsics(0,2) * extrinsicsFrame1(2,0)) - (points2dFrame1[g].x() * extrinsicsFrame1(2,0)));
            A(0, 1) = ((cameraIntrinsics(0,0) * extrinsicsFrame1(0,1)) + (cameraIntrinsics(0,2) * extrinsicsFrame1(2,1)) - (points2dFrame1[g].x() * extrinsicsFrame1(2,1)));
            A(0, 2) = ((cameraIntrinsics(0,0) * extrinsicsFrame1(0,2)) + (cameraIntrinsics(0,2) * extrinsicsFrame1(2,2)) - (points2dFrame1[g].x() * extrinsicsFrame1(2,2)));

            A(1, 0) = ((cameraIntrinsics(1,1) * extrinsicsFrame1(1,0)) + (cameraIntrinsics(1,2) * extrinsicsFrame1(2,0)) - (points2dFrame1[g].y() * extrinsicsFrame1(2,0)));
            A(1, 1) = ((cameraIntrinsics(1,1) * extrinsicsFrame1(1,1)) + (cameraIntrinsics(1,2) * extrinsicsFrame1(2,1)) - (points2dFrame1[g].y() * extrinsicsFrame1(2,1)));
            A(1, 2) = ((cameraIntrinsics(1,1) * extrinsicsFrame1(1,2)) + (cameraIntrinsics(1,2) * extrinsicsFrame1(2,2)) - (points2dFrame1[g].y() * extrinsicsFrame1(2,2)));

            A(2, 0) = ((cameraIntrinsics(0,0) * extrinsicsFrame2(0,0)) + (cameraIntrinsics(0,2) * extrinsicsFrame2(2,0)) - (points2dFrame2[g].x() * extrinsicsFrame2(2,0)));
            A(2, 1) = ((cameraIntrinsics(0,0) * extrinsicsFrame2(0,1)) + (cameraIntrinsics(0,2) * extrinsicsFrame2(2,1)) - (points2dFrame2[g].x() * extrinsicsFrame2(2,1)));
            A(2, 2) = ((cameraIntrinsics(0,0) * extrinsicsFrame2(0,2)) + (cameraIntrinsics(0,2) * extrinsicsFrame2(2,2)) - (points2dFrame2[g].x() * extrinsicsFrame2(2,2)));

            A(3, 0) = ((cameraIntrinsics(1,1) * extrinsicsFrame2(1,0)) + (cameraIntrinsics(1,2) * extrinsicsFrame2(2,0)) - (points2dFrame2[g].y() * extrinsicsFrame2(2,0)));
            A(3, 1) = ((cameraIntrinsics(1,1) * extrinsicsFrame2(1,1)) + (cameraIntrinsics(1,2) * extrinsicsFrame2(2,1)) - (points2dFrame2[g].y() * extrinsicsFrame2(2,1)));
            A(3, 2) = ((cameraIntrinsics(1,1) * extrinsicsFrame2(1,2)) + (cameraIntrinsics(1,2) * extrinsicsFrame2(2,2)) - (points2dFrame2[g].y() * extrinsicsFrame2(2,2)));

            b[0] = ((points2dFrame1[g].x() * extrinsicsFrame1(2,3)) - (cameraIntrinsics(0,2) * extrinsicsFrame1(2,3)) - (cameraIntrinsics(0,0) * extrinsicsFrame1(0,3)));
            b[1] = ((points2dFrame1[g].y() * extrinsicsFrame1(2,3)) - (cameraIntrinsics(1,2) * extrinsicsFrame1(2,3)) - (cameraIntrinsics(1,1) * extrinsicsFrame1(1,3)));
            b[2] = ((points2dFrame2[g].x() * extrinsicsFrame2(2,3)) - (cameraIntrinsics(0,2) * extrinsicsFrame2(2,3)) - (cameraIntrinsics(0,0) * extrinsicsFrame2(0,3)));
            b[3] = ((points2dFrame2[g].y() * extrinsicsFrame2(2,3)) - (cameraIntrinsics(1,2) * extrinsicsFrame2(2,3)) - (cameraIntrinsics(1,1) * extrinsicsFrame2(1,3)));

//            A(0, 0) = ((points2dFrame1[g].x() - cameraIntrinsics(0, 2)) * extrinsicsFrame1(2, 0)) - (cameraIntrinsics(0, 0) * extrinsicsFrame1(0, 0));
//            A(0, 1) = ((points2dFrame1[g].x() - cameraIntrinsics(0, 2)) * extrinsicsFrame1(2, 1)) - (cameraIntrinsics(0, 0) * extrinsicsFrame1(0, 1));
//            A(0, 2) = ((points2dFrame1[g].x() - cameraIntrinsics(0, 2)) * extrinsicsFrame1(2, 2)) - (cameraIntrinsics(0, 0) * extrinsicsFrame1(0, 2));
//
//            A(1, 0) = ((points2dFrame1[g].y() - cameraIntrinsics(1, 2)) * extrinsicsFrame1(2, 0)) - (cameraIntrinsics(1, 1) * extrinsicsFrame1(1, 0));
//            A(1, 1) = ((points2dFrame1[g].y() - cameraIntrinsics(1, 2)) * extrinsicsFrame1(2, 1)) - (cameraIntrinsics(1, 1) * extrinsicsFrame1(1, 1));
//            A(1, 2) = ((points2dFrame1[g].y() - cameraIntrinsics(1, 2)) * extrinsicsFrame1(2, 2)) - (cameraIntrinsics(1, 1) * extrinsicsFrame1(1, 2));
//
//            A(2, 0) = ((points2dFrame2[g].x() - cameraIntrinsics(0, 2)) * extrinsicsFrame2(2, 0)) - (cameraIntrinsics(0, 0) * extrinsicsFrame2(0, 0));
//            A(2, 1) = ((points2dFrame2[g].x() - cameraIntrinsics(0, 2)) * extrinsicsFrame2(2, 1)) - (cameraIntrinsics(0, 0) * extrinsicsFrame2(0, 1));
//            A(2, 2) = ((points2dFrame2[g].x() - cameraIntrinsics(0, 2)) * extrinsicsFrame2(2, 2)) - (cameraIntrinsics(0, 0) * extrinsicsFrame2(0, 2));
//
//            A(3, 0) = ((points2dFrame2[g].y() - cameraIntrinsics(1, 2)) * extrinsicsFrame2(2, 0)) - (cameraIntrinsics(1, 1) * extrinsicsFrame2(1, 0));
//            A(3, 1) = ((points2dFrame2[g].y() - cameraIntrinsics(1, 2)) * extrinsicsFrame2(2, 1)) - (cameraIntrinsics(1, 1) * extrinsicsFrame2(1, 1));
//            A(3, 2) = ((points2dFrame2[g].y() - cameraIntrinsics(1, 2)) * extrinsicsFrame2(2, 2)) - (cameraIntrinsics(1, 1) * extrinsicsFrame2(1, 2));
//
//            b[0] = (cameraIntrinsics(0, 0) * extrinsicsFrame1(0, 3)) + ((cameraIntrinsics(0, 2) - points2dFrame1[g].x()) * extrinsicsFrame1(2, 3));
//            b[1] = (cameraIntrinsics(1, 1) * extrinsicsFrame1(1, 3)) + ((cameraIntrinsics(1, 2) - points2dFrame1[g].y()) * extrinsicsFrame1(2, 3));
//            b[2] = (cameraIntrinsics(0, 0) * extrinsicsFrame2(0, 3)) + ((cameraIntrinsics(0, 2) - points2dFrame2[g].x()) * extrinsicsFrame2(2, 3));
//            b[3] = (cameraIntrinsics(1, 1) * extrinsicsFrame2(1, 3)) + ((cameraIntrinsics(1, 2) - points2dFrame2[g].y()) * extrinsicsFrame2(2, 3));

            // Solve the system of equations
            x = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);
//            x = A.colPivHouseholderQr().solve(b);
            // Push back the coordinates of the newly calculated point
            points3dTriangulation.push_back(Vector3f(x.x(), x.y(), x.z()));

        }
        return points3dTriangulation;

    } else{
        std::cout << "PHASE 4.1: No New Points were present for Triangulation" <<std::endl;
        return points3dTriangulation;
    }
}