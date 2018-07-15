#include <iostream>
#include <fstream>
#include <ctime>
#include <set>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>

#include "Eigen.h"
#include "PointCloud.h"
#include "Optimization.h"
#include <pangolin/pangolin.h>
#include <cmath>

using namespace std;
using namespace cv;

#define FRAMES_GROUND_TRUTH 50
#define SKIP_FRAMES 5
#define POINT_CONSECUTIVE_FRAMES 5

std::string DATA_PATH = PROJECT_DIR + std::string("/data/rgbd_dataset_freiburg3_long_office_household/");
//std::string DATA_PATH = std::string("D:/rgbd_dataset_freiburg2_xyz/rgbd_dataset_freiburg2_xyz/");

// Prototype functions
void triangulationWithLastFrame(std::vector<PointCloud>& frames,std::vector<Vector3f>& global_3D_points,int last_frame_ind);
int consecutiveFrames(std::vector<PointCloud>& , int , int , int );
void generateOffFile(std::string filename, std::vector<Vector3f> points3d, std::vector<cv::Vec3b> colorValues, std::vector<PointCloud> pointClouds);
std::vector<Vector3f> performTriangulation(std::vector<Vector2f> points2dFrame1, std::vector<Vector2f> points2dFrame2, Matrix4f extrinsicsFrame1, Matrix4f extrinsicsFrame2, Matrix3f cameraIntrinsics);
void get_data(std::string, std::vector<cv::Mat>& depth_images, std::vector<cv::Mat>& rgb_images, std::vector<Matrix4f>& transformationMatrices);
Matrix4f getExtrinsicsFromQuaternion(std::vector<string>);
void split(const std::string &s, char delim, std::vector<std::string> &elems);
void visualizeResults(std::vector<std::vector<Vector3f>> visualizationPoints, std::vector<PointCloud> pointClouds);
std::vector<Vector3f> performTriangulation2(std::vector<Vector2f> points2dFrame1, std::vector<Vector2f> points2dFrame2, Matrix4f extFrame1, Matrix4f extFrame2, Matrix3f camIntr);


int main(int argc, char *argv[]){

	int i, j;

    std::cout << "PHASE 0: Get needed data" << std::endl;    

    std::vector<cv::Mat> depthImages, rgbImages;
    std::vector<Matrix4f> transformationMatrices;

    get_data("final_mapping.txt", depthImages, rgbImages, transformationMatrices);

    Matrix3f intrinsicMatrix;
    intrinsicMatrix <<  525.0f, 0.0f, 319.5f,
                        0.0f, 525.0f, 239.5f,
                        0.0f, 0.0f, 1.0f;

    std::cout << "PHASE 1: Finding, matching, discarding outlier keypoints" << std::endl;

    // Detect keypoints
    std::vector<std::vector<cv::KeyPoint>> keypointsAllImgs;
    cv::Ptr<cv::ORB> detectorORB = cv::ORB::create(1000);

    // Descriptors for keypoints
    std::vector<cv::Mat> descriptorsAllImgs;
    cv::Ptr<cv::BRISK> detectorBRISK = cv::BRISK::create();

    // Matching descriptors, we use NORM_HAMMING because the descriptors are BINARY and so it is faster with this norm
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    int nearest_neighbors = 2;
    const float ratio = 0.50;    //parameter to be tuned

    std::vector<std::vector<std::vector<cv::DMatch>>> allMatches;

    // Detect keypoints and calculate descriptors
    for (i=0; i<rgbImages.size(); i++){
        std::vector<cv::KeyPoint> keypointsImg;
        cv::Mat descriptorsImg;

        detectorORB->detect(rgbImages[i], keypointsImg);
        detectorBRISK->compute(rgbImages[i], keypointsImg, descriptorsImg);

        keypointsAllImgs.push_back(keypointsImg);
        descriptorsAllImgs.push_back(descriptorsImg);

        if (i > 0){
            // Match with kNN algorithm between pair of frames
            std::vector<std::vector<cv::DMatch>> matches;
            matcher.knnMatch(descriptorsAllImgs[i-1], descriptorsAllImgs[i], matches, nearest_neighbors);
            allMatches.push_back(matches);
        }
    }

    // Array of frames with corresponding information
    std::vector<PointCloud> pointClouds;

    // Initialize all the frames
    for(i=0; i < rgbImages.size(); i++){

    	Matrix4f extrinsicMatrix;

    	// We take ground truth poses only from the initial FRAMES_GROUND_TRUTH frames
		if (i < FRAMES_GROUND_TRUTH)
			extrinsicMatrix = transformationMatrices[i];
		else
			extrinsicMatrix = Matrix4f::Identity();

		// Create a new point cloud of the current frame
		PointCloud pointCloud = PointCloud(intrinsicMatrix, extrinsicMatrix);
		pointClouds.push_back(pointCloud);
    }

	// Remove outliers
	for (i = 0; i<(FRAMES_GROUND_TRUTH - 2); i++) {

		// Matches between first and third frame to check if keypoint is not noise
		std::vector<std::vector<cv::DMatch>> matchesFirstThirdFrame;
		matcher.knnMatch(descriptorsAllImgs[i], descriptorsAllImgs[i + 2], matchesFirstThirdFrame, nearest_neighbors);

		// Vector of index match between current and nextframe
		std::map<int, int> good_matches, good_matches_prev;

		for (j = 0; j<allMatches[i].size(); j++) {

			// Discard possible outliers, between first and second frame, if the 2 Nearest Neighbors are too close each other
			if (allMatches[i][j][0].distance < ratio * allMatches[i][j][1].distance) {

				// Find the same keypoint both in (frame+1) and (frame+2), so that I know it is not noise
				// Check if the correspondence of first-second and second-third, is equivalent to the first-third one
				if (allMatches[i + 1][allMatches[i][j][0].trainIdx][0].trainIdx == matchesFirstThirdFrame[j][0].trainIdx) {

					// Good single 2d keypoint
					cv::Point2f point2d = keypointsAllImgs[i][allMatches[i][j][0].queryIdx].pt;
					cv::Point2f point2d_next_frame = keypointsAllImgs[i + 1][allMatches[i][j][0].trainIdx].pt;

					float depth1 = depthImages[i].at<uint16_t>(point2d);
					float depth2 = depthImages[i + 1].at<uint16_t>(point2d_next_frame);
					// Only points where depth is VALID
					if (depth1 > 400 && depth2 > 400 &&
						depth1 < 13000 && depth2 < 13000 && (depth1 / depth2 >= 0.9 && depth1 / depth2 <= 1.15)) {
						good_matches[allMatches[i][j][0].queryIdx] = allMatches[i][j][0].trainIdx;
						good_matches_prev[allMatches[i][j][0].trainIdx] = allMatches[i][j][0].queryIdx;
					}
				}
			}
		}
		// Set good keypoints of frame i
		pointClouds[i].setPoints2d(keypointsAllImgs[i]);

		// Set good matches between frame i and i+1
		pointClouds[i].setIndexMatchesFrames(good_matches);
		pointClouds[i + 1].setIndexPrevMatchesFrames(good_matches_prev);
	}
	
    // Visualize matches after removing outliers
//    for(i=0;i<40;i++){
//	    std::map<int, int> tmp_map_index_frame_zero_one = pointClouds[i].getIndexMatchesFrames();
//	    std::vector<cv::DMatch> good_matches_img;
//
//	    for (std::map<int,int>::iterator it = tmp_map_index_frame_zero_one.begin(); it != tmp_map_index_frame_zero_one.end(); ++it)
//	    	good_matches_img.push_back(cv::DMatch(it->first, it->second, 0));
//
//	    cv::Mat imgMatches;
//	    cv::drawMatches(rgbImages[i], keypointsAllImgs[i], rgbImages[i+1], keypointsAllImgs[i+1], good_matches_img, imgMatches);
//
//	    cv::imshow("Good Matches"+std::to_string(i), imgMatches);
//	    cv::waitKey(0);
//	    cv::destroyWindow("Good Matches" + std::to_string(i));
//	}
//	exit(0);


	//****************DEBUG*************************: BACKPROJECTION
	/*for (int n = 0; n < 1; n++) {
		std::vector<Vector3f> global_3D_points;
		std::vector<cv::Vec3b> global_color_points;

		for (i = 0; i < 480; i++) {
			for (j = 0; j < 640; j++) {
				cv::Point2f point(j, i);
				Vector3f point_3D;

				if (depthImages[n].at<uint16_t>(point) > 0) {
					cv::Vec3b colors = rgbImages[n].at<cv::Vec3b>(i, j);

					float depth = (depthImages[n].at<uint16_t>(point) * 1.0f) / 5000.0f;
					point_3D = pointClouds[n].point2dToPoint3d(Vector2f(j, i), depth);
					global_3D_points.push_back(point_3D);
					global_color_points.push_back(colors);
				}
			}
		}

		generateOffFile("/offFiles/gt_new.off", global_3D_points, global_color_points, pointClouds); 
	}*/

    // 3D coordinates of keypoints without duplicates
    std::vector<Vector3f> global_3D_points;
    std::vector<cv::Vec3b> global_color_points;

	// Compute the first frame's 3d points
	for (i = 0; i<FRAMES_GROUND_TRUTH - 2; i++) {

		std::map<int, int> matchesFrames = pointClouds[i].getIndexMatchesFrames();
		std::map<int, int> keypointsTo3DGlobal = pointClouds[i].getGlobal3Dindices();

		for (auto& entry : matchesFrames) {
			int currentInd = entry.first;
			int nextInd = entry.second;

			// Search for points present in at least POINT_CONSECUTIVE_FRAMES frames
			int consFrames = consecutiveFrames(pointClouds, i + 1, nextInd, 1);

			if (consFrames >= POINT_CONSECUTIVE_FRAMES) {

				// 3D POINT NOT IN GLOBAL ARRAY YET
				if (keypointsTo3DGlobal.find(currentInd) == keypointsTo3DGlobal.end()) {

					cv::Point2f keypoint_2D = keypointsAllImgs[i][currentInd].pt;
					Vector3f point_3D;

					// At the beginning, we use depth values from GT
					if (i == 0) {
						float depth = (depthImages[i].at<uint16_t>(keypoint_2D) * 1.0f) / 5000.0f;
						point_3D = pointClouds[i].point2dToPoint3d(Vector2f(keypoint_2D.x, keypoint_2D.y), depth);
					}
					else {
						// We will do triangulation later after optimization
						point_3D = Vector3f(MINF, MINF, MINF);
					}

					// Store inside point cloud the indices of the corresponding 3D points
					int indexThisFrame = currentInd;
					for (j = i; j < (consFrames + i); j++) {

						std::map<int, int> tmp_matches = pointClouds[j].getIndexMatchesFrames();

						pointClouds[j].appendGlobal3Dindices(indexThisFrame, global_3D_points.size());

						indexThisFrame = tmp_matches[indexThisFrame];
					}

					// Append to global 3D vectors					
					global_color_points.push_back(rgbImages[i].at<cv::Vec3b>(keypoint_2D));
					global_3D_points.push_back(point_3D);
				}
			}
		}
	}
	// Optimise on poses and update on points
	Optimization optimizer;
	optimizer.setNbOfIterations(10);
	for (i = 0; i < FRAMES_GROUND_TRUTH - 2; i++) {
		Matrix4f finalPose = optimizer.estimatePose(pointClouds[i], global_3D_points);
		pointClouds[i + 1].setCameraExtrinsics(finalPose);
		std::map<int, int> match_2D_to_3D = pointClouds[i+1].getGlobal3Dindices();
		for (std::map<int, int>::iterator it = match_2D_to_3D.begin(); it != match_2D_to_3D.end(); ++it) {
			if (global_3D_points[it->second][0] == MINF) {
				cv::Point2f keypoint_2D = keypointsAllImgs[i + 1][it->first].pt;
				float depth = (depthImages[i + 1].at<uint16_t>(keypoint_2D) * 1.0f) / 5000.0f;
				Vector3f point_3D = pointClouds[i + 1].point2dToPoint3d(Vector2f(keypoint_2D.x, keypoint_2D.y), depth);
				global_3D_points[it->second] = point_3D;
			}
		}
	}

    std::cout << "PHASE 2: Optimization" << std::endl;
	optimizer.setNbOfIterations(30);
	int frames_to_consider = 8;
	generateOffFile("/offFiles/result_b4_opt.off", global_3D_points, global_color_points, pointClouds);
    //for(i=0; i < (pointClouds.size() - 20 - frames_to_consider); i++){

//    // Code related to visualization start
    std::vector<std::vector<Vector3f>> visualizationPoints;
    // Code related to visualization end

    for(i=0; i < 40; i++){
    	optimizer.estimatePoseWithPoint(pointClouds, global_3D_points, i, frames_to_consider);
        //triangulationWithLastFrame(pointClouds, global_3D_points, frames_to_consider - 1 + i);
    	std::cout << std::endl << std::endl << std::endl;

    	// Code related to visualization start
        visualizationPoints.push_back(global_3D_points);
        // Code related to visualization end
    }

    // Code related to visualization start
    visualizeResults(visualizationPoints, pointClouds);
//    // Code related to visualization end

	//TEST TRIANGULATION
//	std::map<int, int> tmp1 = pointClouds[0].getIndexMatchesFrames();
//
//    std::vector<cv::KeyPoint> curr_2d = pointClouds[0].getPoints2d();
//    std::vector<cv::KeyPoint> next_2d = pointClouds[1].getPoints2d();
//    std::vector<Vector3f> testPoints;
//    std::vector<Vector2f> points2dFrame1;
//    std::vector<Vector2f> points2dFrame2;
//
//    for (std::map<int,int>::iterator it = tmp1.begin(); it != tmp1.end(); ++it){
//
//        cv::KeyPoint point2D_prev = curr_2d[it->first];
//        points2dFrame1.push_back(Vector2f(point2D_prev.pt.x, point2D_prev.pt.y));
//
//        cv::KeyPoint point2D_curr = next_2d[it->second];
//        points2dFrame2.push_back(Vector2f(point2D_curr.pt.x, point2D_curr.pt.y));
//
//        //float dd = (depthImages[1].at<uint16_t>(point2D_curr.pt) * 1.0f) / 5000.0f;
//        //Vector3f test_result = pointClouds[1].point2dToPoint3d(Vector2f(point2D_curr.pt.x, point2D_curr.pt.y), dd);
//        float dd = (depthImages[0].at<uint16_t>(point2D_prev.pt) * 1.0f) / 5000.0f;
//        Vector3f test_result = pointClouds[0].point2dToPoint3d(Vector2f(point2D_prev.pt.x, point2D_prev.pt.y), dd);
//        testPoints.push_back(test_result);
//    }
//    std::vector<Vector3f> tr_result = performTriangulation2(points2dFrame1, points2dFrame2, pointClouds[0].getCameraExtrinsics(), pointClouds[1].getCameraExtrinsics(), pointClouds[0].getCameraIntrinsics());
//    std::vector<Vector3f> tr_result2 = performTriangulation(points2dFrame1, points2dFrame2, pointClouds[0].getCameraExtrinsics(), pointClouds[1].getCameraExtrinsics(), pointClouds[0].getCameraIntrinsics());
//    for(int a=0; a<testPoints.size(); a++){
//        std::cout << testPoints[a].x() << "    " << testPoints[a].y() << "    " << testPoints[a].z() << std::endl;
//        std::cout << tr_result2[a].x() << "    " << tr_result2[a].y() << "    " << tr_result2[a].z() << std::endl;
//        std::cout << tr_result[a].x() << "    " << tr_result[a].y() << "    " << tr_result[a].z() << std::endl;
//        std::cout << std::endl;
//    }

    generateOffFile("/offFiles/result_after_opt.off", global_3D_points, global_color_points, pointClouds);

    return 0;
}


// Method to generate off file for 3d points
void generateOffFile(std::string filename, std::vector<Vector3f> points3d, std::vector<cv::Vec3b> colorValues, std::vector<PointCloud> pointClouds) {
    std::ofstream outFile(PROJECT_DIR + filename);
    if (!outFile.is_open()) return;
    if (points3d.size() == 0) return;
    
    // write header
    outFile << "COFF" << std::endl;
    outFile << points3d.size() << " 0 0" << std::endl;
    
    // Save camera position
    for (int i = 0; i < 48; i++) {
        Matrix4f cameraExtrinsics = pointClouds[i].getCameraExtrinsics();
        Matrix3f rotation = cameraExtrinsics.block(0, 0, 3, 3);
        Vector3f translation = cameraExtrinsics.block(0, 3, 3, 1);
        Vector3f cameraPosition = -rotation.transpose()*translation;
        outFile << cameraPosition.x() << " " << cameraPosition.y() << " " << cameraPosition.z() << " 255 0 0" << std::endl;
    }

    // Save vertices
    for (int i = 0; i < points3d.size(); i++) {
        if (points3d[i].x() == MINF)
            outFile << "0 0 0 0 0 0" << std::endl;
    // OpenCV stores as BGR
        else
        	outFile << points3d[i].x() << " " << points3d[i].y() << " " << points3d[i].z() << " " << 
            static_cast<unsigned>(colorValues[i][2]) << " " << static_cast<unsigned>(colorValues[i][1]) << " " << static_cast<unsigned>(colorValues[i][0]) <<  std::endl;
    }
    outFile.close();
}


void get_data(std::string file_path, std::vector<cv::Mat>& depth_images, std::vector<cv::Mat>& rgb_images, std::vector<Matrix4f>& transformationMatrices){

    std::ifstream inFile;
    std::string fileName;
    
    inFile.open(DATA_PATH + file_path);

    if (!inFile) {
        std::cerr << "Unable to open file.\n" << std::endl;
        exit(1);
    }

    int i = 0;
    while (std::getline(inFile, fileName)) {
        i++;
        if (i % SKIP_FRAMES != 0)
        	continue;

        std::vector<std::string> current_line;
        split(fileName, ' ', current_line);

//        cv::Mat depthImg = cv::imread(DATA_PATH + current_line[0], CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
//        cv::Mat rgbImg = cv::imread(DATA_PATH + current_line[1], CV_LOAD_IMAGE_ANYCOLOR | CV_LOAD_IMAGE_ANYDEPTH);
        cv::Mat depthImg = cv::imread(DATA_PATH + current_line[0],  IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);
        cv::Mat rgbImg = cv::imread(DATA_PATH + current_line[1],  IMREAD_ANYCOLOR | IMREAD_ANYDEPTH);

        if (!depthImg.data || !rgbImg.data) {
            std::cout << "Image not found.\n" << std::endl;
            exit(1);
        }

        depth_images.push_back(depthImg);
        rgb_images.push_back(rgbImg);

        // Save poses
        Matrix4f transformationMatrix = getExtrinsicsFromQuaternion(current_line);
        transformationMatrices.push_back(transformationMatrix);     

        // TAKE FEW IMAGES DURING DEBUG
        if (i > 1000)
            break;
    }

    inFile.close();
}


int consecutiveFrames(std::vector<PointCloud>& PCs, int currPointCloudIndex, int currKeypointIndex, int currRecursion){
	
	std::map<int, int> matchesFrames = PCs[currPointCloudIndex].getIndexMatchesFrames();

	if (matchesFrames.find(currKeypointIndex) ==  matchesFrames.end())
		return currRecursion + 1;
	
	else
		return consecutiveFrames(PCs, currPointCloudIndex + 1, matchesFrames[currKeypointIndex], currRecursion + 1);
}


Matrix4f getExtrinsicsFromQuaternion(std::vector<string> poses){
    int i, j;
    Matrix4f extrinsics;
    extrinsics.setIdentity();

    Eigen::Quaterniond q(Eigen::Vector4d(std::stod(poses[5]), std::stod(poses[6]), std::stod(poses[7]), std::stod(poses[8])));
    Eigen::Matrix<double, 3, 3> rotation = q.normalized().toRotationMatrix();

    for(i = 0; i < 3; i++){
        for(j = 0; j < 3; j++)
            extrinsics(i,j) = rotation(i,j);
    }
    extrinsics(0,3) = std::stod(poses[2]);
    extrinsics(1,3) = std::stod(poses[3]);
    extrinsics(2,3) = std::stod(poses[4]);

    return extrinsics;
}


std::vector<Vector3f> performTriangulation(std::vector<Vector2f> points2dFrame1, std::vector<Vector2f> points2dFrame2, Matrix4f extFrame1, Matrix4f extFrame2, Matrix3f camIntr){
    // Run the triangulation here
    // Find the matching points for which we do not have the 3d points
    // Triangulate them and add them to the point cloud

    std::cout << "PHASE 3.1: Running the triangulation" <<std::endl;

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

            Vector2f pointframe1 = points2dFrame1[g];
            Vector2f pointframe2 = points2dFrame2[g];

            // Calculate the values to be inserted inside A and b for Ax=b
            A(0,0) = extFrame1(2,0) * pointframe1.x() - camIntr(0,0) * extFrame1(0,0) - camIntr(0,2) * extFrame1(2,0);
            A(0,1) = extFrame1(2,1) * pointframe1.x() - camIntr(0,0) * extFrame1(0,1) - camIntr(0,2) * extFrame1(2,1);
            A(0,2) = extFrame1(2,2) * pointframe1.x() - camIntr(0,0) * extFrame1(0,2) - camIntr(0,2) * extFrame1(2,2);

            A(1,0) = extFrame1(2,0) * pointframe1.y() - camIntr(1,1) * extFrame1(1,0) - camIntr(1,2) * extFrame1(2,0);
            A(1,1) = extFrame1(2,1) * pointframe1.y() - camIntr(1,1) * extFrame1(1,1) - camIntr(1,2) * extFrame1(2,1);
            A(1,2) = extFrame1(2,2) * pointframe1.y() - camIntr(1,1) * extFrame1(1,2) - camIntr(1,2) * extFrame1(2,2);


            A(2,0) = extFrame2(2,0) * pointframe2.x() - camIntr(0,0) * extFrame2(0,0) - camIntr(0,2) * extFrame2(2,0);
            A(2,1) = extFrame2(2,1) * pointframe2.x() - camIntr(0,0) * extFrame2(0,1) - camIntr(0,2) * extFrame2(2,1);
            A(2,2) = extFrame2(2,2) * pointframe2.x() - camIntr(0,0) * extFrame2(0,2) - camIntr(0,2) * extFrame2(2,2);

            A(3,0) = extFrame2(2,0) * pointframe2.y() - camIntr(1,1) * extFrame2(1,0) - camIntr(1,2) * extFrame2(2,0);
            A(3,1) = extFrame2(2,1) * pointframe2.y() - camIntr(1,1) * extFrame2(1,1) - camIntr(1,2) * extFrame2(2,1);
            A(3,2) = extFrame2(2,2) * pointframe2.y() - camIntr(1,1) * extFrame2(1,2) - camIntr(1,2) * extFrame2(2,2);


            b[0] = camIntr(0,0) * extFrame1(0,3) + camIntr(0,2) * extFrame1(2,3) - pointframe1.x() * extFrame1(2,3);
            b[1] = camIntr(1,1) * extFrame1(1,3) + camIntr(1,2) * extFrame1(2,3) - pointframe1.y() * extFrame1(2,3);

            b[2] = camIntr(0,0) * extFrame2(0,3) + camIntr(0,2) * extFrame2(2,3) - pointframe2.x() * extFrame2(2,3);
            b[3] = camIntr(1,1) * extFrame2(1,3) + camIntr(1,2) * extFrame2(2,3) - pointframe2.y() * extFrame2(2,3);

            // Solve the system of equations
//            x = A.colPivHouseholderQr().solve(b);
            x = A.bdcSvd(ComputeThinU | ComputeThinV).solve(b);

            //std::cout << (A *  gt_p) << std::endl;
            //std::cout << b << std::endl;
            
            // Push back the coordinates of the newly calculated point
            points3dTriangulation.push_back(Vector3f(x.x(), x.y(), x.z()));

        }
        return points3dTriangulation;

    } else{
        std::cout << "PHASE 3.1: No New Points were present for Triangulation" <<std::endl;
        return points3dTriangulation;
    }
}


void triangulationWithLastFrame(std::vector<PointCloud>& pointClouds, std::vector<Vector3f>& global_3D_points, int ind_last_frame) {
    
	// Extract 2D points
    std::vector<cv::KeyPoint> secondLastFramePoints2d = pointClouds[ind_last_frame - 1].getPoints2d();
    std::vector<cv::KeyPoint> lastFramePoints2d = pointClouds[ind_last_frame].getPoints2d();

    // key => index 2D curr frame | value => index 2d prev frame
    std::map<int, int> prevIndexMatch = pointClouds[ind_last_frame].getIndexPrevMatchesFrames();

    // Mapping 2D and 3D global indices
    std::map<int,int> global3DIndices = pointClouds[ind_last_frame].getGlobal3Dindices();

    // Containers that store data for triangulation
    std::vector<int> minfIndices;
    std::vector<Vector2f> secondLastFramePoints;
    std::vector<Vector2f> lastFramePoints;

    for(auto& entry: global3DIndices) {
    	int pixelInd = entry.first;
    	int globalInd = entry.second;

    	// Accumulate points for triangulation
    	if(global_3D_points[globalInd][0] == MINF){
    		Vector2f point_A = Vector2f(secondLastFramePoints2d[prevIndexMatch[pixelInd]].pt.x, 
    			secondLastFramePoints2d[prevIndexMatch[pixelInd]].pt.y);
    		secondLastFramePoints.push_back(point_A);

    		Vector2f point_B = Vector2f(lastFramePoints2d[pixelInd].pt.x, lastFramePoints2d[pixelInd].pt.y);
    		lastFramePoints.push_back(point_B);

    		minfIndices.push_back(globalInd);
    	}
    }

    // Triangulation
    std::vector<Vector3f> restoredPoints = performTriangulation(secondLastFramePoints, lastFramePoints,
        pointClouds[ind_last_frame - 1].getCameraExtrinsics(), pointClouds[ind_last_frame].getCameraExtrinsics(),
        pointClouds[ind_last_frame].getCameraIntrinsics());

    // store the result into the global 3D points vector
    for(int t = 0; t < restoredPoints.size(); t++)   
        global_3D_points[minfIndices[t]] = restoredPoints[t];
}


void split(const std::string &s, char delim, std::vector<std::string> &elems) {
    std::stringstream ss;
    ss.str(s);
    std::string item;
    while (std::getline(ss, item, delim)) {
        elems.push_back(item);
    }
}

void visualizeResults(std::vector<std::vector<Vector3f>> visualizationPoints, std::vector<PointCloud> pointClouds){

    // Create OpenGL window in single line
    pangolin::CreateWindowAndBind("Main",1080,780);

    // 3D Mouse handler requires depth testing to be enabled
    glEnable(GL_DEPTH_TEST);

    // Issue specific OpenGl we might need
    glEnable (GL_BLEND);
    glBlendFunc (GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA);

    // Define Camera Render Object (for view / scene browsing)
    pangolin::OpenGlRenderState s_cam(
            pangolin::ProjectionMatrix(640,480,420,420,320,240,0.1,1000),
            pangolin::ModelViewLookAt(0,2,-2, 0,0,0, 0,-1,0)
    );

    pangolin::View& d_cam = pangolin::Display("World")
            .SetBounds(0.0f, 1.0f, pangolin::Attach::Pix(120), 1.0, -640.0f/480.0f)
            .SetHandler(new pangolin::Handler3D(s_cam));

    const float tinc = 0.01f;
    float t = 0;
    std::vector<Vector3f> vectorOfPoints;

    // Default hooks for exiting (Esc) and fullscreen (tab).
    int k = 0;
    while( !pangolin::ShouldQuit() )
    {
        if(k < visualizationPoints.size()){
            k = k + 1;
        } else {
            k = visualizationPoints.size();
        }
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        d_cam.Activate(s_cam);

        // Get the points
        glPointSize(2);
        glBegin(GL_POINTS);
        glColor3f(1.0,1.0,1.0);
        for (int i = 0; i < k; i++){
            vectorOfPoints = visualizationPoints[i];
            for(int j = 0; j < vectorOfPoints.size(); j++){
                glVertex3f(vectorOfPoints[j].x(), vectorOfPoints[j].y(), vectorOfPoints[j].z());
            }
        }
        glColor3f(1.0,0.0,0.0);
        for (int i = 0; i < k; i++){
            glVertex3f(pointClouds[i].cameraExtrinsics(0, 3), pointClouds[i].cameraExtrinsics(1, 3), pointClouds[i].cameraExtrinsics(2, 3));
        }
        glEnd();
        sleep(0.01);
        t += tinc;

        // Render graph, Swap frames and Process Events
        pangolin::FinishFrame();
    }
}

std::vector<Vector3f> performTriangulation2(std::vector<Vector2f> points2dFrame1, std::vector<Vector2f> points2dFrame2, Matrix4f extFrame1, Matrix4f extFrame2, Matrix3f camIntr){

    std::vector<cv::Point2f> pointsFrame1;
    std::vector<cv::Point2f> pointsFrame2;

    std::cout << "Sizes: " << points2dFrame1.size() << "     " << points2dFrame2.size() << std::endl;

    // Convert the vectors to vectors of point2f
    for (int x = 0; x < points2dFrame2.size(); x++){
        pointsFrame1.push_back(cv::Point2f(points2dFrame1[x].x(), points2dFrame1[x].y()));
        pointsFrame2.push_back(cv::Point2f(points2dFrame2[x].x(), points2dFrame2[x].y()));
    }

    // Estimate the fundamental matrix
    Mat fundamental_matrix = findFundamentalMat(pointsFrame1, pointsFrame2, FM_RANSAC, 3, 0.99);

    std::cout << "Fundamental Matrix: " << std::endl;
    std::cout << fundamental_matrix << std::endl;
    std::cout << fundamental_matrix.rows << "    " << fundamental_matrix.cols << std::endl;

    Matrix3f fundamentalMatrix;
    fundamentalMatrix.setIdentity();

    for(int i = 0; i < fundamental_matrix.rows; i++)
    {
        const double* Mi = fundamental_matrix.ptr<double>(i);
        for(int j = 0; j < fundamental_matrix.cols; j++){
            fundamentalMatrix(i,j) = (float)Mi[j];
//            std::cout << Mi[j] << "     " << (float)Mi[j] << std::endl;
        }
    }

//    cv::cv2eigen(&fundamental_matrix, &fundamentalMatrix);
//    fundamentalMatrix <<    fundamental_matrix.at<float>(0,0), fundamental_matrix.at<float>(0,1), fundamental_matrix.at<float>(0,2),
//                            fundamental_matrix.at<float>(1,0), fundamental_matrix.at<float>(1,1), fundamental_matrix.at<float>(1,2),
//                            fundamental_matrix.at<float>(2,0), fundamental_matrix.at<float>(2,1), fundamental_matrix.at<float>(2,2);
//    Eigen::Map<Eigen::Matrix<float,3,3>> fundamentalMatrix(fundamental_matrix.ptr<float>(), fundamental_matrix.rows, fundamental_matrix.cols);
//    float *A=(float *)fundamental_matrix.data;
//    Eigen::Map<Eigen::Matrix<float,Eigen::Dynamic,Eigen::Dynamic,Eigen::RowMajor> > fundamentalMatrix (A, fundamental_matrix.rows, fundamental_matrix.cols);
    Matrix3f fundamentalMatrixTranspose = fundamentalMatrix.transpose();

    std::cout << "Fundamental Matrix Eigen: " << std::endl;
    std::cout << fundamentalMatrix << std::endl;
    std::cout << "Fundamental Matrix Eigen Transpose: " << std::endl;
    std::cout << fundamentalMatrixTranspose << std::endl;

//    exit(0);

    // Defining the variables for adjusting the points
    Vector4f adjustedPoints;
    Vector4f actualPoints;
    Vector3f point2;
    Vector3f point1;
    float denominator, jacobian1, jacobian2, jacobian3, jacobian4;
    VectorXf numerator;
    Vector4f jacobian;
    std::vector<Vector2f> adjustedPointsFrame1;
    std::vector<Vector2f> adjustedPointsFrame2;

    for(int i = 0; i < points2dFrame2.size(); i++){
        std::cout << points2dFrame1[i].x() << "  " << points2dFrame1[i].y() << "  " << points2dFrame2[i].x() << "  " << points2dFrame2[i].y() << std::endl;
        actualPoints = Vector4f(points2dFrame1[i].x(), points2dFrame1[i].y(), points2dFrame2[i].x(), points2dFrame2[i].y());
        point2 = Vector3f(points2dFrame2[i].x(), points2dFrame2[i].y(), 1);
        point1 = Vector3f(points2dFrame1[i].x(), points2dFrame1[i].y(), 1);
        numerator = point2.transpose() * fundamentalMatrix * point1;
        jacobian1 = fundamentalMatrixTranspose(0,0) * points2dFrame2[i].x() + fundamentalMatrixTranspose(0,1) * points2dFrame2[i].y() + fundamentalMatrixTranspose(0,2);
        jacobian2 = fundamentalMatrixTranspose(1,0) * points2dFrame2[i].x() + fundamentalMatrixTranspose(1,1) * points2dFrame2[i].y() + fundamentalMatrixTranspose(1,2);
        jacobian3 = fundamentalMatrix(0,0) * points2dFrame1[i].x() + fundamentalMatrix(0,1) * points2dFrame1[i].y() + fundamentalMatrix(0,2);
        jacobian4 = fundamentalMatrix(1,0) * points2dFrame1[i].x() + fundamentalMatrix(1,1) * points2dFrame1[i].y() + fundamentalMatrix(1,2);
        jacobian = Vector4f(jacobian1, jacobian2, jacobian3, jacobian4);
        denominator = pow(jacobian1,2) + pow(jacobian2,2) + pow(jacobian3,2) + pow(jacobian4,2);
        adjustedPoints = actualPoints - ((numerator.x()/denominator) * jacobian);
        adjustedPointsFrame1.push_back(Vector2f(adjustedPoints.x(), adjustedPoints.y()));
        adjustedPointsFrame2.push_back(Vector2f(adjustedPoints.z(), adjustedPoints.w()));
        std::cout << adjustedPoints.x() << "  " << adjustedPoints.y() << "  " << adjustedPoints.z() << "  " << adjustedPoints.w() << std::endl;
        std::cout << std::endl;
    }

    // Now perform the triangulation on adjusted points
    // This vector stores the 3D point coordinates that we will calculate during the triangulation
    std::vector<Vector3f> points3dTriangulation;

    MatrixXf projMatFrame1 = camIntr * extFrame1.block(0,0,3,4);
    MatrixXf projMatFrame2 = camIntr * extFrame2.block(0,0,3,4);

    // Initializing Ax = b
    Eigen::Matrix<double, 6, 4> A;

    for(int i = 0; i < adjustedPointsFrame2.size(); i++){

        for (size_t k=0; k<4; ++k) {
            // first set of points
            A(0, k) = adjustedPointsFrame1[i].x() * projMatFrame1(2, k) - projMatFrame1(0, k);
            A(1, k) = adjustedPointsFrame1[i].y() * projMatFrame1(2, k) - projMatFrame1(1, k);
            A(2, k) = adjustedPointsFrame1[i].x() * projMatFrame1(1, k) - adjustedPointsFrame1[i].y() * projMatFrame1(0, k);
            // second set of points
            A(3, k) = adjustedPointsFrame2[i].x() * projMatFrame2(2, k) - projMatFrame2(0, k);
            A(4, k) = adjustedPointsFrame2[i].y() * projMatFrame2(2, k) - projMatFrame2(1, k);
            A(5, k) = adjustedPointsFrame2[i].x() * projMatFrame2(1, k) - adjustedPointsFrame2[i].y() * projMatFrame2(0, k);
        }

        Eigen::JacobiSVD<Eigen::MatrixXd> svd(A, Eigen::ComputeThinV);
        Eigen::Matrix<double, 4, 4> V = svd.matrixV();

        // Normalize point
        V(0, 3) /= V(3, 3);
        V(1, 3) /= V(3, 3);
        V(2, 3) /= V(3, 3);

        points3dTriangulation.push_back(Vector3f(V(0, 3), V(1, 3), V(2, 3)));

    }

    return points3dTriangulation;

}