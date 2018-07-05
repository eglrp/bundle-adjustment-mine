//
// Created by ritvik on 14.06.18.
//

#ifndef BUNDLE_ADJUSTEMENT_POINTCLOUD_H
#define BUNDLE_ADJUSTEMENT_POINTCLOUD_H

#endif //BUNDLE_ADJUSTEMENT_POINTCLOUD_H

#include "Eigen.h"
#include <iostream>
#include <fstream>

class PointCloud {
public:

    std::vector<Vector3f> points3d;
    std::vector<Vector2f> points2d;
    std::vector<Vector2f> points2dNextFrame;
	std::vector<Vector2f> points2dThirdFrame;
	std::vector<int> positionInGlobal3d;
    std::vector<cv::Vec3b> colorValues;
    Matrix4f cameraExtrinsics;
    Matrix3f cameraIntrinsics;
    std::vector<float> depthMap;
	int valid3DPointSize = 0;

    PointCloud(){

    }

    PointCloud(const Matrix3f& intrinsicMatrix, const Matrix4f& extrinsicMatrix){

        cameraIntrinsics = intrinsicMatrix;
        cameraExtrinsics = extrinsicMatrix;

    }

    //Getters and setters
    std::vector<cv::Vec3b> getColorValues(){
        return colorValues;
    }

    void setColorValues(const std::vector<cv::Vec3b> colors){
        colorValues = colors;
    }

    Matrix4f getCameraExtrinsics() const{
        return cameraExtrinsics;
    }

    void setCameraExtrinsics(const Matrix4f& extrinsicMatrix){
        cameraExtrinsics = extrinsicMatrix;
    }

    Matrix3f getCameraIntrinsics() const{
        return cameraIntrinsics;
    }

    void setCameraIntrinsics(const Matrix3f& intrinsicMatrix){
        cameraIntrinsics = intrinsicMatrix;
    }

    std::vector<Vector3f> getPoints3d() const{
        return points3d;
    }

    void setPoints3d(const std::vector<Vector3f> points3dim){
        points3d = points3dim;
    }

    std::vector<Vector2f> getPoints2d() const{
        return points2d;
    }

    void setPoints2d(const std::vector<Vector2f> points2dim){
        points2d = points2dim;
    }

    void setPoints2dNextFrame(const std::vector<Vector2f> points2dimnextframe){
        points2dNextFrame = points2dimnextframe;
    }

    std::vector<Vector2f> getPoints2dThirdFrame() const{
        return points2dThirdFrame;
    }

	void setPoints2dThirdFrame(const std::vector<Vector2f> points2d) {
		points2dThirdFrame = points2d;
	}

	std::vector<Vector2f> getPoints2dNextFrame() const {
		return points2dNextFrame;
	}

    std::vector<float> getDepthMap(){
        return depthMap;
    }

    void setDepthMap(const std::vector<float> depthVector){
        depthMap = depthVector;
    }

	void setGlobalPosition(const std::vector<int> globalIndices) {
		positionInGlobal3d = globalIndices;
	}

	int getGlobalPosition(int localIdx) {
		return positionInGlobal3d[localIdx];
	}

    //Method to project points3d to points2d
    std::tuple<std::vector<Vector2f>, std::vector<float>> points3dToPoints2d(const Matrix3f& cameraIntrinsics, const Matrix4f& cameraExtrinsics, const std::vector<Vector3f> points3d){

        std::vector<Vector2f> points2d;
        std::vector<float> depthMap;
        Vector3f point;
        Vector2f point2d;
        Matrix3f rotation = cameraExtrinsics.block(0, 0, 3, 3);
        Vector3f translation = cameraExtrinsics.block(0, 3, 3, 1);

        for(int i = 0; i < points3d.size(); i++){
            point = rotation * points3d[i];
            point = point + translation;
            point = cameraIntrinsics * point;
            depthMap.push_back(point.z());
            point2d = Vector2f(point.x()/point.z(), point.y()/point.z());
            points2d.push_back(point2d);
        }

        return std::make_tuple(points2d, depthMap);
    }

    //Method to project points2d to points3d
    std::vector<Vector3f> points2dToPoints3d(const Matrix3f& cameraIntrinsics, const Matrix4f& cameraExtrinsics, const std::vector<Vector2f> points2d, std::vector<float> depthMap){

        std::vector<Vector3f> points3d;
        float fovX = cameraIntrinsics(0, 0);
        float fovY = cameraIntrinsics(1, 1);
        float cX = cameraIntrinsics(0, 2);
        float cY = cameraIntrinsics(1, 2);
        Matrix4f cameraExtrinsicsInv = cameraExtrinsics.inverse();
        Matrix3f rotationInv = cameraExtrinsicsInv.block(0, 0, 3, 3);
        Vector3f translationInv = cameraExtrinsicsInv.block(0, 3, 3, 1);



        for(int i = 0; i < points2d.size(); i++){
			if (depthMap[i] == MINF) {
				points3d.push_back(Vector3f(MINF, MINF, MINF));
			}
			else {
				float x = ((float) points2d[i].x() - cX) / fovX;
				float y = ((float) points2d[i].y() - cY) / fovY;
				float depth = depthMap[i];

				Vector4f backprojected = Vector4f(depth * x, depth * y, depth, 1);
	            Vector4f worldSpace = cameraExtrinsicsInv * backprojected;

	            points3d.push_back(Vector3f(worldSpace[0], worldSpace[1], worldSpace[2]));
	            valid3DPointSize++;
	        }
        }

        return points3d;

    }

    // Method to generate off file for 3d points
    void generateOffFile(std::string filename) {
        std::ofstream outFile(PROJECT_DIR + std::string("/offFiles/" + filename + ".off"));
        if (!outFile.is_open()) return;
        if (points3d.size() == 0) return;
        // write header
        outFile << "COFF" << std::endl;
        outFile << valid3DPointSize + 1 << " 0 0" << std::endl;
		// Save camera position
		Matrix3f rotation = cameraExtrinsics.block(0, 0, 3, 3);
		Vector3f translation = cameraExtrinsics.block(0, 3, 3, 1);
		Vector3f cameraPosition = -rotation.transpose()*translation;
		outFile << cameraPosition.x() << " " << cameraPosition.y() << " " << cameraPosition.z() << " 255 0 0" << std::endl;
        // Save vertices
        for (int i = 0; i < points3d.size(); i++) {
			if (points3d[i].x() == MINF)
				continue;
		// OpenCV stores as BGR
            outFile << points3d[i].x() << " " << points3d[i].y() << " " << points3d[i].z() << " " << 
            	static_cast<unsigned>(colorValues[i][2]) << " " << static_cast<unsigned>(colorValues[i][1]) << " " << static_cast<unsigned>(colorValues[i][0]) <<  std::endl;
        }
        outFile.close();
        return;
    }

	// Method to update the 3d locations, with stored pose and 2d points.
	void computeNewPoints3D() {
		if (points2d.size() > 0 && depthMap.size() > 0) {
			std::vector<Vector3f> newPoints3d;
			float fovX = cameraIntrinsics(0, 0);
			float fovY = cameraIntrinsics(1, 1);
			float cX = cameraIntrinsics(0, 2);
			float cY = cameraIntrinsics(1, 2);
			Matrix4f cameraExtrinsicsInv = cameraExtrinsics.inverse();
			Matrix3f rotationInv = cameraExtrinsicsInv.block(0, 0, 3, 3);
			Vector3f translationInv = cameraExtrinsicsInv.block(0, 3, 3, 1);
			valid3DPointSize = 0;
			for (int i = 0; i < points2d.size(); i++) {
				if (depthMap[i] == MINF) {
					newPoints3d.push_back(Vector3f(MINF, MINF, MINF));
				}
				else {
					float x = ((float)points2d[i].x() - cX) / fovX;
					float y = ((float)points2d[i].y() - cY) / fovY;
					float depth = depthMap[i];

					Vector4f backprojected = Vector4f(depth * x, depth * y, depth, 1);
					Vector4f worldSpace = cameraExtrinsicsInv * backprojected;

					newPoints3d.push_back(Vector3f(worldSpace[0], worldSpace[1], worldSpace[2]));
					valid3DPointSize++;
				}
			}
			points3d = newPoints3d;
		}
	}
};
