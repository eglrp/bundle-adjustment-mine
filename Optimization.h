#include <ceres/ceres.h>
#include <ceres/rotation.h>



template <typename T>
class PoseIncrement {
public:
	explicit PoseIncrement(T* const array) : m_array{ array } { }
	
	void setZero() {
		for (int i = 0; i < 6; ++i)
			m_array[i] = T(0);
	}

	T* getData() const {
		return m_array;
	}

	/**
	 * Applies the pose increment onto the input point and produces transformed output point.
	 * Important: The memory for both 3D points (input and output) needs to be reserved (i.e. on the stack)
	 * beforehand).
	 */
	void apply(T* inputPoint, T* outputPoint) const {
		// pose[0,1,2] is angle-axis rotation.
		// pose[3,4,5] is translation.
		const T* rotation = m_array;
		const T* translation = m_array + 3;

		T temp[3];
		ceres::AngleAxisRotatePoint(rotation, inputPoint, temp);

		outputPoint[0] = temp[0] + translation[0];
		outputPoint[1] = temp[1] + translation[1];
		outputPoint[2] = temp[2] + translation[2];
	}

	static T* pose6DOF(const Matrix4f& rotationMatrix){
		T* pose = new T[3];

		T rotMatrix[9] = { rotationMatrix(0,0), rotationMatrix(1,0), rotationMatrix(2,0), rotationMatrix(0,1), 
			rotationMatrix(1,1), rotationMatrix(2,1), rotationMatrix(0,2), rotationMatrix(1,2), rotationMatrix(2,2)};
		ceres::RotationMatrixToAngleAxis(rotMatrix, pose);

		T* pose6DOF = new T[6];
		pose6DOF[0] = pose[0];
		pose6DOF[1] = pose[1];
		pose6DOF[2] = pose[2];
		pose6DOF[3] = rotationMatrix(0,3);
		pose6DOF[4] = rotationMatrix(1,3);
		pose6DOF[5] = rotationMatrix(2,3);
		
		/*for (int i = 0; i < 6; ++i)
			m_array[i] = pose_six[i];*/

		return pose6DOF;
	}

	/**
	 * Converts the pose increment with rotation in SO3 notation and translation as 3D vector into
	 * transformation 4x4 matrix.
	 */
	static Matrix4f convertToMatrix(T* pose) {
		// pose[0,1,2] is angle-axis rotation.
		// pose[3,4,5] is translation.
		double* rotation = pose;
		double* translation = pose + 3;

		// Convert the rotation from SO3 to matrix notation (with column-major storage).
		double rotationMatrix[9];
		ceres::AngleAxisToRotationMatrix(rotation, rotationMatrix);

		// Create the 4x4 transformation matrix.
		Matrix4f matrix;
		matrix.setIdentity();
		matrix(0, 0) = float(rotationMatrix[0]);	matrix(0, 1) = float(rotationMatrix[3]);	matrix(0, 2) = float(rotationMatrix[6]);	matrix(0, 3) = float(translation[0]);
		matrix(1, 0) = float(rotationMatrix[1]);	matrix(1, 1) = float(rotationMatrix[4]);	matrix(1, 2) = float(rotationMatrix[7]);	matrix(1, 3) = float(translation[1]);
		matrix(2, 0) = float(rotationMatrix[2]);	matrix(2, 1) = float(rotationMatrix[5]);	matrix(2, 2) = float(rotationMatrix[8]);	matrix(2, 3) = float(translation[2]);
		
		return matrix;
	}

private:
	T* m_array;
};


class PointToPointConstraint {
public:
	PointToPointConstraint(const Vector3f& sourcePoint, const Vector2f& targetPoint) :
		m_sourcePoint{ sourcePoint },
		m_targetPoint{ targetPoint }
	{ }

	template <typename T>
	bool operator()(const T* const pose, T* residuals) const {
		// Important: Ceres automatically squares the cost function.

		const T* rotation = pose;
		const T* translation = pose + 3;

		// Rotation.
		T p[3];
		T point[] = {T(m_sourcePoint[0]), T(m_sourcePoint[1]), T(m_sourcePoint[2])};
		ceres::AngleAxisRotatePoint(rotation, point, p);

		// Translation.
		p[0] += translation[0]; 
		p[1] += translation[1]; 
		p[2] += translation[2];

		// Intrinsics
		T fx = T(525.0);
		T fy = T(525.0);
		T mx = T(319.5);
		T my = T(239.5);

		T xp = fx * p[0];
		T yp = fy * p[1];

		// Compute final projected point position.
		T predicted_x = ( xp / p[2] ) + mx;
		T predicted_y = ( yp / p[2] ) + my;

		residuals[0] = predicted_x - T(m_targetPoint[0]);
		residuals[1] = predicted_y - T(m_targetPoint[1]);

		return true;
	}

	static ceres::CostFunction* create(const Vector3f& sourcePoint, const Vector2f& targetPoint) {
		return new ceres::AutoDiffCostFunction<PointToPointConstraint, 2, 6>(
			new PointToPointConstraint(sourcePoint, targetPoint)
		);
	}

protected:
	const Vector3f m_sourcePoint;
	const Vector2f m_targetPoint;
	const float LAMBDA = 0.1f;
};




class Optimization {
public:
	Optimization() : 
		m_nIterations{ 20 }
	{ }

	void setNbOfIterations(unsigned nIterations) {
		m_nIterations = nIterations;
	}


	Matrix4f estimatePose(const PointCloud pointCloud) {
		std::cout << pointCloud.getCameraExtrinsics() << std::endl;
		Matrix4f estimatedPose = pointCloud.getCameraExtrinsics();
		double *pose = PoseIncrement<double>::pose6DOF(estimatedPose);
		for (int i = 0; i < m_nIterations; ++i) {
			// Prepare constraints
			ceres::Problem problem;
			prepareConstraints(pointCloud, pose, problem);

			// Configure options for the solver.
			ceres::Solver::Options options;
			configureSolver(options);

			// Run the solver (for one iteration).
			ceres::Solver::Summary summary;
			ceres::Solve(options, &problem, &summary);
			std::cout << summary.BriefReport() << std::endl;
			//std::cout << summary.FullReport() << std::endl;
			Matrix4f matrix = PoseIncrement<double>::convertToMatrix(pose);
			estimatedPose = matrix;
		}

		return estimatedPose;
	}


private:
	unsigned m_nIterations;

	void configureSolver(ceres::Solver::Options& options) {
		// Ceres options.
		options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
		options.use_nonmonotonic_steps = false;
		options.linear_solver_type = ceres::DENSE_QR;
		options.minimizer_progress_to_stdout = 1;
		options.max_num_iterations = 1;
		options.num_threads = 8;
	}

	void prepareConstraints(const PointCloud& pointCloud, double* pose, ceres::Problem& problem) const {

			// We optimize on the transformation in SE3 notation: 3 parameters for the axis-angle vector of the rotation (its length presents
			// the rotation angle) and 3 parameters for the translation vector. 
			//double *pose;
			//pose = PoseIncrement<double>::pose6DOF(estimatedPose);

			//double check if matrix is same
			//Matrix4f aa = PoseIncrement<double>::convertToMatrix(pose);
			//std::cout << aa << std::endl;

			//!!!!!!!!TO DO!!!!assert(pointClouds[i].GET2DKEYPOINTS.size() == pointClouds[i].GET3DCORRESPONDINGKEYPOINTS.size())
			std::vector<Vector3f> points3d = pointCloud.getPoints3d();
			std::vector<Vector2f> points2d = pointCloud.getPoints2dNextFrame();
			for (int j = 0; j < points3d.size(); j++){

				const auto& sourcePoint = points3d[j];
				const auto& targetPoint = points2d[j];

				if (!sourcePoint.allFinite() || !targetPoint.allFinite())
					continue;

				problem.AddResidualBlock(PointToPointConstraint::create(sourcePoint, targetPoint), NULL, pose);
			}

		return;
	}
};
