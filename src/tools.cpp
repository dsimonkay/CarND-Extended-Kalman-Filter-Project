#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const std::vector<VectorXd> &estimations,
                              const std::vector<VectorXd> &ground_truth) {
  VectorXd rmse(4);
  rmse << 0, 0, 0, 0;

  // checking the validity of the following inputs:
  //  * the estimation vector size should not be zero
  //  * the estimation vector size should equal ground truth vector size
  // ... your code here
  if (estimations.size() == 0) {
    cout << "[Tools::CalculateRMSE] Error: empty input received." << endl;
    return rmse;

  } else if ( estimations.size() != ground_truth.size() ) {
    cout << "[Tools::CalculateRMSE] Error: estimation vector size doesn't match that of the ground truth vector." << endl;
    return rmse;
  }

  VectorXd residuals(4);
  VectorXd residuals_sq(4);

  // accumulating squared residuals
  for (int i = 0; i < estimations.size(); ++i) {

    residuals = estimations[i] - ground_truth[i];
    residuals_sq = residuals.array() * residuals.array();
    rmse += residuals_sq;
  }

  // calculating the mean
  rmse = rmse / estimations.size();

  // calculating the squared root
  rmse = rmse.array().sqrt();

  // returning the result
  return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3, 4);

	// recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	// our first helper variable
	float base = px * px + py * py;

	// checking division by zero
	if (fabs(base) < 0.0001) {
		cout << "[Tools::CalculateJacobian] - Error: division by a (near) zero value." << endl;
    Hj << 0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0;
		return Hj;
	}

	// helper variables, continuned
	float denom_base = 1 / base;
	float denom_sqrt = sqrt(denom_base);
	float denom_sqrt_pow_3 = pow(denom_sqrt, 3);

	//compute the Jacobian matrix
	Hj << px * denom_sqrt, py * denom_sqrt, 0, 0,
        -py * denom_base, px * denom_base, 0, 0,
		    py * (vx*py - vy*px) * denom_sqrt_pow_3, px * (vy*px - vx*py) * denom_sqrt_pow_3, px * denom_sqrt, py * denom_sqrt;

	return Hj;
}
