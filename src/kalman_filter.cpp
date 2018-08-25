#define M_PI  3.14159265358979323846
// #define _USE_MATH_DEFINES

#include "kalman_filter.h"
#include <cmath>
#include <iostream>

using Eigen::MatrixXd;
using Eigen::VectorXd;


// Please note that the Eigen library does not initialize 
// VectorXd or MatrixXd objects with zeros upon creation.

KalmanFilter::KalmanFilter() {}

KalmanFilter::~KalmanFilter() {}

void KalmanFilter::Init(VectorXd &x_in, MatrixXd &P_in, MatrixXd &F_in,
                        MatrixXd &H_in, MatrixXd &R_in, MatrixXd &Q_in) {
  x_ = x_in;
  P_ = P_in;
  F_ = F_in;
  H_ = H_in;
  R_ = R_in;
  Q_ = Q_in;
}


void KalmanFilter::Predict() {

  x_ = F_ * x_;
  MatrixXd Ft = F_.transpose();
  P_ = F_ * P_ * Ft + Q_;
}


void KalmanFilter::Update(const VectorXd &z) {

  // the well known Kalman filter equations -- the classical case
  VectorXd z_pred = H_ * x_;
  VectorXd y = z - z_pred;

  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;

  MatrixXd S = H_ * PHt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  // new estimate
  x_ = x_ + (K * y);
  int x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
  // reviewer's suggestion: You could simplify this expression to P_ -= K * H_ * P_;
}


void KalmanFilter::UpdateEKF(const VectorXd &z) {

  float px = x_(0);
  float py = x_(1);
  float vx = x_(2);
  float vy = x_(3);

  // sanity check
  float d = sqrt(px*px + py*py);
  if ( abs(d) < 0.0001 || abs(px) < 0.0001 ) {

    // Houston, we have a little problem
    std::cout << "[KalmanFilter::UpdateEKF] Error: upcoming division by a near-zero value. Skipping measurement processing part." << std::endl;

    return;
  }

  // computing h(x')...
  VectorXd hx = VectorXd(3);
  hx << d,
        atan2(py, px),
        (px*vx + py*vy) / d;

  // ...then y
  VectorXd y = z - hx;

  // normalizing theta in the measurement error vector
  // if ( y(1) < -M_PI ) {
  //     y(1) += 2 * M_PI;

  // } else if ( y(1) > M_PI ) {
  //     y(1) -= 2 * M_PI;
  // }

  // reviewer's suggestion for normalizing phi:
  const float phi = y(1);
  y(1) = atan2(sin(phi), cos(phi));


  // and the calculation goes on as in the basic case
  MatrixXd Ht = H_.transpose();
  MatrixXd PHt = P_ * Ht;

  MatrixXd S = H_ * PHt + R_;
  MatrixXd Si = S.inverse();
  MatrixXd K = PHt * Si;

  // new estimate
  x_ = x_ + (K * y);
  int x_size = x_.size();
  MatrixXd I = MatrixXd::Identity(x_size, x_size);
  P_ = (I - K * H_) * P_;
  // reviewer's suggestion: You could simplify this expression to P_ -= K * H_ * P_;
}
