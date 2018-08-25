#include "FusionEKF.h"
#include "tools.h"
#include "Eigen/Dense"
#include <iostream>
#include <cmath>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;

  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  // measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  // measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  /**
  TODO:
    * Finish initializing the FusionEKF.
    * Set the process and measurement noises
  */
  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;

  Hj_ << 1  , 1, 0, 0,
         -1, 1, 0, 0,
         1, 1, 1, 1;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {

  // sort of constants
  float noise_ax = 9.0;
  float noise_ay = 9.0;

  /*****************************************************************************
   *  Initialization
   ****************************************************************************/
  if (!is_initialized_) {

    // first measurement
    cout << "[EKF] initializing...";

    // initializing all the variables needed by the Kalman filter

    // state vector x and its dependants
    VectorXd x = VectorXd(4);
    float px = 0;
    float py = 0;
    float vx = 0;
    float vy = 0;

    if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
      // extracting values
      float ro = measurement_pack.raw_measurements_[0];
      float theta = measurement_pack.raw_measurements_[1];
      float ro_dot = measurement_pack.raw_measurements_[2];

      // Converting radar from polar to cartesian coordinates and initialize state.
      px = cos(theta) * ro;
      py = sin(theta) * ro;
      // vx = cos(theta) * ro_dot;
      // vy = sin(theta) * ro_dot;

    } else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER) {
      // initializing the state with the given position and zero velocity
      px = measurement_pack.raw_measurements_[0];
      py = measurement_pack.raw_measurements_[1];
    }

    // initializing the state vector x
    x << px, py, vx, vy;

    // state covariance matrix P
    MatrixXd P = MatrixXd(4, 4);
    P <<  1, 0, 0, 0,
          0, 1, 0, 0,
          0, 0, 1000, 0,
          0, 0, 0, 1000;

    // transition matrix F
    MatrixXd F = MatrixXd(4, 4);
    F <<  1, 0, 1, 0,
          0, 1, 0, 1,
          0, 0, 1, 0,
          0, 0, 0, 1;

    // (empty) process covariance matrix Q
    MatrixXd Q = MatrixXd(4, 4);
    Q <<  0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0,
          0, 0, 0, 0;

    // measurement matrix H -- its value will be updated dynamically
    MatrixXd H = MatrixXd(2, 4);

    // measurement covariance matrix R -- its value will be updated dynamically
    MatrixXd R = MatrixXd(2, 2);

    // that's why we are here: initializing the kalman filter
    ekf_.Init(x, P, F, H, R, Q);

    // storing the timestamp, too
    previous_timestamp_ = measurement_pack.timestamp_;

    // done initializing, no need to predict or update
    is_initialized_ = true;

    cout << " done." << endl;

    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

  // computing the time elapsed between the current and previous measurements
  const float dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;
  previous_timestamp_ = measurement_pack.timestamp_;

  // integrating the elapsed time into the F matrix
  ekf_.F_(0, 2) = dt;
  ekf_.F_(1, 3) = dt;

  // updating the process noise covariance matrix
  const float dt_2 = dt * dt;
  const float dt_3 = dt_2 * dt;
  const float dt_4 = dt_3 * dt;

  // instead of creating the process covariance matrix from scratch every time,
  // we just update the elements that change
  ekf_.Q_(0, 0) = dt_4 * noise_ax / 4.0;
  ekf_.Q_(0, 2) = dt_3 * noise_ax / 2.0;
  ekf_.Q_(1, 1) = dt_4 * noise_ay / 4.0;
  ekf_.Q_(1, 3) = dt_3 * noise_ay / 2.0;
  ekf_.Q_(2, 0) = dt_3 * noise_ax / 2.0;
  ekf_.Q_(2, 2) = dt_2 * noise_ax;
  ekf_.Q_(3, 1) = dt_3 * noise_ay / 2.0;
  ekf_.Q_(3, 3) = dt_2 * noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {

    // Radar updates. Calculating the new Jacobian matrix first
    Tools tools;
    Hj_ = tools.CalculateJacobian(ekf_.x_);

    // adjusting the R and H matrices 
    ekf_.R_ = R_radar_;
    ekf_.H_ = Hj_;

    // finally: calling the updateEKF function on the Kalman filter
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);

  } else {
    // Laser updates. Updating the R and the H matrices first...
    ekf_.R_ = R_laser_;
    ekf_.H_ = H_laser_;

    // ...then performing the update operation
    ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
