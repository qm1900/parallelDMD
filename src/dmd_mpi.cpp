#include <mpi.h>
#include <Eigen/Cholesky>
#include <Eigen/Dense>
#include <Eigen/Eigenvalues>
#include <Eigen/SVD>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

using namespace std;
using namespace Eigen;

typedef Matrix<double, Dynamic, Dynamic, RowMajor> RowMatrixXd;
typedef Matrix<complex<double>, Dynamic, Dynamic, RowMajor> RowMatrixXcd;

int main(int argc, char *argv[]) {
    int npes, myrank;

    MPI::Init(argc, argv);

    myrank = MPI::COMM_WORLD.Get_rank();
    npes = MPI::COMM_WORLD.Get_size();

    long xn = atol(argv[2]);
    long tn = atol(argv[3]);
    long long position = 0;

    int xnProc = ceil((xn + 0.0) / (npes + 0.0));
    int outputPrecision = 9;
    int modeCount = tn;

    RowMatrixXd snapshotsProc(xnProc, tn + 1);
    RowMatrixXd snapshotsInProdProc(tn, tn);
    RowMatrixXd r(tn, tn);
    RowMatrixXd sigmaPseudoInverse(tn, tn);
    RowMatrixXd v(tn, tn), uProc(xnProc, tn);
    RowMatrixXd aTildeProc(tn, tn), aTilde(tn, tn);
    RowMatrixXd x0TildeProc(tn, 1), x0Tilde(tn, 1);

    RowMatrixXcd eigenvec(tn, tn);
    RowMatrixXcd dmdModesAllProc(xnProc, tn);

    VectorXd dmdModeProc(xnProc);
    VectorXd dmdMode(xnProc * npes);

    double *snapshotsRecvBuf = new double[xnProc * (tn + 1)];
    double *tntnBuf = new double[tn * tn];
    double *tn1Buf = new double[tn * 1];

    double *tntn_pt = new double[tn * tn];
    double *tn1_pt = new double[tn * 1];
    double *tntn2_pt = new double[tn * tn];
    double *xnProc_pt = new double[xnProc];
    double *xnProcnpes_pt = new double[xnProc * npes];

    complex<double> *tntnc_pt = new complex<double>[tn * tn];

    MPI::COMM_WORLD.Barrier();
    double timeBegin = MPI::Wtime();

    /* input matrix from a file */
    double *snapshots = NULL;

    if (myrank == 0) {
        if (modeCount > tn) {
            cout << "Invalid output modes number" << endl;
            exit(0);
        }

        cout << "Initializing" << endl;

        snapshots = new double[xn * (tn + 1)];
        ifstream inputFile(argv[1]);
        if (inputFile.is_open()) {
            while (!inputFile.eof() && position < (xn * (tn + 1))) {
                inputFile >> snapshots[position];
                position++;
            }
        } else {
            cout << "input file not found." << endl;
            exit(0);
        }
        inputFile.close();

        cout << "Running" << endl;
    }
    /****************************/

    MPI::COMM_WORLD.Scatter(snapshots, xnProc * (tn + 1), MPI::DOUBLE,
                            snapshotsRecvBuf, xnProc * (tn + 1), MPI::DOUBLE,
                            0);

    // delete[] snapshots;

    snapshotsProc = Map<RowMatrixXd>(snapshotsRecvBuf, xnProc, tn + 1);

    // delete[] snapshotsRecvBuf;

    snapshotsInProdProc = snapshotsProc.block(0, 0, xnProc, tn).transpose() *
                          snapshotsProc.block(0, 0, xnProc, tn);

    tntn_pt = snapshotsInProdProc.data();

    MPI::COMM_WORLD.Reduce(tntn_pt, tntnBuf, tn * tn, MPI::DOUBLE, MPI::SUM, 0);

    if (myrank == 0) {
        VectorXd sigma(tn);

        RowMatrixXd snapshotsInProd = Map<RowMatrixXd>(tntnBuf, tn, tn);

        LLT<RowMatrixXd, Upper> cholesky(snapshotsInProd);
        r = cholesky.matrixU();

        JacobiSVD<RowMatrixXd> svd(r, ComputeThinV);
        sigma = svd.singularValues();
        v = svd.matrixV();

        for (int i = 0; i < tn; ++i) {
            for (int j = 0; j < tn; ++j) {
                if (i == j && sigma(i) != 0) {
                    sigmaPseudoInverse(i, j) = 1.0 / sigma(i);
                } else
                    sigmaPseudoInverse(i, j) = 0;
            }
        }

        tntn_pt = sigmaPseudoInverse.data();
        tntn2_pt = v.data();
    }

    MPI::COMM_WORLD.Bcast(tntn_pt, tn * tn, MPI::DOUBLE, 0);
    MPI::COMM_WORLD.Bcast(tntn2_pt, tn * tn, MPI::DOUBLE, 0);

    sigmaPseudoInverse = Map<RowMatrixXd>(tntn_pt, tn, tn);
    v = Map<RowMatrixXd>(tntn2_pt, tn, tn);

    uProc = snapshotsProc.block(0, 0, xnProc, tn) * v * sigmaPseudoInverse;

    if (myrank != npes - 1) {
        aTildeProc = uProc.adjoint() * snapshotsProc.block(0, 1, xnProc, tn) *
                     v * sigmaPseudoInverse;
        x0TildeProc = uProc.adjoint() * snapshotsProc.block(0, 0, xnProc, 1);

    } else {
        aTildeProc = uProc.block(0, 0, xn - xnProc * (npes - 1), tn).adjoint() *
                     snapshotsProc.block(0, 1, xn - xnProc * (npes - 1), tn) *
                     v * sigmaPseudoInverse;
        x0TildeProc =
          uProc.block(0, 0, xn - xnProc * (npes - 1), tn).adjoint() *
          snapshotsProc.block(0, 0, xn - xnProc * (npes - 1), 1);
    }

    tntn_pt = aTildeProc.data();
    tn1_pt = x0TildeProc.data();

    MPI::COMM_WORLD.Reduce(tntn_pt, tntnBuf, tn * tn, MPI::DOUBLE, MPI::SUM, 0);
    MPI::COMM_WORLD.Reduce(tn1_pt, tn1Buf, tn * 1, MPI::Double, MPI::SUM, 0);

    if (myrank == 0) {
        aTilde = Map<RowMatrixXd>(tntnBuf, tn, tn);
        x0Tilde = Map<RowMatrixXd>(tn1Buf, tn, 1);

        ComplexEigenSolver<RowMatrixXd> ces(aTilde);

        eigenvec = ces.eigenvectors();
        tntnc_pt = eigenvec.data();

        ofstream ofeigenvalReal, ofeigenvalImag, ofamplitudeReal,
          ofamplitudeImag;

        ofeigenvalReal.open("eigenvalReal.dat",
                            ofstream::trunc | ofstream::out);
        ofeigenvalImag.open("eigenvalImag.dat",
                            ofstream::trunc | ofstream::out);
        ofamplitudeReal.open("amplitudeReal.dat",
                             ofstream::trunc | ofstream::out);
        ofamplitudeImag.open("amplitudeImag.dat",
                             ofstream::trunc | ofstream::out);

        for (int i = 0; i < modeCount; ++i) {

          ofeigenvalReal << setprecision(outputPrecision)
                         << ces.eigenvalues().row(i).real() << endl;

          ofeigenvalImag << setprecision(outputPrecision)
                         << ces.eigenvalues().row(i).imag() << endl;

          ofamplitudeReal << setprecision(outputPrecision)
                          << (eigenvec.inverse() * x0Tilde).row(i).real()
                          << endl;

          ofamplitudeImag << setprecision(outputPrecision)
                          << (eigenvec.inverse() * x0Tilde).row(i).imag()
                          << endl;
        }
    }

    MPI::COMM_WORLD.Bcast(tntnc_pt, tn * tn, MPI::DOUBLE_COMPLEX, 0);
    eigenvec = Map<RowMatrixXcd>(tntnc_pt, tn, tn);

    dmdModesAllProc = uProc * eigenvec;

    for (int i = 0; i < modeCount; ++i) {
        dmdModeProc = dmdModesAllProc.col(i).real();

        xnProc_pt = dmdModeProc.data();

        MPI::COMM_WORLD.Barrier();

        MPI::COMM_WORLD.Gather(xnProc_pt, xnProc, MPI::DOUBLE, xnProcnpes_pt,
                               xnProc, MPI::DOUBLE, 0);

        if (myrank == 0) {
            dmdMode = Map<VectorXd>(xnProcnpes_pt, xnProc * npes);

            ofstream ofdmdMode;
            ofdmdMode.open("dmdMode" + to_string(i) + ".dat",
                           ofstream::trunc | ofstream::out);
            ofdmdMode << setprecision(outputPrecision) << dmdMode.head(xn)
                      << endl;
        }
    }

    MPI::COMM_WORLD.Barrier();

    double timeFinish = MPI::Wtime();
    double timing = timeFinish - timeBegin;
    double timingMax;

    MPI::COMM_WORLD.Reduce(&timing, &timingMax, 1, MPI::DOUBLE, MPI::MAX, 0);

    if (myrank == 0) {
        cout << "Calculation finished successfully." << endl
             << "Spatial: " << xn << " points." << endl
             << "Temporal: " << tn + 1 << " state vectors." << endl
             << "Parallelism: " << npes << " cores." << endl
             << "Time consumption: " << timingMax << " s." << endl
             << "Output DMD modes: " << modeCount << endl;
    }

    MPI::Finalize();
    return 0;
}
