// TestMatrix.cpp : 定义控制台应用程序的入口点。
//

#include "Matrix.h"

using namespace std;
int main()
{
	// test generator
	cout << "generate empty matrix : ";
	Matrix A;
	cout << A.isempty() << endl;

	// test resize
	cout << "resize a matrix" << endl;
	A.resize(3, 3);
	cout << "new size:" << A.rows() << "*" << A.cols() << endl;

	// test matrix elements 
	cout << "modify matrix elements by [][]" << endl;
	for (auto i = 0; i < A.rows(); i++) {
		for (auto j = 0; j < A.cols(); j++) {
			A[i][j] = i + j;
		}
	}
	cout << A << endl;

	// test =
	cout << "test B=A" << endl;
	Matrix B;
	B = A;
	cout << "B = " << endl;
	cout << B << endl;

	// test +=
	cout << "test +=: B+=A " << endl;
	B += A;
	cout << B << endl;

	// test trans
	cout << "test trans" << endl;
	B[0][1] = 100;
	auto C = trans(B);
	cout << "B = " << endl  << B << endl;
	cout << "trans(B) = " << endl << C << endl;
	cout << B.trans() << endl;

	// test add_block_by_row
	cout << "test add_block_by_row" << endl;
	auto D = B;
	D.add_block_by_row(C);
	cout << D << endl;

	// test add_block_by_col
	cout << "test add_block_by_col" << endl;
	D = B;
	D.add_block_by_col(C);
	cout << D << endl;

	// test add_row
	cout << "test add_row" << endl;
	vector<double> v{ {1.1,1.5,2} };
	D = B;
	D.add_row(v);
	cout << D << endl;

	// test add_col
	cout << "test add_row" << endl;
	D = B;
	D.add_col(v);
	cout << D << endl;

	// test swap_row
	cout << "test swap_row" << endl;
	D.swap_row(0, 1);
	cout << D << endl;
	cout << "test swap_col" << endl;
	D.swap_col(1, 2);
	cout << D << endl;

	// test ==
	cout << "test ==" << endl;
	cout << (B == D) << endl;
	D = B;
	cout << (B == D) << endl;
	cout << (B != D) << endl;

	// test inverse
	cout << "test inverse matrix" << endl;
	cout << inverse(B) << endl;
	cout << B*inverse(B) << endl;

	// test determinant
	cout << "test determinant" << endl;
	cout << "B = " << endl;
	cout << B << endl;
	cout << "det(B) = " << det(B) << endl;
	cout << "det2(B) = " << det2(B) << endl;

	// test LU decomposition
	cout << "test LU decomposition" << endl;
	cout << LU(B) << endl;

	// test *=
	cout << "test *=: B*=A" << endl;
	B *= A;
	cout << B << endl;

	// test double*Matrix and Matrix*double
	cout << 2 * B << endl;
	cout << B * 2 << endl;

	// test extractmatrix
	cout << "test extractmatrix from B" << endl;
	vector<int> ridx{ {0,2} };
	vector<int> cidx{ {1,0} }; // the index set could be not in order
	cout << extractmatrix(B, ridx, cidx) << endl;

	// test norms
	cout << B << endl;
	cout << "norm_1(B) and norm(B,NORM_1):" << norm_1(B) << "," << norm(B, NORM_1) << endl;
	cout << "norm_2(B) and norm(B,NORM_2):" << norm_2(B) << "," << norm(B, NORM_2) << endl;
	cout << "norm_inf(B) and norm(B,NORM_INF):" << norm_inf(B) << "," << norm(B, NORM_INF) << endl;
	cout << "norm_fro(B) and norm(B,NORM_FRO):" << norm_fro(B) << "," << norm(B, NORM_FRO) << endl;
	cout << "norm_max(B) and norm(B,NORM_MAX):" << norm_max(B) << "," << norm(B, NORM_MAX) << endl;

	// test rank
	cout << "rank of B: " << rang(B) << endl;

	// test condition number
	cout << "condition number of B: " << cond(B) << endl;

	// test ispsd and ispd
	cout << B.ispsd() << ", " << B.ispd() << endl;

	// test matrix power
	cout << mpower(B, 1) << endl;
	
	// test special matrices
	cout << eye(3) << endl;
	cout << zeros(2, 3) << endl;
	cout << ones(3, 2) << endl;

	// test data formats
	// 2darray format
	//auto a = B.data_2Darray();
	// col major array format
	auto b = B.data_col_major();
	cout << B << endl;
	cout << b[0] << b[1] << b[2] << b[3] << b[4] << endl;

	// test svd
	cout << "test sdv" << endl;
	Matrix S, U, VT;
	SVD(B, S, U, VT);
	cout << "S: " << S << endl;
	cout << "U: " << U << endl;
	cout << "VT: " << VT << endl;
	auto S1 = diag(S);
	cout << "U*S*VT" << U*S1*VT << endl;

	// test eig
	cout << "test eig" << endl;
	Matrix AA(3, 3);
	AA[0][0] = 1.6294;
	AA[0][1] = 1.8192;
	AA[0][2] = 0.4055;
	AA[1][1] = 1.2647;
	AA[1][2] = 0.6444;
	AA[2][2] = 1.9150;
	for (auto i = 0; i < 3; i++) {
		for (auto j = 0; j < i; j++) {
			AA[i][j] = AA[j][i];
		}
	}
	Matrix Vr, Vi, REr, REi, LEr, LEi;
	EIG(AA, Vr, Vi, REr, REi, LEr, LEi);
	printMatrix(AA, "AA", NULL);
	printMatrix(Vr, "Eigenvalues", NULL);
	printMatrix(REr, "Eigenvectors", NULL);

	
	//Matrix BB = genMatlabMatrix("[1 7 3; 2 9 12; 5 22 7]", 3, 3);
	//Matrix BB = genMatlabMatrix("[1 2 3; 3 1 2; 2 3 1]", 3, 3);
	Matrix BB = genMatlabMatrix("[-1.01, 3.98, 3.30, 4.43, 7.31;0.86, 0.53, 8.26, 4.96, -6.43;-4.60, -7.04, -3.89, -7.66, -6.16;3.31, 5.29, 8.20, -7.33, 2.47;-4.81, 3.55, -1.51, 6.18, 5.58]",5,5);
	//Matrix BB = genMatlabMatrix("[3 1 0; 0 3 1; 0 0 3]",3,3);
	//BB = trans(BB);
	printMatrix(BB, "B:",NULL);
	EIG(BB, Vr, Vi, REr, REi, LEr, LEi);
	printMatrix(Vr, "Vr:", NULL);
	printMatrix(Vi, "Vi:", NULL);
	printMatrix(REr, "REr:", NULL);
	printMatrix(REi, "REi:", NULL);
	printMatrix(LEr, "LEr:", NULL);
	printMatrix(LEi, "LEi:", NULL);
	printMatrix(BB*REr - REr*diag(Vr), "BB*REr - REr*diag(Vr):", NULL);


	cout << REi.iszeros() << endl;

	// test matrix format conversion
	cpxmat cm;
	format_cpxmat(B, cm);
	cpxqcmat qcm;
	format_cpxqcmat(B, qcm);


}

