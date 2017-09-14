#pragma once
#include <stdio.h>

void AllocateIntVector(int size, int** vector);

void AllocateDoubleVector(int size, double** vector);

void AllocateStr(int size, char** str);

void FreeIntVector(int *vector);

void FreeDoubleVector(double *vector);

void FreeStr(char *str);

void AllocateDoubleMatrix(int row, int col, double*** matrix);

void FreeDoubleMatrix(int row, int col, double** matrix);

void ShowDoubleVector(int size, double* vector, const char* data);

void ShowIntVector(int size, int* vector, const char* data);

void ShowDoubleMatrix(int row, int col, double** matrix, const char* data);

void SumVector(int size, double a, double b, double* x, double* y, double** sum);

void SumMatrix(int row, int col, double a, double b, double** matrix1, double** matrix2, double*** matrixSum);

double ProductVector(int size, double* x, double* y);

double ProductQuad(int size, double* x, double** M);

void ChangeSigneVector(int size, double** x);

double NormMax(int size, double *x, double *y);

double min(double a, double b);

double min3(double a, double b, double c);

void Floor_X(int size, double* x, double** xRes);

void Round_X(int size, double* x, double** xRes);

void Ceil_X(int size, double* x, double** xRes);

void RandomDelta_X(int size, double** x, double delta);

void Random01_X(int size, double** x);

void RandomBinary_X(int size, double** x);

void RandomInterval_X(int size, int* lb, int*ub, double** x);

void RandomInt_X(int size, int* lb, int*ub, double** x);

void RandomMat(int rows, int cols, double*** M, double delta);

int RandomInt(double a, double b);

double RandomReal(double a, double b);

void Mat2CPXMat(const int rows, const int cols, double** M,
                int** matbeg, int** matcnt, int** matind, double** matval);

void Mat2CPXQCMat(const int size, double** M,
                  int* quadnzcnt, int** quadrow, int** quadcol, double** quadval);

void Vec2CPXVec(const int cols,double* qi,int* linnzcnt,int** linind,double** linval);

void SparseMat2Mat(const int rows, const int cols, const int nz,
                   int* I, int* J, double* val, double*** M);

int NnzMat(const int rows, const int cols, double** M);

int NnzVec( const int size, double* v);

void ReadDoubleVector(const char* filename, const int size, double** v);

void ReadDoubleVectorFP(FILE* fp, const int size, double** v);

void ReadIntVector(const char* filename, const int size, int** v);

void ReadIntVectorFP(FILE* fp, const int size, int** v);

void ReadDoubleMatrix(const char* filename, const int rows, const int cols, double*** M);

void ReadDoubleMatrixFP(FILE* fp, const int rows, const int cols, double*** M);

void ReadDoubleMatrixA(const char* filename, int* rows, int* cols, double*** M);

void ReadDoubleMatrixAFP(FILE* fp, int* rows, int* cols, double*** M);

void Eye(const int size, double*** I);

void Ones(const int size, double** V);

double* ExtractDoubleVector(const int begin, const int length, double* v);

//int ReadMMMatrix(const char* filename,int* rows,int* cols,double*** A);
