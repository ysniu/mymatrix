/*********************************************/
/* Language: C                               */
/* Date:  08 Nov 2010                        */
/*********************************************/
#pragma once
#include "MVTool.h"
#include <malloc.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <time.h>

#ifndef CPX_ZERO
#define CPX_ZERO 1E-16
#endif


/*********************************************/
void AllocateIntVector(int size, int** vector)
{
    (*vector) = (int*) calloc(size, sizeof(int));
}
/*********************************************/
void AllocateDoubleVector(int size, double** vector)
{
    (*vector) = (double*) calloc(size, sizeof(double));
}
/*********************************************/
void AllocateStr(int size, char** str)
{
    (*str) = (char*) calloc(size, sizeof(char));
}
/*********************************************/
void FreeIntVector(int *vector)
{
    if (vector!=NULL)
    {
        free(vector);
        vector = NULL;
    }
}
/*********************************************/
void FreeDoubleVector(double *vector)
{
    if (vector!=NULL)
    {
        free(vector);
        vector = NULL;
    }
}
/*********************************************/
void FreeStr(char *str)
{
    if (str!=NULL)
    {
        free(str);
        str = NULL;
    }
}
/*********************************************/
void AllocateDoubleMatrix(int row, int col, double*** matrix)
{
    int i;
    (*matrix) = (double**)calloc(row, sizeof(double*));
    for (i = 0; i<row; i++)
    {
        (*matrix)[i] = (double*)calloc(col, sizeof(double));
    }
}
/*********************************************/
void FreeDoubleMatrix(int row, int col, double** matrix)
{
    int i;
    if (matrix != NULL)
    {
        for (i = 0; i<row; i++)
        {
            if (matrix[i] != NULL)
            {
                free(matrix[i]);
                matrix[i] = NULL;
            }
        }
        free(matrix);
        matrix = NULL;
    }
}
/*********************************************/
void ShowDoubleVector(int size, double* vector, const char* data)
{
    printf(data);
    printf("\n");
    int i;
    for (i=0; i<size; i++)
    {
        printf("\n\t %lf ", vector[i]);
    }
    printf("\n");
}
/*********************************************/
void ShowIntVector(int size, int* vector, const char* data)
{
    int i;
    printf(data);
    printf("\n");
    for (i=0; i<size; i++)
    {
        printf("\n\t %d ", vector[i]);
    }
    printf("\n");
}
/*********************************************/
void ShowDoubleMatrix(int row, int col, double** matrix, const char* data)
{
    int i,j;
    printf(data);
    printf("\n");
    for (i = 0; i<row; i++)
    {
        for (j = 0; j<col; j++)
        {
            printf("%lf\t",matrix[i][j]);
        }
        printf("\n");
    }
}
/*********************************************/
//matrixSum = a*matrix1 + b*matrix2
void SumMatrix(int row, int col, double a, double b, double** matrix1, double** matrix2, double*** matrixSum)
{
    int i, j;
    if (*matrixSum==NULL)
    {
        AllocateDoubleMatrix(row,col,matrixSum);
    }
    for (i = 0; i<row; i++)
    {
        for (j = 0; j<col; j++)
        {
            (*matrixSum)[i][j] = a*matrix1[i][j] + b*matrix2[i][j];
        }
    }
}
/*********************************************/
// sum = a*x+b*y
void SumVector(int size, double a, double b, double* x, double* y, double** sum)
{
    int i;
    if (*sum==NULL)
    {
        AllocateDoubleVector(size,sum);
    }
    for (i = 0; i<size; i++)
    {
        (*sum)[i] = a*x[i] + b*y[i];
    }
}
/*********************************************/
// product = x*y
double ProductVector(int size, double* x, double* y)
{
    int i;
    double product = 0.0;
    for (i = 0; i<size; i++)
    {
        product += x[i] * y[i];
    }
    return product;
}
/*********************************************/
// product = xT*M*x
double ProductQuad(int size, double* x, double** M)
{
    int i,j;
    double objval = 0.0;
    for (i = 0; i<size; i++)
    {
        objval = objval + M[i][i] * x[i] * x[i];
        for (j = i+1; j<size; j++)
        {
            objval = objval + x[i] * x[j] * M[i][j] + x[j] * x[i] * M[j][i];
        }
    }
    return objval;
}

/*********************************************/
double NormMax(int size, double *x, double *y)
{
    int i = 0;
    double norme = 0.;
    double value = 0.;
    for (i = 0; i<size; i++)
    {
        value = fabs(x[i]-y[i]);
        if (norme <= value)
            norme = value;
    }
    return norme;
}
/*********************************************/
double min(double a, double b)
{
    if (b<a) return b;
    else return a;
}
/*********************************************/
double min3(double a, double b, double c)
{
    double t = 0.0;
    t = min(a, b);
    t = min(t, c);
    return t;
}

/*********************************************/
void ChangeSigneVector(int size, double** x)
{
    int i;
    for (i = 0; i<size; i++)
    {
        (*x)[i] = -(*x)[i];
    }
}
/*********************************************/
void Floor_X(int size, double* x, double** xRes)
{
    int i = 0;
    if (*xRes==NULL)
    {
        AllocateDoubleVector(size,xRes);
    }
    for (i = 0; i<size; i++)
    {
        (*xRes)[i] = floor(x[i]);
    }
}
/************************************************/
void Ceil_X(int size, double* x, double** xRes)
{
    int i = 0;
    if (*xRes==NULL)
    {
        AllocateDoubleVector(size,xRes);
    }
    for (i = 0; i<size; i++)
    {
		(*xRes)[i] = ceil(x[i]);
    }
}
/************************************************/
void Round_X(int size, double* x, double** xRes)
{
    int i = 0;
    double value = 0.0;
    if (*xRes==NULL)
    {
        AllocateDoubleVector(size,xRes);
    }
    for (i = 0; i<size; i++)
    {
        value = floor(x[i]);
        if ((x[i] - value)<0.5) (*xRes)[i] = value;
        else (*xRes)[i] = 1.0 + value;
    }
}

/***********************************************
    x in [-delta,delta]
************************************************/
void RandomDelta_X(int size, double** x, double delta)
{
    int i;
    if (*x==NULL)
    {
        AllocateDoubleVector(size,x);
    }
	srand((unsigned)time(NULL));
    for (i = 0; i<size; i++)
    {
        (*x)[i] = -delta + 2*delta*((double)abs(rand())/ RAND_MAX);
    }
}
/***********************************************
    x in [0,1]
************************************************/
void Random01_X(int size, double** x)
{
    int i;
    if (*x==NULL)
    {
        AllocateDoubleVector(size,x);
    }
    srand((unsigned)time(NULL));
    for (i = 0; i<size; i++)
    {
        (*x)[i] =  (double)abs(rand())/ RAND_MAX;
    }
}
/***********************************************
    x in [lb,ub]
************************************************/
void RandomInterval_X(int size, double* lb, double*ub, double** x)
{
    int i;
    if (*x==NULL)
    {
        AllocateDoubleVector(size,x);
    }
    srand((unsigned)time(NULL));
    for (i = 0; i<size; i++)
    {
        (*x)[i] =  lb[i] + (ub[i]-lb[i]) * (double)rand() / RAND_MAX;
    }
}
/***********************************************
    x binary variable
************************************************/
void RandomBinary_X(int size, double** x)
{
    int i;
    if (*x==NULL)
    {
        AllocateDoubleVector(size,x);
    }
    srand((unsigned)time(NULL));
    for (i = 0; i<size; i++)
    {
        if ((double)abs(rand())/ RAND_MAX<0.5)(*x)[i] = 0 ;
        else (*x)[i] = 1;
    }
}
/***********************************************
    x integer in [lb,ub]
************************************************/
void RandomInt_X(int size, double* lb, double*ub, double** x)
{
    int i;
    if (*x==NULL)
    {
        AllocateDoubleVector(size,x);
    }
    srand((unsigned)time(NULL));
    for (i = 0; i<size; i++)
    {
        (*x)[i] =  (int)(lb[i] + (ub[i]-lb[i]) * (double)abs(rand()) / RAND_MAX);
    }
}

/***********************************************
    M[i][j] in [-delta,delta]
************************************************/
void RandomMat(int rows, int cols, double*** M, double delta)
{
    int i,j;
    if (*M==NULL)
    {
        AllocateDoubleMatrix(rows,cols,M);
    }
    srand((unsigned)time(NULL));
    for (i = 0; i<rows; i++)
    {
        for (j=0; j<cols; j++) (*M)[i][j] = -delta + 2*delta*((double)abs(rand())/ RAND_MAX);
    }
}
/************************************************/
// return a random integer in [a,b]
int RandomInt(double a, double b)
{
    return (int)(a + fabs(b-a) * (double)abs(rand()) / RAND_MAX);
}

/************************************************/
// return a random real value in [a,b]
double RandomReal(double a, double b)
{
    return (a + fabs(b-a) * (double)abs(rand()) / RAND_MAX);
}

/************************************************/
void Mat2CPXMat(const int rows, const int cols, double** M,
                int** matbeg, int** matcnt, int** matind, double** matval)
{
    int i,j;
    int nzcols,nz;

    AllocateIntVector(cols,matbeg);
    AllocateIntVector(cols,matcnt);

    nz=NnzMat(rows,cols,M);
    AllocateIntVector(nz,matind);
    AllocateDoubleVector(nz,matval);

    nz=0;
    for (j=0;j<cols;j++)
    {
        nzcols=0;
        (*matbeg)[j]=nz;
        for (i=0;i<rows;i++)
        {
            if (fabs(M[i][j])>CPX_ZERO)
            {
                (*matval)[nz]=M[i][j];
                (*matind)[nz]=i;
                if (nzcols==0) (*matbeg)[j]=nz;
                nz++;
                nzcols++;
            }
        }
        (*matcnt)[j]=nzcols;
    }
}

/***************************************************/
/* return the number of non zero elements of Matrix*/
int NnzMat( const int rows, const int cols,double** M)
{
    int i,j;
    int nz=0;
    for (i=0;i<rows;i++)
    {
        for (j=0;j<cols;j++)
        {
            if (fabs(M[i][j])>CPX_ZERO)
            {
                nz++;
            }
        }
    }
    return nz;
}

/************************************************/
void Vec2CPXVec(const int cols,double* qi,int* linnzcnt,int** linind,double** linval)
{
    int i;
    int nz;

    *linnzcnt = NnzVec(cols,qi);
    AllocateIntVector(*linnzcnt,linind);
    AllocateDoubleVector(*linnzcnt,linval);

    nz=0;
    for (i=0;i<cols;i++)
    {
        if (fabs(qi[i])>CPX_ZERO)
        {
            (*linval)[nz]=qi[i];
            (*linind)[nz]=i;
            nz++;
        }
    }
}

/************************************************/
int NnzVec( const int size, double* v)
{
    int i;
    int nz=0;
    for (i=0;i<size;i++)
    {
        if (fabs(v[i])>CPX_ZERO)
        {
            nz++;
        }
    }
    return nz;
}

/************************************************/
void Mat2CPXQCMat(const int size, double** M,
                  int* quadnzcnt, int** quadrow, int** quadcol, double** quadval)
{
    int i,j;
    int nz;

    *quadnzcnt=NnzMat(size,size,M);
    AllocateIntVector(*quadnzcnt,quadrow);
    AllocateIntVector(*quadnzcnt,quadcol);
    AllocateDoubleVector(*quadnzcnt,quadval);

    nz=0;
    for (j=0;j<size;j++)
    {
        for (i=0;i<size;i++)
        {
            if (fabs(M[i][j])>CPX_ZERO)
            {
                (*quadval)[nz]=M[i][j];
                (*quadrow)[nz]=i;
                (*quadcol)[nz]=j;
                nz++;
            }
        }
    }
}

/************************************************/
void ReadDoubleVector(const char* filename, const int size, double** v)
{
    int i;
    FILE *pf;
    pf = fopen(filename,"r+");
    if (*v==NULL)
    {
        AllocateDoubleVector(size,v);
    }
    for (i=0;i<size;i++) fscanf(pf,"%lf",&v[0][i]);
    fclose(pf);
}

/************************************************/
void ReadDoubleVectorFP(FILE* fp, const int size, double** v)
{
    int i;
    if (*v==NULL)
    {
        AllocateDoubleVector(size,v);
    }
    for (i=0;i<size;i++) fscanf(fp,"%lf",&v[0][i]);
}

/************************************************/
void ReadIntVector(const char* filename, const int size, int** v)
{
    int i;
    FILE *pf;
    pf = fopen(filename,"r+");
    if (*v==NULL)
    {
        AllocateIntVector(size,v);
    }
    for (i=0;i<size;i++) fscanf(pf,"%d",&v[0][i]);
    fclose(pf);
}
/************************************************/
void ReadIntVectorFP(FILE* fp, const int size, int** v)
{
    int i;
    if (*v==NULL)
    {
        AllocateIntVector(size,v);
    }
    for (i=0;i<size;i++) fscanf(fp,"%d",&v[0][i]);
}
/************************************************/
// READ MATRIX WITH GIVEN SIZE ROWS X COLS
void ReadDoubleMatrix(const char* filename, const int rows, const int cols, double*** M)
{
    int i,j;
    FILE *pf;
    pf = fopen(filename,"r+");
    if (*M==NULL)
    {
        AllocateDoubleMatrix(rows,cols,M);
    }
    for (i=0;i<rows;i++)
        for (j=0;j<cols;j++) fscanf(pf,"%lf",&M[0][i][j]);
    fclose(pf);
}
/************************************************/
// READ MATRIX WITH GIVEN SIZE ROWS X COLS FROM FILE POINTER
void ReadDoubleMatrixFP(FILE* fp, const int rows, const int cols, double*** M)
{
    int i,j;
    if (*M==NULL)
    {
        AllocateDoubleMatrix(rows,cols,M);
    }
    for (i=0;i<rows;i++)
        for (j=0;j<cols;j++) fscanf(fp,"%lf",&M[0][i][j]);
}

/************************************************/
// READ MATRIX AND SIZE FROM FILE, ROWS AND COLS ARE GIVEN IN THE FILE
void ReadDoubleMatrixA(const char* filename, int* rows, int* cols, double*** M)
{
    int i,j;
    FILE *pf;
    pf = fopen(filename,"r+");
    fscanf(pf,"%d %d",rows,cols);
    if (*M==NULL)
    {
        AllocateDoubleMatrix(*rows,*cols,M);
    }
    for (i=0;i<*rows;i++)
        for (j=0;j<*cols;j++) fscanf(pf,"%lf",&M[0][i][j]);
    fclose(pf);
}

/************************************************/
// READ MATRIX AND SIZE FROM FILE POINTER, ROWS AND COLS ARE GIVEN IN THE FILE
void ReadDoubleMatrixAFP(FILE* fp, int* rows, int* cols, double*** M)
{
    int i,j;
    fscanf(fp,"%d %d",rows,cols);
    if (*M==NULL)
    {
        AllocateDoubleMatrix(*rows,*cols,M);
    }
    for (i=0;i<*rows;i++)
        for (j=0;j<*cols;j++) fscanf(fp,"%lf",&M[0][i][j]);
}

/************************************************/
void Eye(const int size, double*** I)
{
    int i;
    if (*I==NULL)
    {
        AllocateDoubleMatrix(size,size,I);
    }
    for (i=0;i<size;i++)
    {
        (*I)[i][i]=1.0;
    }
}

/************************************************/
void Ones(const int size, double** V)
{
    int i;
    if (*V==NULL)
    {
        AllocateDoubleVector(size,V);
    }
    for (i=0;i<size;i++)
    {
        (*V)[i]=1;
    }
}

/************************************************/
// begin index from 0
double* ExtractDoubleVector(const int begin, const int length, double* v)
{
    double* v1;
    int i;
    AllocateDoubleVector(length,&v1);
    for (i=0;i<length;i++)
    {
        v1[i]=v[i+begin];
    }
    return v1;
}
/************************************************/
void SparseMat2Mat(const int rows, const int cols, const int nz,
                   int* I, int* J, double* val, double*** M)
{
    int i;
    AllocateDoubleMatrix(rows,cols,M);

    for (i=0;i<nz;i++)
    {
        (*M)[I[i]][J[i]]=val[i];
    }
}

/************************************************/
// read matrix market(MM) matrix
// return 0 if success
//int ReadMMMatrix(const char* filename,int* rows,int* cols,double*** A)
//{
//    int i,nz;
//    int *I=NULL, *J=NULL;
//    double *val=NULL;
//    int status;
//    MM_typecode matcode;
//
//    FILE *pf=NULL;
//    pf = fopen(filename,"r+");
//    if (pf==NULL)
//    {
//        return -1;
//    }
//    status = mm_read_banner(pf,&matcode);
//    if (status != 0)
//    {
//        printf("Error: Could not process Matrix Market banner.\n");
//        return -1;
//    }
//    status=mm_read_mtx_crd_size(pf,rows,cols,&nz);
//    if (status != 0)
//    {
//        return -1;
//    }
//
//    AllocateIntVector(nz,&I);
//    AllocateIntVector(nz,&J);
//    AllocateDoubleVector(nz,&val);
//
//    for (i=0; i<nz; i++)
//    {
//        fscanf(pf, "%d %d %lf\n", &I[i], &J[i], &val[i]);
//        I[i]--;  /* adjust from 1-based to 0-based */
//        J[i]--;
//    }
//    fclose(pf);
//
//    SparseMat2Mat(*rows,*cols,nz,I,J,val,A);
//    FreeIntVector(I);
//    FreeIntVector(J);
//    FreeDoubleVector(val);
//    return 0;
//}
