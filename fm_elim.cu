#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>




__device__ double epsilon = 1e-6; 

__global__ void maxvectPos(const double *dev_A, const double *dev_b,
                           int n, double *outP, double *outN, int *impossible) ;

__global__ void copy_without_column(const double* A,const double* B,
                                    const int* idx, int count, int m, int k, 
                                    double* out_P
                                  );

__global__ void normalize(const double* A,
                          const double* b,
                          const int* idx, int count, int m, int k,
                          double * out_P
                        ) ;

__global__ void compute_diff( 
        const double *pNorm, const double *nNorm,
        double *out_A, double *out_b,
        int numP,int numN,
        int m,
        int TP,int TN);
__global__ void addRows(double * normZ, double* dev_A,double* dev_b,int m,int nZ);

__global__ void classify(const double*  A,
                             int n, int m,
                             int*  P_count, int* N_count, int*  Z_count) ;

__global__ void computeIndex(const double*  A,
                             int n, int m,int k,
                             int*  P_idx, int*  N_idx, int*  Z_idx,
                             int*  P_count, int* N_count, int*  Z_count);

__global__ void calcFinalPN(int hP, int hN,
                            int TP,int TN, 
                            double* pNorm,double* nNorm,
                            double *outP, double *outN, int *impossible); // i want to work only blocks and reduce the final result

void printSystem(double *dev_A, double *dev_b, int n,int m);
                                
void printNormalized(double *normP, int n,int m,int k=0);

int numBlocksFor(int n);



int TP;
int TN;
int NumThPerBlock;
int cap;

int main(int argc,char** argv){

  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
    int NumBlocks;
    
    if(argc==2){
      TP=32;
      TN=32;
      cap=0;
    }else if(argc==5){
      TP=atoi(argv[2]);
      TN=atoi(argv[3]);
      cap=atoi(argv[4]);
    }else{
        printf("prog input TP TN \nGiven argc=%d",argc);
        exit(1);
    }
    NumThPerBlock=TP*TN;

    FILE* F = fopen(argv[1], "r");
    if (!F) { perror("fopen"); return 1; }

    int n,m;
    fscanf(F,"%d %d",&n,&m);
    
    double* A = (double*)malloc(sizeof(double) * (size_t)n * m);
    double* b = (double*)malloc(sizeof(double) * (size_t)n);

    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            fscanf(F, "%lf", &A[i * m + j]);
        }
        fscanf(F, "%lf", &b[i]);
    }
    fclose(F);

    //printf("Read all system done\n");


    cudaEventRecord(start);
    double *dev_A,*dev_b;
    cudaMalloc((void**)&dev_A, sizeof(double)*n*m);
    cudaMalloc((void**)&dev_b, sizeof(double)*n);
    cudaMemcpy(dev_A, A, n*m* sizeof(double),cudaMemcpyHostToDevice);
    cudaMemcpy(dev_b, b, n * sizeof(double),cudaMemcpyHostToDevice);


    int *P_idx,*N_idx,*Z_idx,             // indexing the types of disequations
        *P_count,*N_count,*Z_count,       // count how many for each kind of disequation for each possible index (1..m).
        *P_count_d,*N_count_d,*Z_count_d; // count how many for each kind for a single index chosed
    int hP,hN,hZ;                         // counting how many disequation there are for that coefficient 
    double* pNorm,*nNorm,*zNorm;          // normalized and compacted result of possible kinds before elimination

    //printf("Allocated memeory\n");
    while(m>1){

        NumBlocks=numBlocksFor(n*m);
        //printSystem(dev_A,dev_b,n,m);
        // compute index (let's say we sorted them)
        cudaMalloc((void**)&P_idx, sizeof(int)*n);
        cudaMalloc((void**)&N_idx, sizeof(int)*n);
        cudaMalloc((void**)&Z_idx, sizeof(int)*n);
        cudaMalloc((void**)&P_count, sizeof(int)*m);
        cudaMalloc((void**)&N_count, sizeof(int)*m);
        cudaMalloc((void**)&Z_count, sizeof(int)*m);
        cudaMemset(P_count, 0, sizeof(int)*m);
        cudaMemset(N_count, 0, sizeof(int)*m);
        cudaMemset(Z_count, 0, sizeof(int)*m);
        
    
        classify<<<NumBlocks,NumThPerBlock>>>(dev_A, n, m,
                                            P_count, N_count, Z_count);
        cudaDeviceSynchronize(); 

        int *hPc,*hNc,*hZc;
        hPc=(int*)malloc(sizeof(int)*m);
        hNc=(int*)malloc(sizeof(int)*m);
        hZc=(int*)malloc(sizeof(int)*m);


        cudaMemcpy(hPc, P_count, m*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(hNc, N_count, m*sizeof(int),cudaMemcpyDeviceToHost);
        cudaMemcpy(hZc, Z_count, m*sizeof(int),cudaMemcpyDeviceToHost); 


        int idx=0;
        int minNumEq=hPc[0]*hNc[0]+hZc[0];

        for(int i=1;i<m;i++){
          int curr=hPc[i]*hNc[i]+hZc[i];
          if(curr<minNumEq){
            minNumEq=curr;
            idx=i;
          }
        }
        hP = hPc[idx];
        hN = hNc[idx];
        hZ = hZc[idx];

        cudaMalloc(&P_count_d, sizeof(int));
        cudaMalloc(&N_count_d, sizeof(int));
        cudaMalloc(&Z_count_d, sizeof(int));
        cudaMemset(P_count_d, 0, sizeof(int));
        cudaMemset(N_count_d, 0, sizeof(int));
        cudaMemset(Z_count_d, 0, sizeof(int));
                          





        computeIndex<<<NumBlocks,NumThPerBlock>>>(dev_A, n, m,idx,
                                            P_idx,N_idx,Z_idx,
                                            P_count_d, N_count_d, Z_count_d);

        cudaDeviceSynchronize();

        cudaFree(P_count_d);
        cudaFree(N_count_d);
        cudaFree(Z_count_d);
        cudaFree(P_count); cudaFree(N_count); cudaFree(Z_count);

        free(hPc);free(hNc);free(hZc);

        // nomalize 

        cudaMalloc((void**)&pNorm, sizeof(double)*hP*m);  
        cudaMalloc((void**)&nNorm, sizeof(double)*hN*m);
        cudaMalloc((void**)&zNorm, sizeof(double)*hZ*m);
        



        n = hP * hN + hZ;   // new number of equations
        //printf("Step %d: index %d has P=%d N=%d Z=%d -> new equations n=%d\n",m,idx,hP,hN,hZ,n);
        
        normalize<<<NumBlocks,NumThPerBlock>>>(dev_A, dev_b, P_idx, hP, m, idx, pNorm);
        normalize<<<NumBlocks,NumThPerBlock>>>(dev_A, dev_b, N_idx, hN, m, idx, nNorm);
        copy_without_column<<<NumBlocks,NumThPerBlock>>>(dev_A, dev_b, Z_idx, hZ, m, idx, zNorm);

        cudaFree(dev_A);
        cudaFree(dev_b);

        if(m>2){
          cudaMalloc((void**)&dev_A,sizeof(double)*n*(m-1));
          cudaMalloc((void**)&dev_b,sizeof(double)*n);

          // pairwais difference between the functions
          int numP=(hP + TP - 1) /TP;
          int numN=(hN + TN - 1) / TN;
          int NumBlocksDiff=numP*numN;
          int BlockSizeDiff=TP*TN;
          int sharedMemSize=(TP+TN)*m*sizeof(double);

          compute_diff<<<NumBlocksDiff,BlockSizeDiff,sharedMemSize>>>(pNorm,nNorm,dev_A,dev_b,hP,hN,m,TP,TN);
          
          
          
          // add the last z rows      
          addRows<<<NumBlocks,NumThPerBlock>>>(zNorm, dev_A+(m-1)*(hP*hN),dev_b+(hP*hN),m,hZ);

          cudaFree(P_idx); cudaFree(N_idx); cudaFree(Z_idx);
          cudaFree(pNorm); cudaFree(nNorm); cudaFree(zNorm);

          if(n==0){
            break;
          }
        }
        m--;
    }    

    if(n==0){
      //printf("Feasible\n");
      //return 0;  /removed to test timing
    }    
  

    int numBlocksP = (hP + TP - 1) / TP;
    int numBlocksN = (hN + TN - 1) / TN;  
    int gridPN     = numBlocksP * numBlocksN;
    int blockPN    = TP * TN;
    size_t shmem =(TP+TN+blockPN)* 2  * sizeof(double);
    


    double *bestPos_d, *bestNeg_d;

    cudaMalloc(&bestPos_d, gridPN * sizeof(double));
    cudaMalloc(&bestNeg_d, gridPN * sizeof(double));
    
    int *impossible_d;
    cudaMalloc(&impossible_d, sizeof(int));
    cudaMemset(impossible_d, 0, sizeof(int));



    calcFinalPN<<<gridPN, blockPN, shmem>>>(hP, hN,
        TP,TN, 
         pNorm, nNorm,
        bestPos_d, bestNeg_d,impossible_d); 

    cudaDeviceSynchronize();

    double* hBestPos=(double*)malloc(gridPN*sizeof(double));
    double* hBestNeg=(double*)malloc(gridPN*sizeof(double));

    int hImpossible;
    cudaMemcpy(hBestPos, bestPos_d, gridPN*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(hBestNeg, bestNeg_d, gridPN*sizeof(double), cudaMemcpyDeviceToHost);
    cudaMemcpy(&hImpossible, impossible_d, sizeof(int), cudaMemcpyDeviceToHost);



    if (hImpossible > 0) {
        //printf("Not feasible (impossible constraint)\n");
        //return 0;//  removed for check timie
    }

    double bestPos = INFINITY;
    double bestNeg = -INFINITY;
    for (int i=0; i<gridPN; i++) {
        if (hBestPos[i] < bestPos) bestPos = hBestPos[i];
        if (hBestNeg[i] > bestNeg) bestNeg = hBestNeg[i];
    }

    if(bestPos <= bestNeg){
      //printf("Feasible\n");
    } else {
      //printf("Not feasible\n");
    }

    cudaFree(bestPos_d);
    cudaFree(bestNeg_d);
    cudaFree(impossible_d);

    cudaEventRecord(stop);
    float milliseconds = 0;
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&milliseconds, start, stop);

    printf("%f\n",milliseconds);


}





__global__ void classify(const double*  A,
                             int n, int m,
                             int*  P_count, int* N_count, int*  Z_count) {
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n*m; t += blockDim.x * gridDim.x) {
    int k=t%m;

    double value = A[t];
    if (value > epsilon) {
      atomicAdd(&P_count[k], 1);
    } else if (value < -epsilon) {
      atomicAdd(&N_count[k], 1);
    } else {
      atomicAdd(&Z_count[k], 1);
    }
  }
}

__global__ void computeIndex(const double*  A,
                             int n, int m, int k,
                             int*  P_idx, int*  N_idx, int*  Z_idx,
                             int*  P_count, int* N_count, int*  Z_count) {
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < n; t += blockDim.x * gridDim.x) {
    
    double value = A[t*m + k];
    if (value > epsilon) {
      int pos = atomicAdd(P_count, 1);
      P_idx[pos]=t;
    } else if (value < -epsilon) {
      int pos = atomicAdd(N_count, 1);
      N_idx[pos]=t;
    } else {
      int pos = atomicAdd(Z_count, 1);
      Z_idx[pos]=t;
    }
  }
}


__global__ void normalize(const double* A,
                          const double* b,
                          const int* idx, int count, int m, int k,
                          double * out_P) {
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < count; t += blockDim.x * gridDim.x) {
    int workingIdx = idx[t];
    double coef = 1.0 / A[workingIdx * m + k];

    int outj = 0;
    double* outRow = out_P + t * m;            // <-- stride m
    const double* copyRow = A + workingIdx * m;
    for (int j = 0; j < m; j++) {
      if (j == k) continue;
      outRow[outj++] = copyRow[j] * coef;      // m-1 coeffs
    }
    outRow[outj] = b[workingIdx] * coef;       // RHS at column m-1
  }
}

__global__ void copy_without_column(const double* A, const double* B,
                                    const int* idx, int count, int m, int k,
                                    double* out_P) {
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < count; t += blockDim.x * gridDim.x) {
    int workingIdx = idx[t];
    int outj = 0;
    double* outRow = out_P + t * m;
    const double* copyRow = A + workingIdx * m;
    for (int j = 0; j < m; j++) {
      if (j == k) continue;
      outRow[outj++] = copyRow[j];             // m-1 coeffs
    }
    outRow[outj] = B[workingIdx];              // RHS
  }
}

// idea: using shared memory to compute it:
//   we run the algorithm per blocks
//   each block compute a TN*TP cell of possible moltiplications
//      we load in shared memory the block informations
//      use them later
//     compute the block (p,n) of size TP * TN
__global__ void compute_diff( 
        const double *pNorm, const double *nNorm,
        double *out_A, double *out_b,
        int hP,int hN,
        int m,
        int TP,int TN){
    
    extern __shared__ double smem[];
    double* sP=smem;
    double* sN=smem+TP*m;


    int numBlocksN=(hN + TN - 1) / TN;

    int idx_p= blockIdx.x/numBlocksN;
    int idx_n= blockIdx.x%numBlocksN;

    int p_base=idx_p*TP;
    int n_base=idx_n*TN;

    

    for (int i = threadIdx.x; i < TP*m; i += blockDim.x) {
      int r = i / m, c = i % m;
      int eq = p_base + r;
      if (eq < hP) 
        sP[i] = pNorm[eq * m + c];
    }

    for (int i = threadIdx.x; i < TN*m; i += blockDim.x) {
      int r = i / m, c = i % m;
      int eq = n_base + r;
      if (eq < hN) 
        sN[i] = nNorm[eq * m + c];
    }

    __syncthreads();

    
    int p = threadIdx.x / TN;
    int n = threadIdx.x % TN;
    if (p >= hP || n >= hN) return;

    
    int baseP = p_base + p;
    int baseN = n_base + n;

    if (baseP >= hP || baseN >= hN) return;

    int baseOut = baseP * hN+baseN;
        __syncthreads();
    

  for (int j = 0; j < m - 1; ++j)
    out_A[baseOut * (m - 1) + j] = sP[p*m + j] - sN[n*m + j];
  out_b[baseOut] = sP[p*m + (m - 1)] - sN[n*m + (m - 1)];

}





__global__ void addRows(double * normZ, double* dev_A,double* dev_b,int m,int nZ){
  for (int t = blockIdx.x * blockDim.x + threadIdx.x; t < nZ; t += blockDim.x * gridDim.x) {
    
    int offRes=t*(m-1);
    int offZ=t*(m);
    for(int j=0;j<m-1;j++){
      dev_A[offRes+j]=normZ[offZ+j];
    }
    dev_b[t]=normZ[offZ+m-1];
  }   
} 


// take the minimum between the positive -> x<5 is more strict than x<7
// the maximum between the negative  -> -x<5, -x<7 --norm-> x>-5, x>-7, is better x>-5
__global__ void maxvectPos(const double *dev_A, const double *dev_b,
                           int n, double *outP, double *outN, int *impossible) {
  extern __shared__ double sdata[];
  double *posCalc = sdata;
  double *negCalc = sdata + blockDim.x;

  int tid = threadIdx.x;
  double minValPos = INFINITY;
  double maxValNeg = -INFINITY;

  for(int i = blockIdx.x*blockDim.x + tid; i < n; i += blockDim.x*gridDim.x){
    if(dev_A[i]>-epsilon && dev_A[i]<epsilon){
      if(dev_b[i]<-epsilon)
        atomicAdd(impossible,1);
      continue;
    }
    double coef = dev_b[i]/dev_A[i];
    if(dev_A[i] > 0){
      if(minValPos > coef) minValPos = coef;
    } else {
      if(maxValNeg < coef) maxValNeg = coef;
    }
  }


  posCalc[tid] = minValPos;
  negCalc[tid] = maxValNeg;
  __syncthreads();

  if (atomicAdd(impossible, 0) > 0) return;
  
  // block-level reduction
  for (unsigned int s = blockDim.x/2; s > 0; s >>= 1) {
    if (tid < s) {
      if(posCalc[tid] > posCalc[tid+s]) posCalc[tid] = posCalc[tid+s];
      if(negCalc[tid] < negCalc[tid+s]) negCalc[tid] = negCalc[tid+s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    outP[blockIdx.x] = posCalc[0];
    outN[blockIdx.x] = negCalc[0];
  }
}



__global__ void calcFinalPN(int hP, int hN,int TP,int TN, double* pNorm,double* nNorm,double *outP, double *outN, int *impossible){ // i want to work only blocks and reduce the final result
  extern __shared__ double smem[];  
    double* sP=smem;
    double* sN=sP +TP*2;
    double* posCalc=sN+TN*2;
    double* negCalc=posCalc+blockDim.x;


    int numBlocksN=(hN + TN - 1) / TN;

    int idx_p= blockIdx.x/numBlocksN;
    int idx_n= blockIdx.x%numBlocksN;

    int p_base=idx_p*TP;
    int n_base=idx_n*TN;

    

    for (int i = threadIdx.x; i < TP*2; i += blockDim.x) {
      int r = i / 2, c = i % 2;
      int eq = p_base + r;
      if (eq < hP) 
        sP[i] = pNorm[eq * 2 + c];
    }

    for (int i = threadIdx.x; i < TN*2; i += blockDim.x) {
      int r = i / 2, c = i % 2;
      int eq = n_base + r;
      if (eq < hN) 
        sN[i] = nNorm[eq * 2 + c];
    }

    __syncthreads();

    
    int p = threadIdx.x / TN;
    int n = threadIdx.x % TN;
    if (p >= hP || n >= hN) return;

    
    int baseP = p_base + p;
    int baseN = n_base + n;

    if (baseP >= hP || baseN >= hN) return;

    __syncthreads();



    double A=sP[p*2 ] - sN[n*2 ];
    double b=sP[p*2 + 1] - sN[n*2 + 1];


    int tid = threadIdx.x;
    double minValPos = INFINITY;
    double maxValNeg = -INFINITY;

    if(A>-epsilon && A<epsilon){
      if(b<-epsilon)
        atomicAdd(impossible,1);
    }else{
      double coef = b/A;
      if(A > 0){
        if(minValPos > coef) minValPos = coef;
      } else {
        if(maxValNeg < coef) maxValNeg = coef;
      }
    }
   


  posCalc[tid] = minValPos;
  negCalc[tid] = maxValNeg;

  __syncthreads();

  if (atomicAdd(impossible, 0) > 0) return;
  
  for (unsigned int s = blockDim.x/2; s > 0; s /= 2) {
    if (tid < s) {
      if(posCalc[tid] > posCalc[tid+s]) posCalc[tid] = posCalc[tid+s];
      if(negCalc[tid] < negCalc[tid+s]) negCalc[tid] = negCalc[tid+s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    outP[blockIdx.x] = posCalc[0];
    outN[blockIdx.x] = negCalc[0];
  }
}



void printNormalized(double *normP, int n,int m,int k){
  double *A=(double*)malloc(sizeof(double)*m*n);
  
  
  cudaMemcpy(A, normP, n*m* sizeof(double),cudaMemcpyDeviceToHost);

  for(int i=0;i<n;i++){
    int ia=0;
    for(int j=0;j<m;j++){
      if(j==k)printf("xxxxxx ");
      else{
      printf("%lf ",A[i*m+ia]);
      ia++;
      }
    }
    printf("| %lf\n",A[i*m+m-1]);
  }
  free(A);

}


void printSystem(double *dev_A, double *dev_b, int n,int m){
  printf("StartPrinting\n");
  double *A=(double*)malloc(sizeof(double)*m*n);
  double *b=(double*)malloc(sizeof(double)*n);
  
  
  cudaMemcpy(A, dev_A, n*m* sizeof(double),cudaMemcpyDeviceToHost);
  cudaMemcpy(b, dev_b, n *sizeof(double),cudaMemcpyDeviceToHost);
  


  for(int i=0;i<n;i++){
    for(int j=0;j<m;j++){
      printf("%lf ",A[i*m+j]);
    }
    printf("| %lf\n",b[i]);
  }
  free(A);
  free(b);

}



int numBlocksFor(int work){
  int smCount = 0; cudaDeviceGetAttribute(&smCount, cudaDevAttrMultiProcessorCount, 0);
  int blocks=(work + NumThPerBlock - 1) / NumThPerBlock;
  int localCap = cap> 0 ? cap  : smCount * 32; 
  if (blocks > localCap) blocks = localCap;
  return blocks;
}