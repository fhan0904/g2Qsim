/* kernel routine starts with keyword __global__ */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

// CUDA includes
#include <cuda.h>
#include <curand_kernel.h>

// Root includes
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

TFile *f;
TH1D *hFlush1D;
TH2D *hFlush2D, *hFlush2DCourse;

// timing tests                                                                       
float toddiff(struct timeval*, struct timeval*);
/********************************************************************/
float toddiff(struct timeval *tod2, struct timeval *tod1) {
  float fdt, fmudt;
  long long t1, t2, mut1, mut2;
  long long dt, mudt;
  t1 = tod1->tv_sec;
  mut1 = tod1->tv_usec;
  t2 = tod2->tv_sec;
  mut2 = tod2->tv_usec;
  dt = ( t2 - t1);
  mudt = ( mut2 - mut1);
  fdt = (float)dt;
  fmudt = (float)mudt;
  return fdt + 1.0e-6*fmudt;
}

//seed for curand_init
__device__ unsigned long long iseed = 1234;

__global__ void init_rand( curandState *state, unsigned long long offset) {

   int idx = blockIdx.x*256 + threadIdx.x;
   curand_init( iseed+offset, idx, 0, &state[idx]);
   //printf("curand_init, idx %i,  iseed %llu, offset %llu\n", idx, iseed, offset);
}

__global__ void make_rand(curandState *state, float *randArray) {

   int idx = blockIdx.x*256 + threadIdx.x;
   curandState localState = state[idx];
   randArray[idx] = curand_uniform( &localState);
   state[idx] = localState;
}

__global__ void make_randexp(curandState *state, float *randArray, float tau) {

   int idx = blockIdx.x*256 + threadIdx.x;
   curandState localState = state[idx];
   randArray[idx] = -tau * log( 1.0 -curand_uniform(&localState) );
   state[idx] = localState;
}

__global__ void make_randfill(curandState *state, float *randArray, int32_t *fillArray, int ne, int fill_buffer_max_length, int nfills ) {
  // single thread make complete fill with ne electrons

   const float tau = 6.4e4;          // ns
   const float omega_a = 1.438e-3;   // rad/ns
   float t, y, A, n;                 // mu-decay parameters
   float r, r_test;                  // mu decay rate
   const int GeVToADC = 2048./6.0;       // energy-ADC counts conversion
   const int nsPerTick = 16;

   int idx = blockIdx.x*256 + threadIdx.x;
   curandState localState = state[idx];

   if (idx>=nfills) return;
   
   // make noise
   //printf("pedestal %i\n", idx); 
   float noise = 0., pedestal = 0., sigma = 4.;
   for (int i = 0; i < fill_buffer_max_length; i++){
     noise = pedestal + sigma * curand_normal(&localState); // random from Gaussian using uniform random number 0->1
     *(fillArray + fill_buffer_max_length*idx + i ) = (int32_t)noise;  // fill buffer
     //if (i == 1) printf("idx = %i, i = %i, noise %f, *(fillArray + fill_buffer_max_length*idx + i ) = %i\n", 
     //			idx, i, noise, *(fillArray + fill_buffer_max_length*idx + i ));
   }

   // add hits
   int ADC; 
   float tick;
   for (int i = 0; i < ne; i++){

      t = -tau * log( 1.0 - curand_uniform(&localState) ); // random from exp(-t/tau) using uniform random number 0->1
      tick = t/nsPerTick; 
      y = curand_uniform(&localState);
      A = (2.0*y - 1)/(3.0 - 2.0*y);
      n = y*y*(3.0 - 2.0*y);
 
      r_test = n*(1.0-A*cos(omega_a*t))*0.5;
      r = curand_uniform(&localState);

      if ( r >= r_test )
        {
          continue;
        }

      const float Elab_max = 3.1; // GeV
      const float Pi = 3.1415926;
      double theta = 0; // decay angle
      theta = Pi*curand_uniform(&localState);
      double Elab = 0.5 *Elab_max * y * ( 1.0 + cos(theta)); 
  
      ADC = (int)(GeVToADC*Elab);


      // distribute pulse over several bins
      int itick = (int)tick;
      float rtick = tick - itick;
      float width = 4.0/nsPerTick; // pulse sigma in bin width units
      float gsum = 0;
      for (int k=-2; k<=2; k++) gsum += exp(-0.5*(k+rtick)*(k+rtick)/width/width);

      //printf("HIT %i, idx %i\n",i,idx);
      for (int k=-2; k<=2; k++)
	{
	  int kk = k + itick;
	  if ( kk < 0 || kk >= fill_buffer_max_length ) continue;
          int ADCfrac = ADC*exp(-0.5*(k+rtick)*(k+rtick)/width/width)/gsum;
	  //printf("idx = %i, ne = %i, hit %i, tick = %f, itick = %i, rtick = %f, ADCfrac %i, ADC %i\n", 
	  //	 idx, ne, i, tick, itick, rtick, ADCfrac, ADC);
	  if ( ADCfrac < -2048 ) ADCfrac = -2048;
	  if ( ADCfrac > 2048 ) ADCfrac = 2048;
	  *(fillArray + fill_buffer_max_length*idx + kk ) += ADCfrac;  // fill buffer
	}

      randArray[idx] = t;     // debugging only
      state[idx] = localState;
   }
}

__global__ void make_fillsum( int32_t *fillArray, float *fillSumArray, int nfills, int fill_buffer_max_length ) {

  int idx = blockIdx.x*256 + threadIdx.x;
  
  if (idx < fill_buffer_max_length) {
    
    // initialize flush to zero
    *(fillSumArray + idx) = 0;
    for (int i = 0; i < nfills; i++){
      *(fillSumArray + idx) += *(fillArray + fill_buffer_max_length*i + idx );  // fill buffer
      //if (idx == 1) printf("idx = %i, i = %i, *(fillArray + fill_buffer_max_length*i + idx) = %i, *(fillSumArray + idx) = %f\n", 
      //			   idx, i, *(fillArray + fill_buffer_max_length*i + idx), *(fillSumArray + idx));
      
    }
  }
}

int main(int argc, char * argv[]){
 
  // define nthreads, nblocks for GPU
  // define fill length, clock tick for simulation
  cudaError err;
  const int nsPerFill = 524288, nsPerTick = 16; 
  int fill_buffer_max_length = nsPerFill / nsPerTick;                     
  int nthreads = 256, nblocks1 = 1, nblocks2;
  int nfills, nflushes;
  curandState *d_state;
  float *h_randArray, *d_randArray; 
  int32_t *h_fillArray, *d_fillArray;
  float *h_fillSumArray, *d_fillSumArray; 


  // define run, flush, fill structure
  if (argc == 1) {
    nfills = 256;
    nflushes = 1;
  } else {
    nfills = atoi(argv[1]);
    nflushes = atoi(argv[2]);
  }
  printf("nfills per flush %d, nflushes per run %d\n", nfills, nflushes);

  // define grid structure
  nblocks1 = nfills / nthreads + 1;
  nblocks2 = ( fill_buffer_max_length + nthreads - 1 )/ nthreads;
  printf("per flush grid: nthreads %i, nblocks %i nthreads*nblocks %i\n", nthreads, nblocks1, nthreads*nblocks1 );
  printf("per bin grid: nthreads %i, nblocks %i nthreads*nblocks %i\n", nthreads, nblocks2, nthreads*nblocks2 );

  // histogram binning
  printf("ns per fill %i, ns per bin %i, number of bins %d\n", nsPerFill, nsPerTick, fill_buffer_max_length);               
  hFlush1D = new TH1D("hFlush1D", "hFlush1D", fill_buffer_max_length, 0.0, fill_buffer_max_length );
  hFlush2D = new TH2D("hFlush2D", "hFlush2D", fill_buffer_max_length, 0.0, fill_buffer_max_length, 512, -512, 511 );
  hFlush2DCourse = new TH2D("hFlush2DCourse", "hFlush2DCourse", fill_buffer_max_length, 0.0, fill_buffer_max_length, 250, -1000, 24000 );

  // on average 700 good e's per fill, 2900 e's per fill
  int ne = 2900;
  // divide by 24 for per calorimeter rate.
  ne = ne/24;
  // test
  //ne = 0;

  FILE *fp;
  fp = fopen( "fillSumArray.dat", "w" ); // Open file for writing
 
  // set device number
  int num_devices, device;
  cudaGetDeviceCount(&num_devices);
  if (num_devices > 1) {
     for (device = 0; device < num_devices; device++) {
  	 cudaDeviceProp properties;
	 cudaGetDeviceProperties(&properties, device);
         printf("device %d properties.multiProcessorCount %d\n", 
	         device, properties.multiProcessorCount);
     }			      
  }	      
  cudaSetDevice(0);

  // get some cuda device properties
  cudaDeviceProp prop;
  cudaGetDeviceProperties(&prop, 0);
  printf("Device Number: %d\n", 0);
  printf("Device name: %s\n", prop.name);
  printf("Memory Clock Rate (KHz): %d\n",
  		 prop.memoryClockRate);
  printf("Memory Bus Width (bits): %d\n",
                  prop.memoryBusWidth);


  cudaEvent_t start, stop;
  float elapsedTime;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start, 0);

  h_randArray = (float *)malloc(nfills*sizeof(float));
  cudaMalloc( (void **)&d_randArray, nfills*sizeof(float));
  cudaMalloc( (void **)&d_state, nfills*sizeof(curandState));
  err = cudaThreadSynchronize();
  if ( cudaSuccess != err ) {
    printf("Cuda error in file '%s' in line %i : %s.\n",
             __FILE__, __LINE__, cudaGetErrorString( err) );
  }

  h_fillArray = (int32_t *)malloc(nfills*fill_buffer_max_length*sizeof(int32_t));
  cudaMalloc( (void **)&d_fillArray, nfills*fill_buffer_max_length*sizeof(int32_t));
  err = cudaThreadSynchronize();
  if ( cudaSuccess != err ) {
    printf("Cuda error in file '%s' in line %i : %s.\n",
             __FILE__, __LINE__, cudaGetErrorString( err) );
  }
  h_fillSumArray = (float *)malloc(fill_buffer_max_length*sizeof(float));
  cudaMalloc( (void **)&d_fillSumArray, fill_buffer_max_length*sizeof(float));

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf(" ::: kernel malloc / cudaMalloc time %f ms\n",elapsedTime);

  cudaEventRecord(start, 0);

  unsigned long long offset = 0;
  init_rand<<<nblocks1,nthreads>>>( d_state, offset);

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf(" ::: kernel init_rand time %f ms\n",elapsedTime);

  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);

  for (int j = 0; j < nflushes; j++){

    printf("flush %i\n", j);
    cudaEventRecord(start, 0);

    memset( h_fillArray, 0, nfills*fill_buffer_max_length * sizeof(int32_t) );
    cudaMemcpy( d_fillArray, h_fillArray, nfills*fill_buffer_max_length*sizeof(int32_t), cudaMemcpyHostToDevice);
    cudaThreadSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    if (j == 0) printf(" ::: kernel initialize fillArray %f ms\n",elapsedTime);

    cudaEventRecord(start, 0);
    
    offset++;
    init_rand<<<nblocks1,nthreads>>>( d_state, offset);
    //make_rand<<<nblocks1,nthreads>>>( d_state, d_randArray);
    //make_randexp<<<nblocks1,nthreads>>>( d_state, d_randArray, tau);
    make_randfill<<<nblocks1,nthreads>>>( d_state, d_randArray, d_fillArray, ne, fill_buffer_max_length, nfills);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    if (j == 0) printf(" ::: kernel make_randfill time %f ms\n",elapsedTime);

    cudaEventRecord(start, 0);

    make_fillsum<<<nblocks2,nthreads>>>( d_fillArray, d_fillSumArray, nfills, fill_buffer_max_length);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    if (j == 0) printf(" ::: kernel make_fillsum time %f ms\n",elapsedTime);

    cudaEventRecord(start, 0);

   /*  
    cudaMemcpy( h_randArray, d_randArray, nfills*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nfills; i++){
        if ( *(h_randArray+i) > 0) printf(" randArray[i] = %f[%i]\n", *(h_randArray+i), i );
    }
   */

   /*
    cudaMemcpy( h_fillArray, d_fillArray, nfills*fill_buffer_max_length*sizeof(int32_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nfills*fill_buffer_max_length; i++){
        if ( *(h_fillArray+i) > 0) printf(" fillArray[i] = %u[%i]\n", *(h_fillArray+i), i );
    }
   */

    cudaMemcpy( h_fillSumArray, d_fillSumArray, fill_buffer_max_length*sizeof(float), cudaMemcpyDeviceToHost);
    for (int i = 0; i < fill_buffer_max_length; i++){
      //if ( *(h_fillSumArray+i) > 0) printf(" fillSumArray[i] = %f[%i]\n", *(h_fillSumArray+i), i );
      hFlush1D->Fill( i+1, *(h_fillSumArray+i));
      hFlush2D->Fill( i+1, *(h_fillSumArray+i));
      hFlush2DCourse->Fill( i+1, *(h_fillSumArray+i));
      //fprintf(fp, " %i %f\n", i+1, *(h_fillSumArray+i) );
      
    }
    
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    if (j == 0) printf(" ::: kernel cuda memcpy time %f ms\n",elapsedTime);
 

    /*
    for (int ib = 0; ib < nblocks1; ib++){
        for (int it = 0; it < nthreads; it++){
        }
    }
    */

  }

  gettimeofday(&end_time, NULL);
  printf("elapsed processing time, dt %f secs\n", toddiff(&end_time, &start_time));

  f = new TFile("test.root","recreate");
  f->WriteObject( hFlush1D, "hFlush1D");
  f->WriteObject( hFlush2D, "hFlush2D");
  f->WriteObject( hFlush2DCourse, "hFlush2DCourse");
  f->Close();


  return 0;
}



/*
    for (int i = 0; i < 32; i++){
        printf(" i = %i, input fillArray copy in = %u %u\n", i, h_fillArray[i], *(h_fillArray+i) );
    }
    cudaMemcpy( h_fillArray, d_fillArray, nfills*fill_buffer_max_length*sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaThreadSynchronize();
    for (int i = 0; i < 32; i++){
        printf(" i = %i, input fillArray copy out = %u %u\n", i, h_fillArray[i], *(h_fillArray+i) );
    }
*/


/* 2D array allocation on host/device
                                                                     
  // allocate host fill buffer array of nfills*fill_buffer_max_length
  printf("%s(%d): allocate array of fill buffer, size %d  \n", __func__, __LINE__, nfills );
  int32_t **fill_buffer;
  fill_buffer = (int32_t**) malloc(  nfills*sizeof(int32_t*));                                                                    
  for (int i = 0; i < nblocks1*nthreads; i++){
    fill_buffer[i] = (int32_t*) malloc( fill_buffer_max_length );
    if ( ! fill_buffer[i] )                                                                                                                  
      {                                                                                                                                     
        printf("failed allocation of fill buffer on host\n");
        return;
      }
  }

  // allocate device fill buffer array of nfills*fill_buffer_max_length
  int32_t *devPtr;
  size_t pitch;
  cudaMallocPitch( &devPtr, &pitch,  fill_buffer_max_length * sizeof(int32_t), nfills);

  */

