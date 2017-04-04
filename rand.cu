/* kernel routine starts with keyword __global__ */

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <time.h>
#include <sys/time.h>
#include <math.h>

// CUDA includes
#include <cuda.h>
#include <curand.h>
#include <curand_kernel.h>

// Root includes
#include <TFile.h>
#include <TH1.h>
#include <TH2.h>

//#define NXSEG 1
//#define NYSEG 1
//#define NSEG 1
#define NXSEG 9
#define NYSEG 6
#define NSEG 54

TFile *f;
TH1D *hHits1D;
TH1D *hFlush1D[NSEG];
TH2D *hFlush2D[NSEG];
TH2D *hFlush2DCoarse[NSEG];

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

__global__ void init_rand( curandState *state, unsigned long long offset, unsigned long long seed) {
  /* this GPU kernel function is used to initialize the random states */
   int idx = blockIdx.x*256 + threadIdx.x;
   /* Each thread gets same seed, a different sequence number, and no offset */
   curand_init( seed, idx, 0, &state[idx]);
   //printf("curand_init, idx %i,  iseed %llu, offset %llu\n", idx, iseed, offset);
}

__global__ void zero_int_array( int32_t *array, int length) {

   int idx = blockIdx.x*256 + threadIdx.x;
   if (idx >= length) return;

    *(array + idx) = 0;
}

__global__ void zero_float_array( float *array, int length) {

   int idx = blockIdx.x*256 + threadIdx.x;
   if (idx >= length) return;

    *(array + idx) = 0.0;
}

__global__ void make_rand( curandState *state, float *randArray) {

   int idx = blockIdx.x*256 + threadIdx.x;
   curandState localState = state[idx];
   randArray[idx] = curand_uniform( &localState);
   state[idx] = localState;
}

__global__ void make_randexp( curandState *state, float *randArray, float tau) {

   int idx = blockIdx.x*256 + threadIdx.x;
   curandState localState = state[idx];
   randArray[idx] = -tau * log( 1.0 -curand_uniform(&localState) );
   state[idx] = localState;
}

__global__ void make_randfill( curandState *state, int32_t *hitArray, int32_t *fillArray, int ne, int fill_buffer_max_length, int nfills, bool fillnoise) {
  // single thread make complete fill with ne electrons

   const float tau = 6.4e4;          // ns
   const float omega_a = 1.438e-3;   // rad/ns
   const float magicgamma = 29.3; // gamma factor for magic momentum 
   const int GeVToADC = 2048./6.2;   // energy-ADC counts conversion
   const int nsPerTick = 16;         // Q-method histogram bin size
   const float Elab_max = 3.1; // GeV, maximum positron lab energy
   const float Pi = 3.1415926; // Pi
   const float cyclotronperiod =149.0/nsPerTick; // cyclotron period ns -> ticks
   const float anomalousperiod = 4370./nsPerTick; // omega_c-omega_s period ns -> ticks
   const int nxseg = NXSEG, nyseg = NYSEG; // calo segmentation
   const int nsegs = NSEG; // calo segmentation

   // parameters for positron drift time calculation
   float p0 =    -0.255134;
   float p1 =      65.3034;
   float p2 =     -705.492;
   float p3 =      5267.21;
   float p4 =     -23986.5;
   float p5 =      68348.1;
   float p6 =      -121761;
   float p7 =       131393;
   float p8 =       -78343;
   float p9 =      19774.1;

   // variable for mu-decay
   float t, y, A, n;                 // mu-decay parameters
   float r, r_test;                  // mu decay rate

   // thread index
   int idx = blockIdx.x*256 + threadIdx.x;

   if (idx < nfills) {

     // state index for random number generator
     curandState localState = state[idx];
     
     // make noise
     if (fillnoise) {
       float noise = 0., pedestal = 0., sigma = 4.;
       for (int i = 0; i < nsegs*fill_buffer_max_length; i++){
	 noise = pedestal + sigma * curand_normal(&localState); // random from Gaussian using uniform random number 0->1
	 *(fillArray + fill_buffer_max_length*idx + i ) = (int32_t) noise;  // fill buffer
       }
     }
     
     int nhit = 0; // good hit counter
     float theta = 0; // decay angle
     
     // paramters for positron x,y, time
     float xrand, yrand, xmax; // x,y coordinate random numbers and endpoint of hit distribution across calo x-coordinate
     
     // paramters for positron drift time
     float ylab, phase, drifttime; // uses fit to MC data for omega_a phase versus positron lab energy 
     
     // arrays for storing the hit information before time-ordering
     int ADC, ADCstore[1000], iold[1000];
     float tick, xcoord, ycoord, tickstore[1000], xcoordstore[1000], ycoordstore[1000];
     
     // find hit times, energies, x-coordinate, y-coordinate
     for (int i = 0; i < ne; i++){
       
       // get muon decay time     
       t = -tau * log( 1.0 - curand_uniform(&localState) ); // random from exp(-t/tau) using uniform random number 0->1
       tick = t/nsPerTick;
       
       // get positron energy
       y = curand_uniform(&localState);
       A = (2.0*y - 1)/(3.0 - 2.0*y);
       n = y*y*(3.0 - 2.0*y);
       r_test = n*(1.0-A*cos(omega_a*t))*0.5;
       r = curand_uniform(&localState);  
       if ( r >= r_test ) continue;
       theta = Pi*curand_uniform(&localState);
       float Elab = 0.5 *Elab_max * y * ( 1.0 + cos(theta)); 
       
       // total ADC samples at 800 Mhz sampling rate with 6.2 GeV max range of 2038 ADC samples
       ADC = (int)(GeVToADC*Elab); // makes 2048 sample ADC max = 6.2 GeV
       ADC = ADC/0.4; // divide by max frac (~0.4 from erfunc plot) of 5ns FWHM pulse in peak sample for total samples of pulse
       
       // add empirical energy dependent drift time, see https://muon.npl.washington.edu/elog/g2/Simulation/229
       ylab = Elab/Elab_max;
       phase = p0 + p1*ylab + p2*ylab*ylab + p3*ylab*ylab*ylab + p4*ylab*ylab*ylab*ylab + p5*ylab*ylab*ylab*ylab*ylab + p6*ylab*ylab*ylab*ylab*ylab*ylab + p7*ylab*ylab*ylab*ylab*ylab*ylab*ylab + p8*ylab*ylab*ylab*ylab*ylab*ylab*ylab*ylab + p9*ylab*ylab*ylab*ylab*ylab*ylab*ylab*ylab*ylab; // in mSR units of omega_a
       drifttime = anomalousperiod*phase/(2.*Pi*1000.); // omega_a phase to drift time in Q-method ticks
       tick = tick + drifttime;
       
       // get random (x, y) coordinates
       //xcoord = nxseg * curand_uniform(&localState);
       //ycoord = nyseg * curand_uniform(&localState);
       
       // make rough empirical x-distribution obtained from  https://muon.npl.washington.edu/elog/g2/Simulation/258 (Robin)
       // and rough empirical y-distribution obtained from  https://muon.npl.washington.edu/elog/g2/Simulation/256 (Pete)
       if ( ylab > 0.7 ) {
	 xmax = 185.-533.3*(ylab-0.7);
       } else {
	 xmax = 185.;
       }
       xrand = curand_uniform(&localState);
       xcoord = xmax*xrand/25.0; // x-coordinate -> mm -> segments
       yrand = curand_uniform(&localState);
       ycoord = 1.0+(nyseg-2.0)*yrand; // y-coordinate -> segments
       
       // put hit information into hit array
       tickstore[nhit] = tick;
       ADCstore[nhit] = ADC;
       xcoordstore[nhit] = xcoord;
       ycoordstore[nhit] = ycoord;
       iold[nhit] = nhit;
       nhit++;
     }
     
     // sort array of hits into ascending time-order
     int itemp;
     float temp;
     for (int i = 0; i < nhit; ++i) {
       for (int j = i + 1; j < nhit; ++j) {
	 // if higher index array element j is earlier (t_j < t_i) than lower index  array element i then swap elements
	 if (tickstore[i] > tickstore[j]) {
	   // swap times
	   temp = tickstore[i]; 
	   tickstore[i] = tickstore[j];
	   tickstore[j] = temp;
	   // swap indices for late use in swapping ADC, x, y etc
	   itemp = iold[i]; 
	   iold[i] = iold[j];
	   iold[j] = itemp;
	 }
       }
     }
     
     
     // simple pileup effect - needs work
     //float dt = 0.0, dtmin = -999999.;
     //for (int j = 0; j < i; j++){
     //	dt = tick - tickstore[j];
     //	if ( dt < dtmin ) continue;
     //}
     
     // distribute over calo segments
     // parameters for empirical Gaussian distribution of energy across neighboring segments. Used 
     // https://muon.npl.washington.edu/elog/g2/SLAC+Test+Beam+2016/260 and position where energy in 
     // neighboring xtal is 16% (1 sigma) - giving sigma = 0.19 in units of crystal size 
     float xsig = 0.19, ysig = 0.19; // xtal size units
     
     // distribute over time bin
     //  approx sigma width of 2.1ns from https://muon.npl.washington.edu/elog/g2/SLAC+Test+Beam+2016/38
     const float width = 2.1/nsPerTick; // pulse sigma in q-method bin width units
     
     for (int i = 0; i < nhit; i++){
       
       
       // array is time-ordered
       tick = tickstore[i]; 
       // array isn't time-ordered
       ADC = ADCstore[iold[i]];
       xcoord = xcoordstore[iold[i]];
       ycoord = ycoordstore[iold[i]];
       //printf(" i %i, iold %i, tick %f, ADC %i, xcoord %f, ycoord %f\n", i, iold[i], tick, ADC, xcoord, ycoord);
       
       int itick = (int)tick;
       float rtick = tick - itick;
       hitArray[itick]++; // debugging only
       
       for (int ix = 0; ix < nxseg; ix++) {
	 for (int iy = 0; iy < nyseg; iy++) {
	   
	   // energy in segment
	   float fsegment = (1./(xsig*sqrt(2.*Pi)))*exp(-(ix+0.5-xcoord)*(ix+0.5-xcoord)/(2.*xsig*xsig))*(1./(ysig*sqrt(2.*Pi)))*exp(-(iy+0.5-ycoord)*(iy+0.5-ycoord)/(2.*ysig*ysig));
	   int ADCsegment = fsegment*ADC;
	   if (ADCsegment < 1) continue;
	   
	   int xysegmentoffset = (ix+iy*nxseg)*fill_buffer_max_length; // offset for segment in samples array
	   //printf("ix %i, iy %i, offset %i\n", ix, iy, xysegmentoffset);
	   
	   // distribute pulse over several contiguous bins
	   float gsum = 0;
	   for (int k=-2; k<=2; k++) gsum += exp(-0.5*(k+rtick)*(k+rtick)/width/width);
	   for (int k=-2; k<=2; k++) {
	     int kk = k + itick;
	     if ( kk < 0 || kk >= fill_buffer_max_length ) continue;
	     int ADCfrac = ADCsegment*exp(-0.5*(k+rtick)*(k+rtick)/width/width)/gsum;
	     if ( ADCfrac > 2048 ) ADCfrac = 2048;
	     if ( ADCfrac >= 1 ) *(fillArray + nsegs*fill_buffer_max_length*idx + xysegmentoffset + kk ) += ADCfrac;  // fill buffer
	   }
	 } // y-distribution loop
       } // x-distribution loop
     } // good hits
     
     // state index for random number generator
     state[idx] =  localState;
     
     /*
     // fill test pattern
     for (int i = 0; i < nsegs; i++) {
     for (int j = 0; j < fill_buffer_max_length; j++) {
     *(fillArray + nsegs*fill_buffer_max_length*idx + fill_buffer_max_length*i + j) += i;  // fill buffer
     }
     }
     */
   }
}

__global__ void make_fillsum( curandState *state, int32_t *fillArray, float *fillSumArray, int nfills, int fill_buffer_max_length, bool flushnoise ) {

  int idx = blockIdx.x*256 + threadIdx.x;
  curandState localState = state[idx];

  int nxsegs = NXSEG, nysegs = NYSEG;
  int nsegs = NSEG;
  
  // fill_buffer_max_length is Q-method bins per segment per fill
  if (idx < nsegs * fill_buffer_max_length) {

    // initialize flush to zero (now use cudaMemset)
    //*(fillSumArray + idx) = 0.0;

    // add fills in flush
    for (int i = 0; i < nfills; i++) {
      *(fillSumArray + idx) += (float) *(fillArray + (nsegs*fill_buffer_max_length)*i + idx );  // fill buffer
    }

    // add noise
    if (flushnoise) {
      float pedestal = 0., sigma = 4.;
      float noise = pedestal + sigma * curand_normal(&localState); // random from Gaussian using uniform random number 0->1
      *(fillSumArray + idx) += noise;  // fill buffer
      state[idx] = localState;
    }
 
  }
}

__global__ void make_hitsum( int32_t *hitArray, float *hitSumArray, int nfills, int fill_buffer_max_length, bool flushnoise) {

  int idx = blockIdx.x*256 + threadIdx.x;
  
  // fill_buffer_max_length is Q-method bins per segment per fill
  if (idx < fill_buffer_max_length) {
    
    // initialize flush to zero (now use cudaMemset)
    //  *(hitSumArray + idx) = 0.0;
    
    // add fills in flush
    for (int i = 0; i < nfills; i++) {
      *(hitSumArray + idx) += (float) *( hitArray + (fill_buffer_max_length)*i + idx );  // fill buffer
    }   
  }
}

int main(int argc, char * argv[]){
 
  // define nthreads, nblocks, arrays for GPU
  cudaError err;
  int nthreads = 256, nblocks1, nblocks2, nblocks3, nblocks4, nblocks5;
  int nfills, nflushes;
  curandState *d_state, *d_state2;
  float *h_randArray, *d_randArray;
  // Q-method arrays 
  int32_t *h_fillArray, *d_fillArray;
  float *h_fillSumArray, *d_fillSumArray; 
  // tur hits arrays 
  int32_t *h_hitArray, *d_hitArray;
  float *h_hitSumArray, *d_hitSumArray; 

  // define fill length, clock tick for simulation
  //const int nsPerFill = 4096, nsPerTick = 16; 
  const int nsPerFill = 560000, nsPerTick = 16; 
  int fill_buffer_max_length = nsPerFill / nsPerTick;
  int nxsegs = NXSEG, nysegs = NYSEG;
  int nsegs = NSEG;
 
  // define run, flush, fill structure
  if (argc == 1) {
    nfills = 256;
    nflushes = 1;
  } else {
    nfills = atoi(argv[1]);
    nflushes = atoi(argv[2]);
  }
  printf("nsegments per calo %d, nfills per flush %d, nflushes per run %d\n", nsegs, nfills, nflushes);

  // define grid structure
  nblocks1 = nfills / nthreads + 1;
  nblocks2 = ( nsegs * fill_buffer_max_length + nthreads - 1 )/ nthreads;
  nblocks3 = ( fill_buffer_max_length + nthreads - 1 )/ nthreads;
  nblocks4 = ( nsegs * nfills * fill_buffer_max_length + nthreads - 1 )/ nthreads;
  nblocks5 = ( nfills * fill_buffer_max_length + nthreads - 1 )/ nthreads;
  printf("per flush grid: nthreads %i, nblocks %i nthreads*nblocks %i\n", nthreads, nblocks1, nthreads*nblocks1 );
  printf("per bin grid: nthreads %i, nblocks %i nthreads*nblocks %i\n", nthreads, nblocks3, nthreads*nblocks3 );
  printf("per bin per segment grid: nthreads %i, nblocks %i nthreads*nblocks %i\n", nthreads, nblocks2, nthreads*nblocks2 );
  printf("per bin per fill grid: nthreads %i, nblocks %i nthreads*nblocks %i\n", nthreads, nblocks5, nthreads*nblocks5 );
  printf("per bin per segment per fill grid: nthreads %i, nblocks %i nthreads*nblocks %i\n", nthreads, nblocks4, nthreads*nblocks4 );

  // histogram binning
  printf("ns per fill %i, ns per bin %i, number of bins %d\n", nsPerFill, nsPerTick, fill_buffer_max_length);
  Char_t hname[256];
  for (int ih = 0; ih < nsegs; ih++) {
    sprintf( hname, "\n hFlush1D%02i", ih);
    hFlush1D[ih] = new TH1D( hname, hname, fill_buffer_max_length, 0.0, fill_buffer_max_length );
    sprintf( hname, "\n hFlush2D%02i", ih);
    hFlush2D[ih] = new TH2D( hname, hname, fill_buffer_max_length, 0.0, fill_buffer_max_length, 128, -64, 63 );
    sprintf( hname, "\n hFlush2DCoarse%02i", ih);
    hFlush2DCoarse[ih] = new TH2D( hname, hname, fill_buffer_max_length, 0.0, fill_buffer_max_length, 128, 0, 2048 );
  }
  sprintf( hname, "\n hHits1D");
  hHits1D = new TH1D( hname, hname, fill_buffer_max_length, 0.0, fill_buffer_max_length );

  // from TDR 
  // events for 100ppb stat uncertainty, 1.6e11
  // >30us, >1.86 GeV positrons for 24 calos per fill, 1100
  // positrons for 24 calos per fill, 5500 (from TDR Fig 16.8 for energy and exp(-t/tau) for time) 
  // fills for 140 ppb, 1.5e8

  // on average 1100 good e's per fill, 5500 e's per fill
  int ne = 5500*2;
  // divide by 24 for per calorimeter rate.
  ne = ne / 24;
  // no pulse test
  //ne = 0;
  // do fill-by-fill noise
  bool fillbyfillnoise = false; 
  // do flush-by-flush noise
  bool flushbyflushnoise = false; 

  //FILE *fp;
  //fp = fopen( "fillSumArray.dat", "w" ); // Open file for writing
 
  // set device number
  int num_devices, device;
  cudaGetDeviceCount(&num_devices);
  if (num_devices > 1) {
     for (device = 0; device < num_devices; device++) {
  	 cudaDeviceProp properties;
	 cudaGetDeviceProperties(&properties, device);
         printf("device %d properties.multiProcessorCount %d\n", device, properties.multiProcessorCount);
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

  // device arrays for random number logic
  // could avoid the factor nfills in arrays by using atomicAdd() if only want flushed histograms?
  cudaMalloc( (void **)&d_state, nsegs*nfills*sizeof(curandState));
  cudaMalloc( (void **)&d_state2, nsegs*fill_buffer_max_length*sizeof(curandState));
  err = cudaThreadSynchronize();
  if ( cudaSuccess != err ) {
    printf("Cuda error in file '%s' in line %i : %s.\n",
             __FILE__, __LINE__, cudaGetErrorString( err) );
  }

  // host, device arrays for random, muon type fills
  h_randArray = (float *)malloc(nsegs*nfills*sizeof(float));
  cudaMalloc( (void **)&d_randArray, nsegs*nfills*sizeof(float));
  h_fillArray = (int32_t *)malloc(nsegs*nfills*fill_buffer_max_length*sizeof(int32_t));
  cudaMalloc( (void **)&d_fillArray, nsegs*nfills*fill_buffer_max_length*sizeof(int32_t));
  h_hitArray = (int32_t *)malloc(nfills*fill_buffer_max_length*sizeof(int32_t));
  cudaMalloc( (void **)&d_hitArray, nfills*fill_buffer_max_length*sizeof(int32_t));
  err = cudaThreadSynchronize();
  if ( cudaSuccess != err ) {
    printf("Cuda error in file '%s' in line %i : %s.\n",
             __FILE__, __LINE__, cudaGetErrorString( err) );
  }

  // host, device arrays for flushes
  h_fillSumArray = (float *)malloc(nsegs*fill_buffer_max_length*sizeof(float));
  cudaMalloc( (void **)&d_fillSumArray, nsegs*fill_buffer_max_length*sizeof(float));
  h_hitSumArray = (float *)malloc(fill_buffer_max_length*sizeof(float));
  cudaMalloc( (void **)&d_hitSumArray, fill_buffer_max_length*sizeof(float));
  err = cudaThreadSynchronize();
  if ( cudaSuccess != err ) {
    printf("Cuda error in file '%s' in line %i : %s.\n",
             __FILE__, __LINE__, cudaGetErrorString( err) );
  }

  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf(" ::: kernel malloc / cudaMalloc time %f ms\n",elapsedTime);

  // initialization for random number generator every 100 flushes
  cudaEventRecord(start, 0);
  init_rand<<<nblocks1,nthreads>>>( d_state, 0, time(NULL));
  init_rand<<<nblocks2,nthreads>>>( d_state2, 0, time(NULL));
  
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsedTime, start, stop);
  printf(" ::: kernel init_rand time %f ms\n",elapsedTime);

  struct timeval start_time, end_time;
  gettimeofday(&start_time, NULL);
  
  for (int j = 0; j < nflushes; j++){

    printf("flush %i\n", j);

    cudaEventRecord(start, 0);
    /* memset on host, copy to device 
    cudaMemcpy( d_fillArray, h_fillArray, nsegs*nfills*fill_buffer_max_length*sizeof(int32_t), cudaMemcpyHostToDevice);
    */
    // zero arrays for next flush

    // too many blocks if size nsegs*nfills*fill_buffer_max_length
    for (int iz = 0; iz < nfills; iz++){
      zero_int_array<<<nblocks2,nthreads>>>( &d_fillArray[iz*nsegs*fill_buffer_max_length], nsegs*fill_buffer_max_length);
    }
    err=cudaGetLastError();
    if(err!=cudaSuccess) {
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(0);
    }  
    zero_int_array<<<nblocks5,nthreads>>>( d_hitArray, nfills*fill_buffer_max_length);
    err=cudaGetLastError();
    if(err!=cudaSuccess) {
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(0);
    }  
    zero_float_array<<<nblocks2,nthreads>>>( d_fillSumArray, nsegs*fill_buffer_max_length);
    err=cudaGetLastError();
    if(err!=cudaSuccess) {
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(0);
    }  
    zero_float_array<<<nblocks3,nthreads>>>( d_hitSumArray, fill_buffer_max_length);
    err=cudaGetLastError();
    if(err!=cudaSuccess) {
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(0);
    }  
    /*
    err = cudaMemset( d_hitArray, 0, nfills*fill_buffer_max_length*sizeof(int32_t));
    if ( err != cudaSuccess ) {
      printf("cudaMemset error!\n");
      return;
    }
    cudaMemset( d_fillArray, 0, nsegs*nfills*fill_buffer_max_length*sizeof(int32_t));
    if ( err != cudaSuccess ) {
      printf("cudaMemset error!\n");
      return;
    }
    cudaMemset( d_fillSumArray, 0.0, nsegs*fill_buffer_max_length*sizeof(float));
    if ( err != cudaSuccess ) {
      printf("cudaMemset error!\n");
      return;
    }
    cudaMemset( d_hitSumArray, 0.0, fill_buffer_max_length*sizeof(float));
    if ( err != cudaSuccess ) {
      printf("cudaMemset error!\n");
      return;
    }
    */
    cudaThreadSynchronize();

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    if (j == 0) printf(" ::: kernel initialize fillArray %f ms\n",elapsedTime);

    cudaEventRecord(start, 0);
    
    //init_rand<<<nblocks1,nthreads>>>( d_state, time(NULL));
    //make_rand<<<nblocks1,nthreads>>>( d_state, d_randArray); // testing
    //make_randexp<<<nblocks1,nthreads>>>( d_state, d_randArray, tau); // testing
    make_randfill<<<nblocks1,nthreads>>>( d_state, d_hitArray, d_fillArray, ne, fill_buffer_max_length, nfills, fillbyfillnoise);
    err=cudaGetLastError();
    if(err!=cudaSuccess) {
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(0);
    }  

    /*
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    if (j == 0) printf(" ::: kernel make_randfill time %f ms\n",elapsedTime);

    cudaEventRecord(start, 0);

    make_hitsum<<<nblocks3,nthreads>>>( d_hitArray, d_hitSumArray, nfills, fill_buffer_max_length, flushbyflushnoise);
    err=cudaGetLastError();
    if(err!=cudaSuccess) {
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(0);
    }  

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    if (j == 0) printf(" ::: kernel make_hitsum time %f ms\n",elapsedTime);
    */

    cudaEventRecord(start, 0);

    //init_rand<<<nblocks2,nthreads>>>( d_state2, time(NULL));
    make_fillsum<<<nblocks2,nthreads>>>( d_state2, d_fillArray, d_fillSumArray, nfills, fill_buffer_max_length, flushbyflushnoise);
    err=cudaGetLastError();
    if(err!=cudaSuccess) {
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(0);
    }  

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

    cudaMemcpy( h_fillArray, d_fillArray, nfills*fill_buffer_max_length*sizeof(int32_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < nfills*fill_buffer_max_length; i++){
        if ( *(h_fillArray+i) > 0) printf(" fillArray[i] = %u[%i]\n", *(h_fillArray+i), i );
    }
   */

    cudaEventRecord(start, 0);
    cudaMemcpy( h_fillSumArray, d_fillSumArray, nsegs*fill_buffer_max_length*sizeof(float), cudaMemcpyDeviceToHost);
    err=cudaGetLastError();
    if(err!=cudaSuccess) {
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(0);
    }  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    cudaEventRecord(start, 0);
    cudaMemcpy( h_hitSumArray, d_hitSumArray, fill_buffer_max_length*sizeof(float), cudaMemcpyDeviceToHost);
    err=cudaGetLastError();
    if(err!=cudaSuccess) {
      printf("Cuda failure %s:%d: '%s'\n",__FILE__,__LINE__,cudaGetErrorString(err));
      exit(0);
    }  
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);

    for (int i = 0; i < nsegs*fill_buffer_max_length; i++){
      int ih = i / fill_buffer_max_length;
      int ib = i % fill_buffer_max_length;
      //if (ib == 0) printf("i %i, ib %i, ih %i, *(h_fillSumArray+i) %f\n", i, ib, ih, *(h_fillSumArray+i));
      hFlush1D[ih]->Fill( ib+1, *(h_fillSumArray+i));
      hFlush2D[ih]->Fill( ib+1, *(h_fillSumArray+i));
      hFlush2DCoarse[ih]->Fill( ib+1, *(h_fillSumArray+i));
      //fprintf(fp, " %i %f\n", i+1, *(h_fillSumArray+i) );      
    }
    for (int ib = 0; ib < fill_buffer_max_length; ib++){
      hHits1D->Fill( ib+1, *(h_hitSumArray+ib));
    }

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&elapsedTime, start, stop);
    if (j == 0) printf(" ::: kernel cuda memcpy time + fill root histograms %f ms\n",elapsedTime);

  }

  // free device arrays
  cudaFree(d_state);
  cudaFree(d_state2);
  cudaFree(d_randArray);
  cudaFree(d_hitArray);
  cudaFree(d_fillArray);
  cudaFree(d_fillSumArray);
  cudaFree(d_hitSumArray);

  
  gettimeofday(&end_time, NULL);
  printf("elapsed processing time, dt %f secs\n", toddiff(&end_time, &start_time));

  f = new TFile("test.root","recreate");
  for (int ih = 0; ih < nsegs; ih++) {
    //int ih = 0;
    //printf("writing segment %i\n", ih);
    sprintf( hname, "h%02i", ih);
    f->WriteObject( hFlush1D[ih], hname);
    sprintf( hname, "s%02i", ih);
    f->WriteObject( hFlush2D[ih], hname);
    sprintf( hname, "sc%02i", ih);
    f->WriteObject( hFlush2DCoarse[ih], hname);
    //f->WriteObject( hFlush2DCourse, "hFlush2DCourse");
  }
  sprintf( hname, "hHits");
  f->WriteObject( hHits1D, hname);
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

