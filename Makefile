NVCC =  /usr/local/cuda/bin/nvcc
#NVCC =  /home/daq/DAQ/cuda/bin/nvcc
#NVCC =   /mnt/sw/cuda-5.5/bin/nvcc

CUDAPATH =  /usr/local/cuda
#CUDAPATH =  /home/daq/DAQ/cuda
#CUDAPATH = /mnt/sw/cuda-5.5

NVCCFLAGS = -I$(CUDAPATH)/include

include cuda.mk

LFLAGS = -L$(CUDAPATH)/lib64 -lcuda -lcudart -lcurand -lm

#ROOTCFLAGS := $(shell  $(ROOTSYS)/bin/root-config --cflags)
#ROOTCFLAGS := -m64 -I/home/daq/DAQ/root_5.30_x86_64/include
ROOTCFLAGS := -m64 -I$(ROOTSYS)/include
ROOTCFLAGS += -DHAVE_ROOT -DUSE_ROOT
#ROOTLIBS   := $(shell  $(ROOTSYS)/bin/root-config --libs) -Wl,-rpath,$(ROOTSYS)/lib
#ROOTLIBS   := -L/home/daq/DAQ/root_5.30_x86_64/lib -lGpad -lHist -lGraf -lGraf3d -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lRIO -lNet -lThread -lCore -lCint -lm -ldl -L$(ROOTSYS)/lib
ROOTLIBS   := -lGpad -lHist -lGraf -lGraf3d -lTree -lRint -lPostscript -lMatrix -lPhysics -lMathCore -lRIO -lNet -lThread -lCore -lCint -lm -ldl -L$(ROOTSYS)/lib 


ROOTLIBS   += -lThread

#LIB      += $(ROOTLIBS)
LFLAGS  += $(ROOTLIBS)
NVCCFLAGS += $(ROOTCFLAGS)

all:
#	$(NVCC) $(NVCCFLAGS) $(LFLAGS) -o Timing Timing.cu
#	$(NVCC) $(NVCCFLAGS) $(LFLAGS) -o vecadd vecadd.cu
	$(NVCC) $(NVCCFLAGS) $(LFLAGS) -o rand rand.cu


