#--------------------------------------------------------------------
# CUDA
.SUFFIXES : .cu .cu_dbg.o .c_dbg.o .cpp_dbg.o .cu_rel.o .c_rel.o .cpp_rel.o .cubin .ptx

# Add new SM Versions here as devices with new Compute Capability are released
#SM_VERSIONS   := 10 11 12 13 20 21
#SM_VERSIONS   := 13

# Compilers
NVCC       := $(CUDASYS)/bin/nvcc 

# Includes
INCLUDES  += -I. -I$(CUDASYS)/include  -I$(CUDASYS)/NVIDIA_GPU_Computing_SDK/shared/inc -I$(CUDASYS)/NVIDIA_GPU_Computing_SDK/C/common/inc 
#-I$(MIDASSYS)/include

# Warning flags
CXXWARN_FLAGS := \
	-W -Wall \
	-Wimplicit \
	-Wswitch \
	-Wformat \
	-Wchar-subscripts \
	-Wparentheses \
	-Wmultichar \
	-Wtrigraphs \
	-Wpointer-arith \
	-Wcast-align \
	-Wreturn-type \
	-Wno-unused-function \
	$(SPACE)

CWARN_FLAGS := $(CXXWARN_FLAGS) \
	-Wstrict-prototypes \
	-Wmissing-prototypes \
	-Wmissing-declarations \
	-Wnested-externs \
	-Wmain \



# Determining the necessary Cross-Compilation Flags
# 32-bit OS, but we target 64-bit cross compilation
ifeq ($(OSTYPE),linux64)
    NVCCFLAGS       += -m64
    LIB_ARCH         = x86_64
    CUDPPLIB_SUFFIX  = x86_64
    ifneq ($(DARWIN),)
         CXX_ARCH_FLAGS += -arch x86_64
    else
         CXX_ARCH_FLAGS += -m64
    endif
else 
# 64-bit OS, and we target 32-bit cross compilation
    ifeq ($(OSTYPE),linux)
        NVCCFLAGS       += -m32
        LIB_ARCH         = i386
        CUDPPLIB_SUFFIX  = i386
        ifneq ($(DARWIN),)
             CXX_ARCH_FLAGS += -arch i386
        else
             CXX_ARCH_FLAGS += -m32
        endif
    endif	
endif


CXXFLAGS  += $(CXXWARN_FLAGS) $(CXX_ARCH_FLAGS) $(INCLUDES)
CCFLAGS    += $(CWARN_FLAGS) $(CXX_ARCH_FLAGS) $(INCLUDES)
NVCCFLAGS += $(INCLUDES) -O2 --compiler-options -fno-strict-aliasing \
	-arch compute_20 --ptxas-options=-v 
#-Xcicc -O0 -Xptxas -O0 // to turn off gpu optimization

#	-gencode=arch=compute_20,code=\"sm_20,compute_20\"
LINKFLAGS +=
LINK      += $(LINKFLAGS) $(CXX_ARCH_FLAGS)

# Common flags
#COMMONFLAGS += $(INCLUDES) -DUNIX

# Debug/release configuration
ifeq ($(dbg),1)
	COMMONFLAGS += -g
	NVCCFLAGS   += -D_DEBUG -G
	CXXFLAGS    += -D_DEBUG
	CFLAGS      += -D_DEBUG
	BINSUBDIR   := debug
	LIBSUFFIX   := D
else 
	COMMONFLAGS += -O2 
	BINSUBDIR   := release
	LIBSUFFIX   := 
	NVCCFLAGS   += --compiler-options -fno-strict-aliasing
	CXXFLAGS    += -fno-strict-aliasing
	CFLAGS      += -fno-strict-aliasing
endif

ifeq ($(OSTYPE),linux64)
       LDFLAGS       += -L$(CUDASYS)/lib64 -L$(CUDASYS)/NVIDIA_GPU_Computing_SDK/shared/lib -L$(CUDASYS)/NVIDIA_GPU_Computing_SDK/C/lib -L$(CUDASYS)/NVIDIA_GPU_Computing_SDK/C/common/lib/linux -L/usr/lib64/nvidia  -lcuda -lcudart 
else
       LDFLAGS      += -L$(CUDASYS)/lib -L$(CUDASYS)/NVIDIA_GPU_Computing_SDK/shared/lib -L$(CUDASYS)/NVIDIA_GPU_Computing_SDK/C/lib -L$(CUDASYS)/NVIDIA_GPU_Computing_SDK/C/common/lib/linux -lcuda -lcudart 
endif




################################################################################
# Rules
################################################################################

# Default arch includes gencode for sm_10, sm_20, and other archs from GENCODE_ARCH declared in the makefile
%.cu.o : $(SRCDIR)%.cu $(CU_DEPS)
	$(NVCC) $(GENCODE_SM10) $(GENCODE_ARCH) $(GENCODE_SM20) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -c $<

# Default arch includes gencode for sm_10, sm_20, and other archs from GENCODE_ARCH declared in the makefile
%.cubin : $(SRCDIR)%.cu cubindirectory
	$(VERBOSE)$(NVCC) $(GENCODE_SM10) $(GENCODE_ARCH) $(GENCODE_SM20) $(CUBIN_ARCH_FLAG) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -cubin $<

%.ptx : $(SRCDIR)%.cu ptxdirectory
	$(VERBOSE)$(NVCC) $(CUBIN_ARCH_FLAG) $(NVCCFLAGS) $(SMVERSIONFLAGS) -o $@ -ptx $<

#
# The following definition is a template that gets instantiated for each SM
# version (sm_10, sm_13, etc.) stored in SMVERSIONS.  It does 2 things:
# 1. It adds to OBJS a .cu_sm_XX.o for each .cu file it finds in CUFILES_sm_XX.
# 2. It generates a rule for building .cu_sm_XX.o files from the corresponding 
#    .cu file.
#
# The intended use for this is to allow Makefiles that use common.mk to compile
# files to different Compute Capability targets (aka SM arch version).  To do
# so, in the Makefile, list files for each SM arch separately, like so:
# This will be used over the default rule abov
#
# CUFILES_sm_10 := mycudakernel_sm10.cu app.cu
# CUFILES_sm_12 := anothercudakernel_sm12.cu
#
#define SMVERSION_template
##OBJS += $(patsubst %.cu,$(OBJDIR)/%.cu_$(1).o,$(notdir $(CUFILES_$(1))))
#OBJS += $(patsubst %.cu,$(OBJDIR)/%.cu_$(1).o,$(notdir $(CUFILES_sm_$(1))))
#$(OBJDIR)/%.cu_$(1).o : $(SRCDIR)%.cu $(CU_DEPS)
##	$(VERBOSE)$(NVCC) -o $$@ -c $$< $(NVCCFLAGS)  $(1)
#	$(VERBOSE)$(NVCC) -gencode=arch=compute_$(1),code=\"sm_$(1),compute_$(1)\" $(GENCOD#E_SM20) -o $$@ -c $$< $(NVCCFLAGS)
#endef

