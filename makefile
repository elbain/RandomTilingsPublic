CXX = g++
LDLIBS = OpenCL
LDFLAGS = "C:\Program Files (x86)\Intel\OpenCL SDK\6.3\lib\x86"
srcdir = "C:\Program Files (x86)\Intel\OpenCL SDK\6.3\include"
#LDFLAGS = "C:\Program Files (x86)\IntelSWTools\sw_dev_tools\OpenCL\sdk\lib\x86"
#srcdir = "C:\Program Files (x86)\IntelSWTools\sw_dev_tools\OpenCL\sdk\include"
CXXFLAGS = -w -std=gnu++11

#DEBUG = -DDEBUG -g -G 
CUDALINK = -lcuda
#ARCH = -arch=sm_30

NVCC = nvcc $(DEBUG) $(ARCH) -I. --ptxas-options=-v  

openclTest: common.o src/common/common.h
	$(CXX) $(CXXFLAGS) -o openclTest common.o Examples/openclTest.cpp  -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)

cudaTest: src/common/common.h
	${NVCC} -o $@ Examples/cudaTest.cpp  ${CUDALINK}

ifndef cuda
MinimalDominoEx: common.o src/common/common.h file_reader.o DominoTiler.o src/Domino/DominoTiler.h 
	$(CXX) $(CXXFLAGS) -o MinimalDominoEx common.o file_reader.o DominoTiler.o Examples/MinimalDominoEx.cpp  -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)
else
MinimalDominoEx: src/common/common.h src/common/common.cpp src/Domino/DominoTiler.h src/Domino/DominoTiler.cpp  src/Domino/dominokernel.cu src/Domino/dominokernel.cuh
	${NVCC} -o $@ src/common/common.cpp src/Domino/DominoTiler.cpp src/Domino/dominokernel.cu Examples/MinimalDominoEx.cpp  ${CUDALINK}
endif

ifndef cuda
AztecDiamondCFTPEx: common.o src/common/common.h file_reader.o DominoTiler.o src/Domino/DominoTiler.h 
	$(CXX) $(CXXFLAGS) -o AztecDiamondCFTPEx common.o file_reader.o DominoTiler.o Examples/AztecDiamondCFTPEx.cpp  -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)	
else
AztecDiamondCFTPEx: src/common/common.h src/common/common.cpp src/Domino/DominoTiler.h src/Domino/DominoTiler.cpp  src/Domino/dominokernelCFTP.cu src/Domino/dominokernel.cuh
	${NVCC} -o $@ -DCFTP src/common/common.cpp src/Domino/DominoTiler.cpp src/Domino/dominokernelCFTP.cu Examples/AztecDiamondCFTPEx.cpp  ${CUDALINK}	
endif

ifndef cuda
MinimalLozengeEx: common.o src/common/common.h file_reader.o LozengeTiler.o src/Lozenge/LozengeTiler.h
	$(CXX) $(CXXFLAGS) -o MinimalLozengeEx common.o file_reader.o LozengeTiler.o Examples/MinimalLozengeEx.cpp  -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)	
else
MinimalLozengeEx: src/common/common.h src/common/common.cpp src/Lozenge/LozengeTiler.h src/Lozenge/LozengeTiler.cpp  src/Domino/dominokernel.cu src/Domino/dominokernel.cuh
	${NVCC} -o $@ src/common/common.cpp src/Lozenge/LozengeTiler.cpp src/Lozenge/lozengekernel.cu Examples/MinimalLozengeEx.cpp  ${CUDALINK}
endif
	
ifndef cuda
RectTriangleEx: common.o src/common/common.h file_reader.o RectTriangleTiler.o src/RectTriangle/RectTriangleTiler.h
	$(CXX) $(CXXFLAGS) -o RectTriangleEx common.o file_reader.o RectTriangleTiler.o Examples/RectTriangleEx.cpp  -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)	
else
RectTriangleEx: src/common/common.h src/common/common.cpp src/RectTriangle/RectTriangleTiler.h src/RectTriangle/RectTriangleTiler.cpp  src/RectTriangle/recttrianglekernel.cu src/RectTriangle/recttrianglekernel.cuh
	${NVCC} -o $@ src/common/common.cpp src/RectTriangle/RectTriangleTiler.cpp src/RectTriangle/recttrianglekernel.cu Examples/RectTriangleEx.cpp  ${CUDALINK}
endif
	
ifndef cuda
TriangleDimerEx: common.o src/common/common.h file_reader.o TriangleDimerTiler.o src/TriangleDimer/TriangleDimerTiler.h
	$(CXX) $(CXXFLAGS) -o TriangleDimerEx common.o file_reader.o TriangleDimerTiler.o Examples/TriangleDimerEx.cpp  -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)	
else
TriangleDimerEx: src/common/common.h src/common/common.cpp src/TriangleDimer/TriangleDimerTiler.h src/TriangleDimer/TriangleDimerTiler.cpp  src/TriangleDimer/triangledimerkernel.cu src/TriangleDimer/triangledimerkernel.cuh
	${NVCC} -o $@ src/common/common.cpp src/TriangleDimer/TriangleDimerTiler.cpp src/TriangleDimer/triangledimerkernel.cu Examples/TriangleDimerEx.cpp  ${CUDALINK}
endif

ifndef cuda
WeightedAztecDiamondEx: common.o src/common/common.h file_reader.o DominoTiler.o src/Domino/DominoTiler.h 
	$(CXX) $(CXXFLAGS) -o WeightedAztecDiamondEx common.o file_reader.o DominoTiler.o Examples/WeightedAztecDiamondEx.cpp  -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)
else
WeightedAztecDiamondEx: src/common/common.h src/common/common.cpp src/Domino/DominoTiler.h src/Domino/DominoTiler.cpp  src/Domino/dominokerneltwoperiodic.cu src/Domino/dominokernel.cuh
	${NVCC} -o $@ src/common/common.cpp src/Domino/DominoTiler.cpp src/Domino/dominokerneltwoperiodic.cu Examples/WeightedAztecDiamondEx.cpp  ${CUDALINK}
endif

common.o: src/common/common.cpp src/common/common.h
	$(CXX) $(CXXFLAGS) -c src/common/common.cpp -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)
	
DominoTiler.o: src/Domino/DominoTiler.cpp src/Domino/DominoTiler.h src/common/common.h src/TinyMT/file_reader.h
	$(CXX) $(CXXFLAGS) -c src/Domino/DominoTiler.cpp -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)
	
LozengeTiler.o: src/Lozenge/LozengeTiler.cpp src/Lozenge/LozengeTiler.h src/common/common.h src/TinyMT/file_reader.h
	$(CXX) $(CXXFLAGS) -c src/Lozenge/LozengeTiler.cpp -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)	

RectTriangleTiler.o: src/RectTriangle/RectTriangleTiler.cpp src/RectTriangle/RectTriangleTiler.h src/common/common.h src/TinyMT/file_reader.h
	$(CXX) $(CXXFLAGS) -c src/RectTriangle/RectTriangleTiler.cpp -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)
	
TriangleDimerTiler.o: src/TriangleDimer/TriangleDimerTiler.cpp src/TriangleDimer/TriangleDimerTiler.h src/common/common.h src/TinyMT/file_reader.h
	$(CXX) $(CXXFLAGS) -c src/TriangleDimer/TriangleDimerTiler.cpp -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)

file_reader.o: src/TinyMT/file_reader.cpp src/TinyMT/file_reader.h
	$(CXX) $(CXXFLAGS) -c src/TinyMT/file_reader.cpp -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)

ifndef cuda
AztecDiamondTiling: common.o src/common/common.h file_reader.o DominoTiler.o src/Domino/DominoTiler.h 
	$(CXX) $(CXXFLAGS) -o AztecDiamondTiling common.o file_reader.o DominoTiler.o AztecDiamondWaves/AztecDiamondTiling.cpp  -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)	
else
AztecDiamondTiling: src/common/common.h src/Domino/DominoTiler.h src/Domino/DominoTiler.cpp src/common/common.cpp src/Domino/dominokernel.cu src/Domino/dominokernel.cuh AztecDiamondWaves/AztecDiamondTiling.cpp
	${NVCC} -o $@ src/common/common.cpp src/Domino/DominoTiler.cpp src/Domino/dominokernel.cu AztecDiamondWaves/AztecDiamondTiling.cpp ${CUDALINK}
endif

ifndef cuda
AztecDiamondAvHeight: common.o src/common/common.h file_reader.o DominoTiler.o src/Domino/DominoTiler.h 
	$(CXX) $(CXXFLAGS) -o AztecDiamondAvHeight common.o file_reader.o DominoTiler.o AztecDiamondWaves/AztecDiamondAvHeight.cpp  -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)  
else
AztecDiamondAvHeight: src/common/common.h src/Domino/DominoTiler.h src/Domino/DominoTiler.cpp src/common/common.cpp src/Domino/dominokernel.cuh src/Domino/dominokernel.cu AztecDiamondWaves/AztecDiamondAvHeight.cpp
	${NVCC} -o $@ src/common/common.cpp src/Domino/DominoTiler.cpp src/Domino/dominokernel.cu AztecDiamondWaves/AztecDiamondAvHeight.cpp ${CUDALINK}
endif

ifndef cuda
TilingTxtToSvg: common.o src/common/common.h file_reader.o DominoTiler.o src/Domino/DominoTiler.h 
	$(CXX) $(CXXFLAGS) -o  TilingTxtToSvg common.o file_reader.o DominoTiler.o AztecDiamondWaves/TilingTxtToSvg.cpp  -I$(srcdir) -L$(LDFLAGS) -l$(LDLIBS)  
else
TilingTxtToSvg: src/common/common.h src/Domino/DominoTiler.h src/Domino/DominoTiler.cpp src/common/common.cpp  AztecDiamondWaves/TilingTxtToSvg.cpp
	${NVCC} -o $@ src/common/common.cpp src/Domino/DominoTiler.cpp AztecDiamondWaves/TilingTxtToSvg ${CUDALINK}
endif

clean:
	rm *.o
