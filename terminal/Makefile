CC = arm-linux-gnueabihf-gcc
CXX = arm-linux-gnueabihf-g++

CFLAGS = -O3 -Wall
CFLAGS += -Wl,-rpath-link,../../linux-devkit/sysroots/armv7ahf-neon-linux-gnueabi/usr/lib \
		  -Wl,-rpath-link,../../linux-devkit/sysroots/armv7ahf-neon-linux-gnueabi/lib \
		  -I../../linux-devkit/sysroots/armv7ahf-neon-linux-gnueabi/usr/include \
		  -I../../linux-devkit/sysroots/armv7ahf-neon-linux-gnueabi/usr/include/glib-2.0 \
		  -I../../linux-devkit/sysroots/armv7ahf-neon-linux-gnueabi/usr/lib/glib-2.0/include \
		  -L../../linux-devkit/sysroots/armv7ahf-neon-linux-gnueabi/usr/lib \
		  -L../../linux-devkit/sysroots/armv7ahf-neon-linux-gnueabi/lib 
CFLAGS += -lopencv_core -lopencv_highgui -lopencv_videoio -lopencv_imgproc -lopencv_imgcodecs -lopencv_video -lopencv_calib3d
CFLAGS += -lOpenCL -locl_util
CFLAGS += -lpthread -lasound -lrt

all:

	$(CXX)  -o main  main.cpp stereoRectified.cpp circle.cpp  $(CFLAGS)


clean:
	rm -rf main *.o
	rm -rf  circle *.o
	rm -rf stereoRectified *.o