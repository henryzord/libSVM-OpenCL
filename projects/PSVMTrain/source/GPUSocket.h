#ifndef _GPUSOCKET_H
#define _GPUSOCKET_H

#include <CL\cl.h> //OpenCL

#include <string.h> //C string
#include <stdio.h> //C I/O
#include <stdlib.h> //c basic functions
#include <iostream> //C++ I/O
#include <string> //C++ string
#include <fstream> //C++ write in files 

#ifndef _LIBSVM_H
#include "psvm.h"
#endif

#define DIAGONAL 1
#define ONE_VERSUS_ALL 2
#define GET_INSTANCE 4

//#include <thread>

using namespace std;

class GPUSocket {

private:

	unsigned int gcd(unsigned int x, unsigned int y) {
		while (x * y != 0) {
			(x >= y)? x = x % y : y = y % x;
		}
		return (x + y);
	}

protected:

	cl_platform_id platform;
	cl_device_id *devices;
	cl_context context;
	cl_command_queue commandQueue;
	cl_kernel kernel;
	cl_program program;

	//memory objects
	cl_mem mem_image;
	cl_mem mem_readbuffer;

	point_t max;

	int count_attribute;

	//converts the kernel file into a string
	std::string convertKernelToString(const char *filename) {
		size_t size;
		char*  str;
		std::fstream f(filename, (std::fstream::in | std::fstream::binary));

		if (f.is_open()) {
			size_t fileSize;
			f.seekg(0, std::fstream::end);
			size = fileSize = (size_t)f.tellg();
			f.seekg(0, std::fstream::beg);
			str = new char[size + 1];
			/*
			if (!str) { 
			f.close();
			return;
			}*/
			f.read(str, fileSize);
			f.close();
			str[size] = '\0';
			return str;
		}
		std::string error = std::string("Error: failed to open file ") + std::string(filename);
		throw exception(error.c_str());
	}

	//return the OpenCL error strings, given an OpenCL error code.
	const inline char *getErrorMessage(cl_int err) {
		switch (err) {
		case 0: return "CL_SUCCESS";
		case -1: return "CL_DEVICE_NOT_FOUND";
		case -2: return "CL_DEVICE_NOT_AVAILABLE";
		case -3: return "CL_COMPILER_NOT_AVAILABLE";
		case -4: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
		case -5: return "CL_OUT_OF_RESOURCES";
		case -6: return "CL_OUT_OF_HOST_MEMORY";
		case -7: return "CL_PROFILING_INFO_NOT_AVAILABLE";
		case -8: return "CL_MEM_COPY_OVERLAP";
		case-9: return "CL_IMAGE_FORMAT_MISMATCH";
		case -10: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
		case -11: return "CL_BUILD_PROGRAM_FAILURE";
		case -12: return "CL_MAP_FAILURE";
		case -30: return "CL_INVALID_VALUE";
		case -31: return "CL_INVALID_DEVICE_TYPE";
		case -32: return "CL_INVALID_PLATFORM";
		case -33: return "CL_INVALID_DEVICE";
		case -34: return "CL_INVALID_CONTEXT";
		case -35: return "CL_INVALID_QUEUE_PROPERTIES";
		case -36: return "CL_INVALID_COMMAND_QUEUE";
		case -37: return "CL_INVALID_HOST_PTR";
		case -38: return "CL_INVALID_MEM_OBJECT";
		case -39: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
		case -40: return "CL_INVALID_IMAGE_SIZE";
		case -41: return "CL_INVALID_SAMPLER";
		case -42: return "CL_INVALID_BINARY";
		case -43: return "CL_INVALID_BUILD_OPTIONS";
		case -44: return "CL_INVALID_PROGRAM";
		case -45: return "CL_INVALID_PROGRAM_EXECUTABLE";
		case -46: return "CL_INVALID_KERNEL_NAME";
		case -47: return "CL_INVALID_KERNEL_DEFINITION";
		case -48: return "CL_INVALID_KERNEL";
		case -49: return "CL_INVALID_ARG_INDEX";
		case -50: return "CL_INVALID_ARG_VALUE";
		case -51: return "CL_INVALID_ARG_SIZE";
		case -52: return "CL_INVALID_KERNEL_ARGS";
		case -53: return "CL_INVALID_WORK_DIMENSION";
		case -54: return "CL_INVALID_WORK_GROUP_SIZE";
		case -55: return "CL_INVALID_WORK_ITEM_SIZE";
		case -56: return "CL_INVALID_GLOBAL_OFFSET";
		case -57: return "CL_INVALID_EVENT_WAIT_LIST";
		case -58: return "CL_INVALID_EVENT";
		case -59: return "CL_INVALID_OPERATION";
		case -60: return "CL_INVALID_GL_OBJECT";
		case -61: return "CL_INVALID_BUFFER_SIZE";
		case -62: return "CL_INVALID_MIP_LEVEL";
		case -63: return "CL_INVALID_GLOBAL_WORK_SIZE";
		default: return "Unknown OpenCL error";
		}
	}

	inline void handleErrorMessage(cl_int err) {
		if (err != CL_SUCCESS) {
			cout << getErrorMessage(err) << endl;
			throw exception(getErrorMessage(err));
		}
	}

	void buildProgram(const char *kernel_filename) {
		program = createProgram(kernel_filename);
		cl_int err = clBuildProgram(program, 1, devices, NULL, NULL, NULL);

		if(err != CL_SUCCESS) {
			size_t length;
			char buffer[8192]; //size OpenCL debug returning string
			cl_int err0 = clGetProgramBuildInfo(program, devices[0], CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &length);
			handleErrorMessage(err0);
			std::cout << "--- Build log ---" << std::endl << buffer << std::endl;
		}
		handleErrorMessage(err);
	}

	cl_uint getNumberOfPlatforms() {
		cl_uint numPlatforms;	//the NO. of platforms
		cl_int	err = clGetPlatformIDs(0, NULL, &numPlatforms);
		handleErrorMessage(err);
		return numPlatforms;
	}

	/**
	* Picks the first avaiable platform.
	* Documentation: "The host plus a collection 
	* of devices managed by the OpenCL framework 
	* that allow an application to share resources 
	* and execute kernels on devices in the platform." 
	*/
	void getPlatform() {
		cl_uint numberPlatforms = getNumberOfPlatforms();
		if (numberPlatforms > 0) {
			cl_platform_id* platforms = (cl_platform_id*)malloc(numberPlatforms * sizeof(cl_platform_id));
			cl_int err = clGetPlatformIDs(numberPlatforms, platforms, NULL);
			handleErrorMessage(err);
			platform = platforms[0];
			free(platforms);

			//write platform info.
			char answer[1024];
			clGetPlatformInfo(platform, CL_PLATFORM_VERSION, 1024, (void*)&answer, NULL);
			cout << "Device: " << answer << endl;

		} else {
			throw exception("No avaiable platform!");
		}
	}

	/**
	* Get an OpenCL device: GPU (first option), APU or CPU (either one).
	*/
	void getDevice() {
		cl_int err;
		cl_uint	gpuDevices, cpuDevices;

		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, NULL, &gpuDevices);
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, NULL, &cpuDevices);

		if (gpuDevices == 0) {
			devices = (cl_device_id*)malloc(cpuDevices * sizeof(cl_device_id));
			err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, cpuDevices, devices, NULL);
			handleErrorMessage(err);
		} else {
			devices = (cl_device_id*)malloc(gpuDevices * sizeof(cl_device_id));
			err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, gpuDevices, devices, NULL);
			handleErrorMessage(err);
		}	
	}

	void createContext() {
		cl_int err;
		context = clCreateContext(NULL, 1, devices, NULL, NULL, &err);
		handleErrorMessage(err);
	}

	void createCommandQueue() {
		cl_int err;
		commandQueue = clCreateCommandQueue(context, devices[0], 0, &err);
		handleErrorMessage(err);
	}

	/**
	* Creates the program.
	* filename - kernel filename
	*/
	cl_program createProgram(const char *filename) {
		string sourceStr = convertKernelToString(filename);

		cl_int err = CL_SUCCESS;
		const char *source = sourceStr.c_str();

		const cl_uint size = 1;
		size_t sourceSize[size] = { strlen(source) };

		cl_program program = clCreateProgramWithSource(context, size, &source, sourceSize, &err);
		handleErrorMessage(err);
		return program;
	}

public:

	/**
	* GPUSocket constructor.
	* Parameters:
	* kernel_filename - Name of kernel file.
	* program_name - Name of kernel function inside kernel file.
	*/
	GPUSocket(const char *kernel_filename, const char *program_name) {
		getPlatform();
		getDevice();

		createContext();
		createCommandQueue();
		buildProgram(kernel_filename);

		//cria kernel
		cl_int err;
		kernel = clCreateKernel(program, program_name, &err);
		handleErrorMessage(err);

		max = isImageCapable();
	}

	~GPUSocket() {
		cl_int err = CL_SUCCESS;

		err = clReleaseKernel(kernel); //Release kernel.
		handleErrorMessage(err);
		err = clReleaseProgram(program); //Release the program object.
		handleErrorMessage(err);
		err = clReleaseCommandQueue(commandQueue); //Release  Command queue.
		handleErrorMessage(err);
		err = clReleaseContext(context); //Release context.
		handleErrorMessage(err);

		clReleaseMemObject(mem_image);
		clReleaseMemObject(mem_readbuffer);

		if (devices != NULL) {
			delete devices;
		}
	}

	//queues a request to see if images are supported. If true, the max 
	//size of an image is returned; an exception is thrown otherwise.
	point_t isImageCapable() {
		cl_bool is_supported;
		cl_int err = CL_SUCCESS;

		err = clGetDeviceInfo(devices[0], CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), (void*)&is_supported, NULL);
		handleErrorMessage(err);

		if(!is_supported) {
			throw exception("Images are not supported by this device.");
		}

		point_t p;
		size_t size;
		err = clGetDeviceInfo(devices[0], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), (void*)&size, NULL);
		handleErrorMessage(err);
		p.x = (int)size;

		err = clGetDeviceInfo(devices[0], CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), (void*)&size, NULL);
		handleErrorMessage(err);
		p.y = (int)size;

		return p;
	}

	//get the total global memory size, in bytes.
	cl_ulong getAvaiableGlobalMemory() {
		cl_int err = CL_SUCCESS;

		cl_ulong size;
		err = clGetDeviceInfo(devices[0], CL_DEVICE_GLOBAL_MEM_SIZE, sizeof(cl_ulong), (void*)&size, NULL);
		handleErrorMessage(err);

		return size;
	}

	void createImage(int width, int height) {
		cl_int err = CL_SUCCESS;
		
		_cl_image_format format;
		format.image_channel_order = CL_RGBA;
		format.image_channel_data_type = CL_FLOAT;

		int 
			width_attribute = width + (4 - (width % 4)),
			width_pixels = width_attribute/4,
			height_pixels = height;

		this->count_attribute = width;

		size_t max_height, max_width;

		err = clGetDeviceInfo(devices[0], CL_DEVICE_IMAGE2D_MAX_HEIGHT, sizeof(size_t), &max_height, NULL);
		handleErrorMessage(err);
		err = clGetDeviceInfo(devices[0], CL_DEVICE_IMAGE2D_MAX_WIDTH, sizeof(size_t), &max_width, NULL);
		handleErrorMessage(err);

		if((width * height) > (max_width * max_height)) {
			throw exception("dataset bigger than gpu memory.\n");
		}

		if(width > (int)max_width) {
			throw exception("not enought space to store attributes!\n");
		}
		
		int column_width;

		if(height > (int)max_height) {
			column_width = (int)ceil((double)height / ((double)max_height));
			column_width *= width_attribute;
			width_pixels = column_width / 4;
			height = (int)max_height;
		} else {
			column_width = width_pixels;
		}

		mem_image = clCreateImage2D(context, CL_MEM_READ_ONLY, &format, column_width, height, NULL, NULL, &err); //4 for RGBA
		handleErrorMessage(err);

		mem_readbuffer = clCreateBuffer(context, CL_MEM_WRITE_ONLY, column_width * height * sizeof(float), NULL, &err);
		handleErrorMessage(err);
	}

	void writeDatasetToImage(svm_node **node, int height, int width, float *attribute_buffer) {
		int width_pixels = (width + (4 - (width % 4)))/4, width_values = (width + (4 - (width % 4)));

		for(int n = 0; n < height; n++) {
			svm_node *ahead = node[n];
			
			for(int z = 0; z < width_values; z++) {
				if(z == ahead->index-1) {
					attribute_buffer[z] = (float)ahead->value;
					ahead++;
				} else {
					attribute_buffer[z] = 0.f;
				}
			}

			size_t origin[3] = {0, n, 0}, region[3] = {width_pixels, 1, 1};

			cl_int err = clEnqueueWriteImage(commandQueue, mem_image, CL_TRUE, origin, region, width_values * sizeof(float), NULL, (void*)attribute_buffer, 0, NULL, NULL);
			handleErrorMessage(err);
		}
	}

	//write a single instance to gpu. width must be divisible by 4
	void writeInstanceToImage(float *values, int instance_width, int row) {
		if(instance_width % 4 != 0) {
			fprintf(stderr, "One instance has number of attributes not divisible by 4!\n");
		}

		int 
			gpu_column = (row/max.y) * (instance_width / 4), 
			gpu_row = row % max.y;

		//origin is where it begins; region is the length
		size_t origin[3] = {gpu_column, gpu_row, 0}, region[3] = {(instance_width/4), 1, 1}; 

		cl_int err = clEnqueueWriteImage(commandQueue, mem_image, CL_TRUE, origin, region, instance_width * sizeof(float), NULL, (void*)values, 0, NULL, NULL);
		handleErrorMessage(err);
	}

	void supportsImages() {
		cl_bool is_supported;
		size_t needed;
		clGetDeviceInfo(devices[0], CL_DEVICE_IMAGE_SUPPORT, sizeof(cl_bool), &is_supported, &needed);
		printf("Are images supported? %d\n", is_supported);
	}

	void run(int count_attribute, int count_instance, float *buffer, int pivot, int mode) {
		cl_int err = CL_SUCCESS;
		
		cl_mem mem_mode = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), (void*)&mode, &err);
		handleErrorMessage(err);
		cl_mem mem_pivot = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), (void*)&pivot, &err);
		handleErrorMessage(err);
		cl_mem mem_attribute = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), (void*)&count_attribute, &err);
		handleErrorMessage(err);
		cl_mem mem_instance = clCreateBuffer(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(int), (void*)&count_instance, &err);
		handleErrorMessage(err);

		err = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void*)&mem_mode);
		handleErrorMessage(err);

		err = clSetKernelArg(kernel, 1, sizeof(cl_mem), (void*)&mem_attribute);
		handleErrorMessage(err);

		err = clSetKernelArg(kernel, 2, sizeof(cl_mem), (void*)&mem_instance);
		handleErrorMessage(err);

		err = clSetKernelArg(kernel, 3, sizeof(cl_mem), (void*)&mem_pivot);
		handleErrorMessage(err);

		err = clSetKernelArg(kernel, 4, sizeof(cl_mem), (void*)&mem_image); 
		handleErrorMessage(err);

		err = clSetKernelArg(kernel, 5, sizeof(cl_mem), (void*)&mem_readbuffer);
		handleErrorMessage(err);

		const int work_dim = 1;

		size_t max_work_group_size;
		clGetDeviceInfo(devices[0], CL_DEVICE_MAX_WORK_GROUP_SIZE, sizeof(size_t), &max_work_group_size, NULL);
		int num_workgroups = (count_instance - 1) / (int)max_work_group_size + 1;
		size_t global_size_padded = num_workgroups * max_work_group_size;

		size_t 
			global_work_size[work_dim] = {global_size_padded},
			local_work_size[work_dim] = {NULL};

		cl_event event_run;

		err = clEnqueueNDRangeKernel(commandQueue, kernel, work_dim, NULL, global_work_size, NULL, 0, NULL, &event_run); 
		handleErrorMessage(err);

		err = clEnqueueReadBuffer(commandQueue, mem_readbuffer, CL_TRUE, 0, count_instance * sizeof(float), buffer, 1, &event_run, NULL);
		handleErrorMessage(err);

		err = clReleaseMemObject(mem_mode);
		handleErrorMessage(err);
		err = clReleaseMemObject(mem_pivot);
		handleErrorMessage(err);
	}
};

#endif //_GPUSOCKET_H