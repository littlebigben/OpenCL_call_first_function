#include <iostream>
#include <fstream>
#include <sstream>
#include <cassert>

#include <CL/cl.hpp>


void displayDevicesInfos(std::vector<cl::Device> &devices)
{
	//display devices name and OpenCL version
	for (unsigned int i = 0; i < devices.size(); ++i)
	{
		auto ret = devices[i].getInfo<CL_DEVICE_NAME>();
		std::cout << ret << std::endl;
		ret = devices[i].getInfo<CL_DEVICE_VERSION>();
		std::cout << ret << std::endl;
	}
}

int main()
{
	std::vector<cl::Device> devices;
	cl_int error = CL_SUCCESS;

	//get patforms
	std::vector<cl::Platform> platforms;
	cl_int  status = cl::Platform::get(&platforms);
	if (CL_SUCCESS != status)
    {
        std::cout << "Can't get platforms" << std::endl;
		return EXIT_FAILURE;
    }
	
	//get devices
	error = platforms[0].getDevices(CL_DEVICE_TYPE_GPU, &devices);
	
	assert(CL_SUCCESS == error);

	//device used for execution
	cl::Device device = devices[0];

	//initializing host memory
	float* h_src = new float;
	*h_src = 5.0f;

	std::cout << "Initial value: " << *h_src << std::endl;
	std::cout << "OpenCL kernel will add 1 to the value." << std::endl; 

	//get program sources
	size_t src_size = 0;
	const char* path = "./HostDeviceExchange.cl";
	
	std::ostringstream ostream;
	std::ifstream my_file(path);
	ostream << my_file.rdbuf();
	my_file.close();
	std::string fileContent = ostream.str();

	const char* source = fileContent.c_str();

	//creates queue and program
	cl::Program::Sources sources;
	sources.push_back(std::pair<const char*, ::size_t>(source, src_size));
	cl::Context context(device, NULL, NULL);
	assert(error == CL_SUCCESS);
	cl::Program program(context, sources);
	assert(CL_SUCCESS == error);

	//builds the program
	program.build();
	assert(CL_SUCCESS == error);

	//extracting the kernel
	cl::Kernel hostDeviceExchange(program, "HostDeviceExchangeSample");
	assert(CL_SUCCESS == error);

	//initializing device memory
	cl::Buffer d_src(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(float), h_src, &error);
	assert(CL_SUCCESS == error);
	cl::Buffer d_res(context, CL_MEM_WRITE_ONLY, sizeof(float), NULL, &error);
	assert(CL_SUCCESS == error);

	//enqueuing parameters
	error = hostDeviceExchange.setArg<cl::Buffer>(0, d_src);
	error = hostDeviceExchange.setArg<cl::Buffer>(1, d_res);
	assert(CL_SUCCESS == error);

	//launching kernel
	cl::CommandQueue queue(context, device, 0);
	assert(error == CL_SUCCESS);
	error = queue.enqueueNDRangeKernel(hostDeviceExchange, cl::NullRange, cl::NDRange(1), cl::NDRange(1));
	assert(CL_SUCCESS == error);

	//reading back
	float* result = new float;
	queue.enqueueReadBuffer(d_res, CL_TRUE, 0, sizeof(float), (void*)result);
	assert((*h_src + 1) == *result);

	std::cout << "Return value: " << *result << std::endl;

	delete h_src;
	delete result;

	//wait input to stop the program execution
	std::string tmp2;
	std::cin >> tmp2;
}