
__kernel void HostDeviceExchangeSample (__global const float* src, __global float* res)
{
    *res = *src + 1;
}