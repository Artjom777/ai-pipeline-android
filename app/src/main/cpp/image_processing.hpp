#ifndef IMAGE_PROCESSING_HPP
#define IMAGE_PROCESSING_HPP
#include <cstdint>
#include <vector>
void bicubicUpscaleRGBA(const uint8_t* src, int srcW, int srcH, uint8_t* dst, int dstW, int dstH);
void lanczosUpscaleRGBA(const uint8_t* src, int srcW, int srcH, uint8_t* dst, int dstW, int dstH);
#endif
