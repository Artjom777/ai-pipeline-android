#ifndef UPSCALE_NCNN_HPP
#define UPSCALE_NCNN_HPP
#include "pipeline_common.hpp"
#include <string>
#include <vector>
class UpscaleNCNNPipeline {
public:
UpscaleNCNNPipeline();
~UpscaleNCNNPipeline();
bool initialize(const std::string& modelDir);
bool upscaleImage(const std::vector<uint8_t>& inRgba, int inW, int inH, std::vector<uint8_t>& outRgba, int targetW, int targetH, float remainingTimeSec, bool& usedFallback, IProgressCallback* callback);
void releaseVulkanInstance();
bool isInitialized() const;
private:
bool initialized;
std::string modelPath;
int tileSize;
int tileOverlap;
void processTile2x(const uint8_t* inImg, int inW, int inH, int tileX, int tileY, int tileW, int tileH, uint8_t* outImg, int outW, int outH, int scale);
};
#endif
