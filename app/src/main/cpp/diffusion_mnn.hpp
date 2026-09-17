#ifndef DIFFUSION_MNN_HPP
#define DIFFUSION_MNN_HPP
#include "pipeline_common.hpp"
#include <memory>
#include <string>
#include <vector>
class DiffusionMNNPipeline {
public:
DiffusionMNNPipeline();
~DiffusionMNNPipeline();
bool initialize(const std::string& modelDir);
bool generateImage(const std::string& prompt, std::vector<uint8_t>& outRgba512, IProgressCallback* callback);
void releaseSessionAndOpenCL();
bool isInitialized() const;
private:
bool initialized;
std::string modelsPath;
int latentW;
int latentH;
int latentC;
void runEulerALCMSchedulerStep(std::vector<float>& latents, const std::vector<float>& noisePred, int stepIndex, int totalSteps);
void decodeLatentsToRgba(const std::vector<float>& latents, std::vector<uint8_t>& outRgba);
};
#endif
