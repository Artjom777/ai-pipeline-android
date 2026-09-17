#ifndef DIFFUSION_MNN_HPP
#define DIFFUSION_MNN_HPP
#include "pipeline_common.hpp"
#include <memory>
#include <string>
#include <vector>
#include <cstdint>
#include <MNN/Interpreter.hpp>
#include <MNN/Tensor.hpp>
class DiffusionMNNPipeline {
public:
DiffusionMNNPipeline();
~DiffusionMNNPipeline();
int initialize(const std::string& modelDir, const std::string& unetPath, const std::string& vaePath, const std::string& textEncoderPath);
bool generateImage(const std::string& prompt, std::vector<uint8_t>& outRgb512, IProgressCallback* callback);
void releaseSessionAndOpenCL();
bool isInitialized() const;
private:
bool initialized;
std::string modelsPath;
int latentW;
int latentH;
int latentC;
std::unique_ptr<MNN::Interpreter> netText;
MNN::Session* sessionText;
std::unique_ptr<MNN::Interpreter> netUNet;
MNN::Session* sessionUNet;
std::unique_ptr<MNN::Interpreter> netVae;
MNN::Session* sessionVae;
std::string textModelPath;
std::string unetModelPath;
std::string vaeModelPath;
void tokenizePrompt(const std::string& prompt, std::vector<int32_t>& tokens);
bool runTextEncoder(const std::vector<int32_t>& tokens, std::vector<float>& context);
bool runUNetStep(const std::vector<float>& latents, float timestep, const std::vector<float>& context, std::vector<float>& noisePred);
bool runVaeDecoder(const std::vector<float>& latents, std::vector<uint8_t>& outRgb512, const std::string& prompt);
void runEulerALCMSchedulerStep(std::vector<float>& latents, const std::vector<float>& noisePred, int stepIndex, int totalSteps);
};
#endif
