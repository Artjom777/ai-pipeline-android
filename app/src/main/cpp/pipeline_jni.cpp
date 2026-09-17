#include "pipeline_common.hpp"
#include "diffusion_mnn.hpp"
#include "upscale_ncnn.hpp"
#include <memory>
#include <cstring>
class JNIProgressBridge:public IProgressCallback{
public:
JNIProgressBridge(JNIEnv* env,jobject callbackObj):mEnv(env),mCallbackObj(callbackObj){
if(callbackObj){
jclass cls=env->GetObjectClass(callbackObj);
mMethodId=env->GetMethodID(cls,"onProgress","(IFLjava/lang/String;)V");
}else{
mMethodId=nullptr;
}
}
void onStageProgress(int stage,float progress,const char* statusMsg) override{
if(!mCallbackObj||!mMethodId)return;
jstring jMsg=mEnv->NewStringUTF(statusMsg?statusMsg:"");
mEnv->CallVoidMethod(mCallbackObj,mMethodId,static_cast<jint>(stage),static_cast<jfloat>(progress),jMsg);
mEnv->DeleteLocalRef(jMsg);
}
private:
JNIEnv* mEnv;
jobject mCallbackObj;
jmethodID mMethodId;
};
static std::unique_ptr<DiffusionMNNPipeline> sDiffusionPipeline;
static std::unique_ptr<UpscaleNCNNPipeline> sUpscalePipeline;
extern "C" JNIEXPORT jboolean JNICALL
Java_com_aipipe_app_NativePipelineBridge_nativeInit(JNIEnv* env,jobject thiz,jstring modelDir_){
const char* modelDir=env->GetStringUTFChars(modelDir_,nullptr);
LOGI("Native init requested with modelDir: %s", modelDir);
sDiffusionPipeline=std::make_unique<DiffusionMNNPipeline>();
sUpscalePipeline=std::make_unique<UpscaleNCNNPipeline>();
bool mnnOk=sDiffusionPipeline->initialize(modelDir);
bool ncnnOk=sUpscalePipeline->initialize(modelDir);
env->ReleaseStringUTFChars(modelDir_,modelDir);
LOGI("Native engines initialized. MNN OpenCL: %s, NCNN Vulkan: %s", mnnOk?"OK":"FAIL", ncnnOk?"OK":"FAIL");
return static_cast<jboolean>(mnnOk&&ncnnOk);
}
extern "C" JNIEXPORT jobject JNICALL
Java_com_aipipe_app_NativePipelineBridge_nativeExecutePipeline(JNIEnv* env,jobject thiz,jstring prompt_,jint timeoutSec,jint targetW,jint targetH,jobject callback,jobject outBitmap512,jobject outBitmap4K){
auto pipelineStart=std::chrono::steady_clock::now();
const char* prompt=env->GetStringUTFChars(prompt_,nullptr);
LOGI("Starting execution pipeline. Timeout: %ds, Target: %dx%d, Prompt: '%s'", timeoutSec, targetW, targetH, prompt);
JNIProgressBridge progressBridge(env,callback);
if(!sDiffusionPipeline)sDiffusionPipeline=std::make_unique<DiffusionMNNPipeline>();
if(!sUpscalePipeline)sUpscalePipeline=std::make_unique<UpscaleNCNNPipeline>();
ExecutionMetrics metrics;
std::vector<uint8_t> rgba512;
auto diffStart=std::chrono::steady_clock::now();
bool diffOk=sDiffusionPipeline->generateImage(prompt,rgba512,&progressBridge);
auto diffEnd=std::chrono::steady_clock::now();
metrics.diffusionDurationMs=std::chrono::duration<float,std::milli>(diffEnd-diffStart).count();
env->ReleaseStringUTFChars(prompt_,prompt);
if(!diffOk){
LOGE("Stage 1 diffusion failed on Mali OpenCL backend");
return nullptr;
}
if(outBitmap512){
void* pixels512=nullptr;
if(AndroidBitmap_lockPixels(env,outBitmap512,&pixels512)>=0&&pixels512){
std::memcpy(pixels512,rgba512.data(),rgba512.size());
AndroidBitmap_unlockPixels(env,outBitmap512);
LOGI("Locked and updated Stage 1 AndroidBitmap 512x512");
}
}
auto curTime=std::chrono::steady_clock::now();
float elapsedSecSoFar=std::chrono::duration<float>(curTime-pipelineStart).count();
float remainingBudgetSec=static_cast<float>(timeoutSec)-elapsedSecSoFar;
LOGI("Stage 1 completed in %.2f ms. Elapsed pipeline time: %.2f s. Remaining budget: %.2f s", metrics.diffusionDurationMs, elapsedSecSoFar, remainingBudgetSec);
std::vector<uint8_t> rgba4K;
bool usedFallback=false;
auto upStart=std::chrono::steady_clock::now();
bool upOk=sUpscalePipeline->upscaleImage(rgba512,512,512,rgba4K,targetW,targetH,remainingBudgetSec,usedFallback,&progressBridge);
auto upEnd=std::chrono::steady_clock::now();
metrics.upscaleDurationMs=std::chrono::duration<float,std::milli>(upEnd-upStart).count();
metrics.fallbackTriggered=usedFallback;
if(!upOk){
LOGE("Stage 2 upscale failed on Mali Vulkan backend");
return nullptr;
}
if(outBitmap4K){
void* pixels4K=nullptr;
if(AndroidBitmap_lockPixels(env,outBitmap4K,&pixels4K)>=0&&pixels4K){
size_t copyBytes=std::min(rgba4K.size(),static_cast<size_t>(targetW*targetH*4));
std::memcpy(pixels4K,rgba4K.data(),copyBytes);
AndroidBitmap_unlockPixels(env,outBitmap4K);
LOGI("Locked and updated Stage 2 AndroidBitmap %dx%d", targetW, targetH);
}
}
auto pipelineEnd=std::chrono::steady_clock::now();
metrics.totalDurationMs=std::chrono::duration<float,std::milli>(pipelineEnd-pipelineStart).count();
LOGI("Pipeline execution completed in %.2f ms (Budget: %d s, Budget Kept: %s). Fallback: %s", metrics.totalDurationMs, timeoutSec, (metrics.totalDurationMs<=timeoutSec*1000.0f)?"YES":"OVERDUE", usedFallback?"YES":"NO");
jclass resultClass=env->FindClass("com/aipipe/app/PipelineResult");
jmethodID constructor=env->GetMethodID(resultClass,"<init>","(FFFZZ)V");
jobject resultObj=env->NewObject(resultClass,constructor,metrics.diffusionDurationMs,metrics.upscaleDurationMs,metrics.totalDurationMs,static_cast<jboolean>(metrics.fallbackTriggered),static_cast<jboolean>(metrics.totalDurationMs<=timeoutSec*1000.0f));
return resultObj;
}
extern "C" JNIEXPORT void JNICALL
Java_com_aipipe_app_NativePipelineBridge_nativeRelease(JNIEnv* env,jobject thiz){
LOGI("Releasing native pipeline resources (MNN OpenCL + NCNN Vulkan)");
if(sDiffusionPipeline){
sDiffusionPipeline->releaseSessionAndOpenCL();
sDiffusionPipeline.reset();
}
if(sUpscalePipeline){
sUpscalePipeline->releaseVulkanInstance();
sUpscalePipeline.reset();
}
}
